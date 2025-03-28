import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from einops import rearrange

def get_ALiBi(head_num, length):
    slopes = t.tensor([2 ** (-8 * i / head_num) for i in range(head_num)])
    slopes = slopes.unsqueeze(1).unsqueeze(1)  # Shape: (num_heads, 1, 1)
        
    positions = t.arange(length).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, max_seq_len)
    alibi_bias = -positions * slopes  # Shape: (num_heads, 1, max_seq_len)
        
    return alibi_bias    

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, head_num, drop_out) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.head_num = head_num
        self.head_channels = hidden_channels // head_num

        self.q_proj = nn.Linear(self.in_channels, self.hidden_channels)
        self.k_proj = nn.Linear(self.in_channels, self.hidden_channels)
        self.v_proj = nn.Linear(self.in_channels, self.hidden_channels)
        self.o_proj = nn.Linear(self.hidden_channels, self.out_channels)

        self.drop_out = nn.Dropout(drop_out)

    # x: (B, L, in_C)
    def forward(self, x, seq_mask):
        batch_size, length, in_channels = x.size()

        attention_masks = None
        if seq_mask is not None:
            attention_masks = t.matmul(seq_mask, seq_mask.transpose(1, -1))

        Q_v = self.q_proj(x)
        # Q_v: (B, L, hid_C)
        K_v = self.k_proj(x)
        # K_v: (B, L, hid_C)
        V_v = self.v_proj(x)
        # V_v: (B, L, hid_C)

        Q_state = Q_v.view(batch_size, length, self.head_num, self.head_channels).transpose(1, 2)
        # Q_state: (B, Head_N, L, Head_C)
        K_state = K_v.view(batch_size, length, self.head_num, self.head_channels).transpose(1, 2)
        # K_state: (B, Head_N, L, Head_C)
        V_state = V_v.view(batch_size, length, self.head_num, self.head_channels).transpose(1, 2)
        # V_state: (B, Head_N, L, Head_C)

        attention_weights = t.matmul(Q_state, K_state.transpose(2, -1)) / math.sqrt(self.head_channels)

        alibi_bias = get_ALiBi(self.head_num, length)

        attention_masks += alibi_bias

        if attention_masks is not None:
            attention_weights = attention_weights.masked_fill(attention_masks == 0, float('-inf'))

        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = self.drop_out(attention_weights)

        output = t.matmul(attention_weights, V_state)
        # output: (B, Head_N, L, Head_C)
        output = output.transpose(1, 2).contiguous()
        # output: (B, L, Head_N, Head_C)
        output = output.view(batch_size, length, -1)
        # output: (B, L, Hid_C)
        output = self.o_proj(output)
        # output: (B, L, Out_C)

        return output


class FeedForward(nn.Module):
    def __init__(self, in_channels, feed_channels, drop_out):
        self.in_proj = nn.Linear(in_channels, feed_channels)
        self.activate = F.relu()
        self.dropout = nn.Dropout(drop_out)
        self.out_proj = nn.Linear(feed_channels, in_channels)

    # x: (B, L, C)
    def forward(self, x):
        output = self.in_proj(x)
        output = self.activate(output)
        output = self.dropout(output)
        output = self.out_proj(output)

        return output

class ELA(nn.Module):
    def __init__(self, in_channels, kernal_size):
        super().__init__()
        self.in_channels = in_channels
        self.padding = kernal_size // 2
        self.conv = nn.Conv1d(self.in_channels, self.in_channels, kernal_size, padding=self.padding, groups=self.in_channels, bias=False)
        self.gn = nn.GroupNorm(16, self.in_channels)
        self.sig = F.sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_h = t.mean(x, dim=3, keepdim=True).view(batch_size, channel, height)
        x_w = t.mean(x, dim=2, keepdim=True).view(batch_size, channel, width)
        x_h = self.sig(self.gn(self.conv(x_h))).view(batch_size, channel, height, 1)
        x_w = self.sig(self.gn(self.conv(x_w))).view(batch_size, channel, 1, width)

        return x * x_h * x_w
    
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(t.cat([x_h, x_w], dim=2))
        x_h, x_w = t.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (t.matmul(x11, x12) + t.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        return self.conv(x)
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out):
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, drop_out)
        )
        self.ema = EMA(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x + self.ema(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.GroupNorm(1, out_channels),
			nn.ReLU(inplace=True),
            nn.Dropout(drop_out)
        )
        self.conv = DoubleConv(in_channels, out_channels, drop_out)
        self.ema = EMA(out_channels)
        
    def forward(self, x1, x2):
        x = self.up_conv(x1)
        x = t.cat((x2, x), dim=1)
        x = self.conv(x)
        return x + self.ema(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        self.conv1 = DoubleConv(in_channels, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)
        self.down5 = DownConv(512, 1024)

        self.up5 = UpConv(1024, 512)
        self.up4 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: (B, in_C, L, L)
        x = self.conv1(x)
        # x: (B, 64, L, L)
        x1 = self.down2(x)
        # x1: (B, 128, L, L)
        x2  = self.down3(x1)
        # x2: (B, 256, L, L)
        x3 = self.down4(x2)
        # x3: (B, 512, L, L)
        x4 = self.down5(x3)
        # x4: (B, 1024, L, L)

        x5 = self.up5(x4, x3)
        # x5: (B, 512, L, L)
        x6 = self.up4(x5, x2)
        # x6: (B, 256, L, L)
        x7 = self.up3(x6, x1)
        # x7: (B, 128, L, L)
        x8 = self.up2(x7, x) 
        # x8: (B, 64, L, L)
        out = self.out_conv(x8)
        # out: (B, 1, L, L)
        out = (out + out.transpose(2, -1)) // 2 

        return out.squeeze(1)
        # out: (B, L, L)

class OuterProduct(nn.Module):
    def __init__(self, in_channels=256, hid_channels=32, pairwise_chanels=64):
        super().__init__()
        self.proj_down1 = nn.Linear(in_channels, hid_channels)
        self.proj_down2 = nn.Linear(hid_channels ** 2, pairwise_chanels)

    def forward(self, seq_rep, pair_rep=None):
        seq_rep=self.proj_down1(seq_rep)
        outer_product = t.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product=outer_product + pair_rep

        return outer_product 

class DyT(nn.Module):
    def __init__(self, hidden_channels, init_alpha):
        super().__init__()
        self.alpha = nn.Parameter(t.ones(1) * init_alpha)
        self.beta = nn.Parameter(t.ones(hidden_channels))
        self.gamma = nn.Parameter(t.ones(hidden_channels))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta 

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, head_num, drop_out, init_alpha, feed_para):
        super().__init__()
        self.atten = MultiHeadAttention(in_channels, out_channels, in_channels, head_num, drop_out)
        self.dyt1 = DyT(in_channels, init_alpha)
        self.feedforward = FeedForward(in_channels, feed_para * in_channels, drop_out)
        self.dyt2 = DyT(in_channels, init_alpha)

    def forward(self, x):
        x1 = self.atten(x)
        x1 = self.dyt1(x1)
        x = x + x1
        x2 = self.feedforward(x)
        x2 = self.dyt2(x2)
        x = x + x2
        return x
    


        







