'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-09 10:52:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-19 22:19:13
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\main\model.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from modules import TransformerEncoder, Unet, OuterProduct

class MainModel(nn.Module):
    def __init__(self, pre_channels=644, in_channels=256, hid_channels=64, out_channels=1, drop_out=0.05, init_alpha=0.5, n_encoder_layers=6, head_num=8):
        super().__init__()
        self.preprocess = nn.Linear(pre_channels, in_channels)
        self.encoders = nn.Sequential(
            TransformerEncoder(in_channels, in_channels, head_num, drop_out, init_alpha, feed_para=4)
            for _ in range(n_encoder_layers)
        )
        self.outer_product = OuterProduct(in_channels, hid_channels, hid_channels)
        self.unet = Unet(hid_channels, out_channels)
    
    def forward(self, src, src_mask, pair_rep):
        # pair_rep: For add extra info to image-like data representation
        src = self.encoders(src, src_mask)
        pair_rep = self.outer_product(src, pair_rep)
        pair_rep = self.unet(pair_rep)
        return pair_rep
