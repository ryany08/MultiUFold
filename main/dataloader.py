'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-09 10:53:18
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-19 16:17:47
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\main\dataloader.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import os
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset




def collate_fn(batch_data):
    # max_length in the batch
    max_length = max(data_dict["sequence"].size(1) for data_dict in batch_data)

    sequence = []
    seq_mask = []
    contact = []
    length = []

    for i in range(len(batch_data)):
        to_pad = max_length - batch_data[i]["length"]
        sequence.append(F.pad(batch_data[i]["sequence"], (0, to_pad, 0, 0), value=0))
        seq_mask.append(F.pad(batch_data[i]["seq_mask"], (0, to_pad), value=0))
        contact.append(F.pad(batch_data[i]["contact"], (0, to_pad, 0, to_pad), value=0))
        length.append(batch_data[i]["length"])

    # sequence: (B, C, L)
    sequence = t.stack(sequence)
    # seq_mask: (B, L, 1)
    seq_mask = t.stack(seq_mask).unsqueeze(-1)
    # contact: (B, L, L)
    contact = t.stack(contact)
    # length: (B)
    length = t.tensor(length, dtype=t.float32)
    # contact_mask: (B, L, L)
    contact_mask = t.matmul(seq_mask, seq_mask.transpose(1, -1))

    return {
        "seq_rep": sequence,
        "seq_mask": seq_mask,
        "contact": contact,
        "contact_mask": contact_mask,
        "length": length
    }