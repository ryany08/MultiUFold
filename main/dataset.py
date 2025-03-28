'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-09 10:53:10
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-26 03:11:24
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\main\dataset.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import pickle
import collections
from os.path import join
from random import shuffle
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from typing import List
from multimolecule import RnaTokenizer, RnaFmModel

class Dataset(Dataset):
    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths
        self.tokenizer = RnaTokenizer.from_pretrained("pre_model/rnafm", add_pooling_layer=False)
        self.model = RnaFmModel.from_pretrained("pre_model/rnafm", add_pooling_layer=False)
        self.nt_dict = {char: idx for idx, char in enumerate("ACGU")}
        self.extra_dict = {
            "R": ["G", "A"],
            "Y": ["C", "U"],
            "K": ["G", "U"],
            "M": ["A", "C"],
            "S": ["G", "C"],
            "W": ["A", "U"],
            "B": ["G", "U", "C"],
            "D": ["G", "A", "U"],
            "H": ["A", "C", "U"],
            "V": ["G", "C", "A"],
            "N": ["G", "A", "C", "U"],
        }
        self.nt_num = 4
        self.data = []
        if len(paths) > 1:
            for path in self.paths:
                for data in self.load_data(path):
                    self.data.append(data)
        elif len(paths) == 1:
            self.data = self.load_data(paths[0])
        else:
            ValueError("Path can not be null")
        self.contact, self.sequence, self.length, self.name = self.process_data()

    def load_data(self, path):
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact seq_raw length name bp_list')
        with open(path, 'rb') as f:
            load_data = pickle.load(f)
        return load_data
    
    def process_data(self):
        shuffle(self.data)
        contact_list = [item.contact for item in self.data]
        data_seq_raw_list = [item.seq_raw for item in self.data]
        data_length_list = [item.length for item in self.data]
        data_name_list = [item.name for item in self.data]

        contact_array = np.stack(contact_list, axis=0)

        return contact_array, data_seq_raw_list, data_length_list, data_name_list


    # pre_dim is for tokenized sequence from RNAMSM
    def encode_sequence(self, seq):
        emb = t.zero((self.nt_num, len(seq)), dtype=t.float32)
        for k, nt in enumerate(seq):
            if nt in self.nt_dict:
                emb[self.nt_dict[nt], k] = 1
            elif nt in self.extra_dict:
                ind = [self.nt_dict.index(n) for n in self.extra_dict[nt]]
                emb[ind, k] = 1
            else:
                continue
        input = self.tokenizer(seq, return_tensors="pt")
        output = self.model(**input, output_attentions=True)
        # to get the encoded sequence
        hidden_state = output["last_hidden_state"].squeeze(0)[1:-1].transpose(0, 1)
        hidden_state = t.concat((emb, hidden_state), 0)
        # hidden_state: (C, L)
        return hidden_state
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):
        contact = self.contact[idx]
        sequence = self.sequence[idx]
        length = self.length[idx]
        seq_mask = t.ones(length, dtype=t.float32)

        encoded_sequence = self.encode_sequence(sequence)

        return {
            "sequence": encoded_sequence, 
            # sequence: (C, L)
            "contact": contact, 
            # contact: (L, L)
            "seq_mask": seq_mask,
            # seq_mask: (1, L)
            "length": length
        }
    
    