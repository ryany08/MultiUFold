'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-18 16:35:16
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-26 03:13:29
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\test.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch as t
import numpy as np
import argparse
import collections
from utils.data_utils import load_config_from_yaml, get_data 
from main.model import MainModel
from main.postprocess import post_process

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="config/configs.yaml")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config_path)

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    model_dir = config.model_dir

    model = MainModel()
    model.load_state_dict(t.load(model_dir, map_location='cuda:0'))
    model.to(device)
    model.eval()

    _, val_data, test_data = get_data(config)

    precisions = []
    recalls = []
    f1_scores = []

    for batch in test_data:
        contacts_batch = t.Tensor(batch["contact"].float()).to(device)
        seq_embedding_batch = t.Tensor(batch["seq_rep"].float()).to(device)
        length = batch["length"]

        with t.no_grad():    
            pred_contacts = model(seq_embedding_batch)
    
        precision, recall, f1_score = post_process(pred_contacts, contacts_batch, length)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    print('Average testing F1 score with post-processing: ', np.average(f1_scores))
    print('Average testing precision with post-processing: ', np.average(precisions))
    print('Average testing recall with post-processing: ', np.average(recalls))


if __name__ == "__main__":
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact seq_raw length name bp_list')


    