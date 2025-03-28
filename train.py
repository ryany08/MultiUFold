'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-18 16:35:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-26 03:12:59
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\train.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch as t
import argparse
import collections
from utils.data_utils import load_config_from_yaml, get_data 
from utils.loss import PotentialLoss
from main.model import MainModel
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="config/configs.yaml")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config_path)

    epochs = config.epochs
    device = t.device("cuda: 0" if t.cuda.is_available() else "cpu")
    pos_weight = t.Tensor([300]).to(device)
    criterion = PotentialLoss(pos_weight)
    
    model = MainModel().to(device)
    optimizer = Ranger(model.parameters())

    train_dataloader, _, _ = get_data(config)
    
    steps_done = 0
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
    
            contacts_batch = t.Tensor(batch["contact"].float()).to(device)
            seq_embedding_batch = t.Tensor(batch["seq_rep"].float()).to(device)
            
            pred_contacts = model(seq_embedding_batch)
    
            seq_mask = t.Tensor(batch["seq_mask"].float()).to(device)
            
            contact_mask =  t.matmul(seq_mask, seq_mask.transpose(1, -1))
            # Compute loss
            loss_u = criterion(pred_contacts[contact_mask == 1], contacts_batch[contact_mask == 1])
    
            # Optimize the model
            optimizer.zero_grad()
            loss_u.backward()
            optimizer.step()
            steps_done = steps_done+1
    
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))

        if epoch > -1:
            t.save(model.state_dict(),  f'models/train/model_{epoch}.pt')

if __name__ == "__main__":
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact seq_raw length name bp_list')