import yaml
import sys
import collections
import torch as t
import numpy as np

sys.path.append("..")

import pickle
# from main.dataset import Dataset
# from torch.utils.data import DataLoader
from os.path import join
from main.dataloader import collate_fn


ROOT_PATH = "./data"

dataset_choices = ["RNAStrAlign", "bpRNA", "PDBnew", "all"]

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)


# def get_data(config):
#     assert config.dataset in dataset_choices

#     if config.dataset == 'RNAStrAlign':
#         train = Dataset([join(ROOT_PATH, config.dataset, 'train')], upsampling=True)
#         val = Dataset([join(ROOT_PATH, config.dataset, 'val')])
#         test = Dataset([join(ROOT_PATH, config.dataset, 'test')])

#     elif config.dataset == 'bpRNA':
#         train = Dataset([join(ROOT_PATH, config.dataset, 'TR0')], upsampling=True)
#         val = Dataset([join(ROOT_PATH, config.dataset, 'VL0')])
#         test = Dataset([join(ROOT_PATH, config.dataset, 'TS0')])

#     elif config.dataset == 'PDBnew':
#         train = Dataset([join(ROOT_PATH, config.dataset, 'TR1')], upsampling=True)
#         val = Dataset([join(ROOT_PATH, config.dataset, 'VL1')])
#         test = Dataset([join(ROOT_PATH, config.dataset, 'TS1'),
#                         join(ROOT_PATH, config.dataset, 'TS2'),
#                         join(ROOT_PATH, config.dataset, 'TS3')
#                         ])

#     elif config.dataset == 'all':
#         train = Dataset([join(ROOT_PATH, 'RNAStrAlign/train'),
#                          join(ROOT_PATH, 'bpRNA/TR0/')], upsampling=True)
#         val = Dataset([join(ROOT_PATH, 'bpRNA/VL0/'),
#                        join(ROOT_PATH, 'RNAStrAlign/val')])
#         test = Dataset([join(ROOT_PATH, 'bpRNA/TS0/'),
#                         join(ROOT_PATH, 'RNAStrAlign/test'),
#                         join(ROOT_PATH, 'bpRNAnew/bpRNAnew')])

#     else:
#         raise NotImplementedError

#     train_loader = DataLoader(train,
#                               batch_size=config.train_batch_size,
#                               shuffle=True,
#                               num_workers=config.num_workers,
#                               collate_fn=collate_fn,
#                               pin_memory=config.pin_memory,
#                               drop_last=True)

#     val_loader = DataLoader(val,
#                             batch_size=config.val_batch_size,
#                             shuffle=True,
#                             num_workers=config.num_workers,
#                             collate_fn=collate_fn,
#                             pin_memory=config.pin_memory,
#                             drop_last=False)

#     test_loader = DataLoader(test,
#                              batch_size=config.test_batch_size,
#                              shuffle=True,
#                              num_workers=config.num_workers,
#                              collate_fn=collate_fn,
#                              pin_memory=config.pin_memory,
#                              drop_last=False)

#     return train_loader, val_loader, test_loader

def pre_process_data(dir):
    with open(dir, 'rb') as file:
        load_file = pickle.load(file)
        print(load_file[0])
        new_list = []
        for i in load_file:
            contact = np.zeros((i.length, i.length))
            for pair in i.bp_list:
                if pair[0] >= i.length or pair[1] >= i.length:
                    break
                contact[pair[0], pair[1]] = 1
                new_list.append(RNA_SS_data(contact, i.seq_raw, i.length, i.name, i.bp_list))
        print(new_list[0])
        
        with open(dir, 'wb') as file2:
            pickle.dump(new_list, file2)
        

if __name__ == "__main__":
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact seq_raw length name bp_list')