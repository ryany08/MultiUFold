'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-18 14:36:46
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-25 17:42:52
FilePath: \Two-Stages_U-Net_MutlscaleCNN_Min-Cost_\main\utils.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch as t
import torch.nn.functional as F

from arnie.pk_predictors import _hungarian


def mask_diagonal(matrix, mask_value=0):
    matrix=matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix

# for testing and inference
# contact: (B, L, L)
# input arguements:
# contact: contact_matrix, in batch
# length: vector of length, in batch
# isSigmoid: boolean, whether to sigmoid the contact_matrix
# to transform the contact mateix straight to structure and bp_list by hungarian
def contact_2_dotbracket_and_bplist(contact, length, isSigmoid=False):
    result_bps = []
    result_structure = []
    if isSigmoid:
        contact = F.sigmoid(contact)
    batch = contact.size(0)
    threshold = 0.5
    for i in range(batch):
        next_contact = contact[i, :length[i], length[i]]
        s,bp=_hungarian(mask_diagonal(next_contact), theta=threshold, min_len_helix=1)
        result_bps.append(bp)
        result_structure.append(s)
    return result_bps, result_structure

# for testing and inference
# structure: string, representing the dot_bracket structure
# returning the bp_list stored in list    
def dotbracket_2_bp(structure):
    stack={'(':[],
           '[':[],
           '<':[],
           '{':[]}
    pop={')':'(',
         ']':'[',
         '>':"<",
         '}':'{'}       
    
    bp_list=[]

    for i,s in enumerate(structure):
        if s in stack:
            stack[s].append((i,s))
        elif s in pop:
            forward_bracket=stack[pop[s]].pop()
            bp_list.append([forward_bracket[0],i])

    return bp_list  

