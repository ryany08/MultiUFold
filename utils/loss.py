import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# using cross-entrophy to calculate
class PotentialLoss(nn.Module):
    def __init__(self, weight):
        self.loss = nn.BCEWithLogitsLoss(weight)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)