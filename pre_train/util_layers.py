import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()


    def forward(self, x):
        return x.squeeze()
