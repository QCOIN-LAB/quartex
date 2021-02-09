# -*- coding: utf-8 -*-

from operations import *

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        norm = torch.norm(x, dim=-1)
        x_ = F.normalize(x, p=2, dim=-1)

        return x_, norm