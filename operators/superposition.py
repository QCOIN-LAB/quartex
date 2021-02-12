# -*- coding: utf-8 -*-

from operations import *

class Superposition(nn.Module):
    def __init__(self):
        super(Superposition, self).__init__()

    def forward(self, x, weight=None):
        re_x, im_x = x

        if weight is None:
            re_x = torch.mean(re_x, dim=-2)
            im_x = torch.mean(im_x, dim=-2) 
        else:
            weight = weight.unsqueeze(-1)
            re_x = torch.sum(re_x * weight, dim=-2)
            im_x = torch.sum(im_x * weight, dim=-2)
        
        return (re_x, im_x)