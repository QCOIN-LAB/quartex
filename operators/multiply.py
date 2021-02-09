# -*- coding: utf-8 -*-

from operations import *

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        amp_x, pha_x = x
    
        re_x_ = torch.cos(pha_x) * amp_x
        im_x_ = torch.sin(pha_x) * amp_x

        return (re_x_, im_x_)