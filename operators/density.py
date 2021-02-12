# -*- coding: utf-8 -*-

from operations import *

class Density(nn.Module):
    def __init__(self):
        super(Density, self).__init__()

    def forward(self, x):
        re_x, im_x = x

        re_x, im_x = re_x.unsqueeze(-1), im_x.unsqueeze(-1)

        re_x = torch.matmul(re_x, re_x.transpose(-2, -1)) \
            + torch.matmul(im_x, im_x.transpose(-2, -1))  
        im_x = torch.matmul(im_x, re_x.transpose(-2, -1)) \
            - torch.matmul(re_x, im_x.transpose(-2, -1))

        return (re_x, im_x)