# -*- coding: utf-8 -*-

from operations import *

class Density(nn.Module):
    def __init__(self):
        super(Density, self).__init__()

    def forward(self, x):
        re_x, im_x = x

        re_x_, im_x_ = re_x.unsqueeze(-1), im_x.unsqueeze(-1)

        re_x_ = torch.matmul(re_x_, re_x_.transpose(-2, -1)) \
            + torch.matmul(im_x, im_x.transpose(-2, -1))  
        im_x_ = torch.matmul(im_x_, re_x_.transpose(-2, -1)) \
            - torch.matmul(re_x_, im_x_.transpose(-2, -1))

        return (re_x_, im_x_)