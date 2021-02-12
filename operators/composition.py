# -*- coding: utf-8 -*-

from operations import *

class Composition(nn.Module):
    def __init__(self):
        super(Composition, self).__init__()

    def forward(self, x, y):
        re_x, im_x = x
        x_len = re_x.shape[-2]
        re_y, im_y = y
        y_len = re_y.shape[-2]

        re_x, im_x = re_x.unsqueeze(-1), im_x.unsqueeze(-1)
        re_x = re_x.unsqueeze(-3).expand(-1, -1, y_len, -1, -1)
        im_x = im_x.unsqueeze(-3).expand(-1, -1, y_len, -1, -1)
        re_y, im_y = re_y.unsqueeze(-1), im_y.unsqueeze(-1)
        re_y = re_y.unsqueeze(-4).expand(-1, x_len, -1, -1, -1)
        im_y = im_y.unsqueeze(-4).expand(-1, x_len, -1, -1, -1)

        re_z = torch.matmul(re_x, re_y.transpose(-2, -1)) \
            + torch.matmul(im_x, im_y.transpose(-2, -1))  
        im_z = torch.matmul(im_x, re_y.transpose(-2, -1)) \
            - torch.matmul(re_x, im_y.transpose(-2, -1))

        return (re_z.flatten(-2, -1), im_z.flatten(-2, -1))