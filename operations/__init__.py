# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# out-of-place operations

def normalize(x):
    norm = torch.norm(x, dim=-1)
    x = F.normalize(x, p=2, dim=-1)

    return x, norm

def multiply(x):
    amp_x, pha_x = x
    
    re_x = torch.cos(pha_x) * amp_x
    im_x = torch.sin(pha_x) * amp_x

    return (re_x, im_x)

def density(x):
    re_x, im_x = x

    re_x, im_x = re_x.unsqueeze(-1), im_x.unsqueeze(-1)

    re_x = torch.matmul(re_x, re_x.transpose(-2, -1)) \
        + torch.matmul(im_x, im_x.transpose(-2, -1))  
    im_x = torch.matmul(im_x, re_x.transpose(-2, -1)) \
        - torch.matmul(re_x, im_x.transpose(-2, -1))

    return (re_x, im_x)

def composition(x, y):
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

def superposition(x, weight=None):
        re_x, im_x = x

        if weight is None:
            re_x = torch.mean(re_x, dim=-2)
            im_x = torch.mean(im_x, dim=-2) 
        else:
            weight = weight.unsqueeze(-1)
            re_x = torch.sum(re_x * weight, dim=-2)
            im_x = torch.sum(im_x * weight, dim=-2)
        
        return (re_x, im_x)

def mixture(x, weight=None):
        re_x, im_x = x

        if weight is None:
            re_x = torch.mean(re_x, dim=-3)
            im_x = torch.mean(im_x, dim=-3) 
        else:
            weight = weight.unsqueeze(-1).unsqueeze(-1)
            re_x = torch.sum(re_x * weight, dim=-3)
            im_x = torch.sum(im_x * weight, dim=-3)
        
        return (re_x, im_x)

def n_gram(x, n=3):
    batch_size, seq_len = x.shape

    pad_len = n - 1
    left_pad_len = pad_len // 2
    right_pad_len = pad_len - left_pad_len
    left_pad_zeros = torch.zeros(batch_size, left_pad_len).to(x.device)
    right_pad_zeros = torch.zeros(batch_size, right_pad_len).to(x.device)
    x = torch.cat([left_pad_zeros, x, right_pad_zeros], dim=-1)

    ngrams = []
    slice_begin_index = 0
    slice_end_index = -1
    for i in range(seq_len):
        slice_begin_index = i
        slice_end_index = i + n
        slice_indices = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long).to(x.device)
        ngram = torch.index_select(x, -1, index=slice_indices)
        ngrams.append(ngram)
            
    ngram_mat = torch.stack(ngrams, dim=-2)
    
    return ngram_mat

def measurement(x, op):
    re_x, im_x = x
    
    amp_op, pha_op = op
    amp_op, _ = normalize(amp_op)

    re_op, im_op = multiply((amp_op, pha_op))
    re_op, im_op = density((re_op, im_op))
    
    # only real part is non-zero
    out = torch.matmul(re_x.flatten(-2, -1), re_op.flatten(-2, -1).t()) \
        - torch.matmul(im_x.flatten(-2, -1), im_op.flatten(-2, -1).t())
    
    return out