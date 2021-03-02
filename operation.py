# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# out-of-place operations

def embedding(x, embed_param):
    amp_embed = embed_param[0](x)
    pha_embed = embed_param[1](x)
    mix_embed = embed_param[2](x)

    return (amp_embed, pha_embed), mix_embed

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

    re_x_, im_x_ = re_x.unsqueeze(-1), im_x.unsqueeze(-1)

    re_x = torch.matmul(re_x_, re_x_.transpose(-2, -1)) \
        + torch.matmul(im_x_, im_x_.transpose(-2, -1))  
    im_x = torch.matmul(im_x_, re_x_.transpose(-2, -1)) \
        - torch.matmul(re_x_, im_x_.transpose(-2, -1))

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
        re_x = torch.mean(re_x, dim=-3, keepdim=True)
        im_x = torch.mean(im_x, dim=-3, keepdim=True) 
    else:
        weight = weight.unsqueeze(-1)
        re_x = torch.sum(re_x * weight, dim=-3, keepdim=True)
        im_x = torch.sum(im_x * weight, dim=-3, keepdim=True)
    
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

    op = multiply(op)
    re_op, im_op = density(op)
    
    # only real part is non-zero
    p = torch.matmul(re_x.flatten(-2, -1), re_op.flatten(-2, -1).t()) \
        - torch.matmul(im_x.flatten(-2, -1), im_op.flatten(-2, -1).t())
    
    approx_one_hot = gumble_softmax(p, dim=-1)
    collapsed_re_x = torch.einsum('bse,emn->bsmn', approx_one_hot, re_op)
    collapsed_im_x = torch.einsum('bse,emn->bsmn', approx_one_hot, im_op)
    
    return p, (collapsed_re_x, collapsed_im_x)

def gumble_softmax(x, dim, temperature=0.1, force_hard=True):     
    _, max_idx = x.max(dim, keepdim=True)
    x_hard = torch.zeros_like(x).scatter_(dim, max_idx, 1.0)
    
    gumble_noise = torch.zeros_like(x).uniform_()
    gumble_noise = - torch.log(1e-7 - torch.log(gumble_noise + 1e-7))
    x = F.softmax((torch.log(x + 1e-7) + gumble_noise) / temperature, dim=dim)

    if force_hard:
        return x_hard - x.detach() + x
    else:
        return x

