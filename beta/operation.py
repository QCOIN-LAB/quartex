# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# out-of-place operations

def embedding(x, embed_param, mix_param):
    batch_size, x_len = x.shape
    x = x.flatten(-2, -1)
    embed = torch.index_select(embed_param, 0, x).reshape(batch_size, x_len, -1)
    mix = torch.index_select(mix_param, 0, x).reshape(batch_size, x_len, -1)
    #embed = F.embedding(x, embed_param, padding_idx=0)
    #mix = F.embedding(x, mix_param, padding_idx=0)

    return embed, mix

def normalize(x):
    norm = torch.norm(x, p=2, dim=-1)
    x = F.normalize(x, p=2, dim=-1)

    return x, norm

def multiply(x):
    amp_x, pha_x = x
    
    re_x = torch.cos(pha_x) * amp_x
    im_x = torch.sin(pha_x) * amp_x

    return x

def density(x):
    x = x.unsqueeze(-1)
    x = torch.matmul(x, x.transpose(-2, -1))

    return x

def composition(x, y):
    x_len = x.shape[-2]
    y_len = y.shape[-2]

    x = x.unsqueeze(-1)
    x = x.unsqueeze(-3).expand(-1, -1, y_len, -1, -1)
    y = y.unsqueeze(-1)
    y = y.unsqueeze(-4).expand(-1, x_len, -1, -1, -1)

    z = torch.matmul(x, y.transpose(-2, -1))

    return z.flatten(-2, -1)

def superposition(x, weight=None):
    if weight is None:
        x = torch.mean(x, dim=-2)
    else:
        weight = weight.unsqueeze(-1)
        x = torch.sum(x * weight, dim=-2)
    
    return x

def mixture(x, weight=None):
    if weight is None:
        x = torch.mean(x, dim=-3, keepdim=True)
    else:
        weight = weight.unsqueeze(-1)
        x = torch.sum(x * weight, dim=-3, keepdim=True)
    
    return x

def n_gram(x, n=3):
    batch_size, x_len = x.shape

    pad_len = n - 1
    left_pad_len = pad_len // 2
    right_pad_len = pad_len - left_pad_len
    left_pad_zeros = torch.zeros(batch_size, left_pad_len).to(x.device)
    right_pad_zeros = torch.zeros(batch_size, right_pad_len).to(x.device)
    x = torch.cat([left_pad_zeros, x, right_pad_zeros], dim=-1)

    ngrams = []
    slice_begin_index = 0
    slice_end_index = -1
    for i in range(x_len):
        slice_begin_index = i
        slice_end_index = i + n
        slice_indices = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long).to(x.device)
        ngram = torch.index_select(x, -1, index=slice_indices)
        ngrams.append(ngram)
            
    ngram_mat = torch.stack(ngrams, dim=-2)
    
    return ngram_mat

def measurement(x, measurement_param, collapse=True):
    measurement_desity = density(measurement_param)
    
    # only real part is non-zero
    p = torch.matmul(x.flatten(-2, -1), measurement_desity.flatten(-2, -1).t()).real
    
    if collapse: # collapse
        approx_one_hot = gumble_softmax(p, dim=-1)
        post_x = torch.einsum('bse,emn->bsmn', torch.complex(approx_one_hot, torch.zeros_like(approx_one_hot)), measurement_desity)
    else: # post mixture
        post_x = torch.einsum('bse,emn->bsmn', torch.complex(p, torch.zeros_like(p)), measurement_desity)
        
    return p, post_x

def gumble_softmax(x, dim, temperature=0.1, force_hard=True): 
    #x = F.normalize(x, p=1, dim=-1)    
    _, max_idx = x.max(dim, keepdim=True)
    x_hard = torch.zeros_like(x).scatter_(dim, max_idx, 1.0)
    
    gumble_noise = torch.zeros_like(x).uniform_()
    gumble_noise = - torch.log(1e-7 - torch.log(gumble_noise + 1e-7))
    x = F.softmax((torch.log(x + 1e-7) + gumble_noise) / temperature, dim=dim)

    if force_hard:
        return x_hard - x.detach() + x
    else:
        return x

