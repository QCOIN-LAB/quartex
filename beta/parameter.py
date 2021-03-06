# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

import operation as op
from utils.constraint import ConstrainedParameter, UnitNormConstraint, RangeConstraint


def embedding_param(token2id, from_path='./glove/glove.6B.50d.txt', embed_size=50, freeze=False):
    fin = open(from_path, 'r', encoding='utf-8', errors='ignore')
    embed_mat = np.random.uniform(-1/np.sqrt(embed_size), 1/np.sqrt(embed_size), (len(token2id), embed_size))
    embed_mat[0, :] = np.zeros((1, embed_size)) # [pad]
    for line in fin:
        elements = line.rstrip().split()
        token, vec = ' '.join(elements[:-embed_size]), elements[-embed_size:]
        if token in token2id:
            embed_mat[token2id[token]] = np.asarray(vec, dtype=np.float32)
    fin.close()

    embed_mat = torch.from_numpy(embed_mat).float()
    re_embed, mix = op.normalize(embed_mat)
    im_embed = torch.zeros_like(re_embed)

    embed_param = ConstrainedParameter(torch.complex(re_embed, im_embed))
    embed_param.add_contraint(UnitNormConstraint())
    mix_param = nn.Parameter(mix.unsqueeze(-1))

    return (embed_param, mix_param)

def measurement_param(density_size, measurement_size=20):
    measurement_param = ConstrainedParameter(torch.randn(measurement_size, density_size, dtype=torch.cfloat))
    measurement_param.add_contraint(UnitNormConstraint())
    #nn.init.orthogonal_(measurement_param.data)

    return measurement_param
