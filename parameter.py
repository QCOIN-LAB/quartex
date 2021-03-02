# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

import operation as op
from utils.constraint import ConstrainedParameter, UnitNormConstraint, RangeConstraint


def embedding_param(token2id, from_path='./glove/glove.6B.50d.txt', embed_size=50, freeze=False):
    fin = open(from_path, 'r', encoding='utf-8', errors='ignore')
    embed_mat = np.random.uniform(-1/np.sqrt(embed_size), 1/np.sqrt(embed_size), (len(token2id), embed_size))
    # <pad>
    embed_mat[0, :] = np.zeros((1, embed_size)) 
    for line in fin:
        elements = line.rstrip().split()
        token, vec = ' '.join(elements[:-embed_size]), elements[-embed_size:]
        if token in token2id:
            embed_mat[token2id[token]] = np.asarray(vec, dtype=np.float32)
    fin.close()
    embed_mat = torch.from_numpy(embed_mat)

    sign_mat = torch.sign(embed_mat)
    amp_embed = sign_mat * embed_mat
    amp_embed, mix_embed = op.normalize(amp_embed)
    amp_embed = nn.Embedding.from_pretrained(amp_embed, padding_idx=0, freeze=freeze)
    #amp_embed.weight.add_contraint(UnitNormConstraint())
    mix_embed = nn.Embedding.from_pretrained(mix_embed.unsqueeze(-1), padding_idx=0, freeze=freeze)
    pha_embed = np.pi * (1 - sign_mat) / 2 # based on phase belongs to [0, 2*pi]
    # when amp is 0, pha shoule be randomly initialized
    pha_embed[sign_mat == 0] = (2 * np.pi * torch.rand_like(pha_embed))[sign_mat == 0]
    pha_embed = nn.Embedding.from_pretrained(pha_embed, padding_idx=0, freeze=freeze)
    #pha_embed.weight.add_contraint(RangeConstraint(0, 2 * np.pi))

    return (amp_embed, pha_embed, mix_embed)

def measurement_param(density_size, measurement_size=20):
    amp_param = ConstrainedParameter(torch.DoubleTensor(measurement_size, density_size))
    amp_param.add_contraint(UnitNormConstraint())
    nn.init.orthogonal_(amp_param.data)
    pha_param = ConstrainedParameter(torch.DoubleTensor(measurement_size, density_size))
    pha_param.add_contraint(RangeConstraint(0, 2 * np.pi))
    nn.init.constant_(pha_param.data, 0.0)

    return (amp_param, pha_param)
