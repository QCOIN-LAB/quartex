# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import operations as opn
import operators as opr

class QDNN(nn.Module):
    def __init__(self, cfg, tokenizer):
        super(QDNN, self).__init__()
        self.embed = opr.Embedding.from_glove('/glove/glove.6B.50d.txt', tokenizer.token2idx)
        self.measurement = opr.Measurement(cfg.embed_size, cfg.measurement_size)
    
    def forward(self, x):
        (amp_x, pha_x), mix_x  = self.embed(x)
        weight = F.softmax(mix_x, dim=1)
        x = opn.multiply((amp_x, pha_x))
        x = opn.density(x)

        x = opn.mixture(x, weight=weight)
        p, _ = self.measurement(x)

        return p