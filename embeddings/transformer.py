# -*- coding: utf-8 -*-

from operations import *
from utils import *

from transformers import AutoConfig, AutoModel

# TODO

class Transformer(nn.Module):
    def __init__(self, path, freeze=False):
        super(Transformer, self).__init__()
        sign_mat = torch.sign(embed_mat)
        amp_embed = sign_mat * embed_mat
        amp_embed.data, mix_embed = normalize(amp_embed)
        self.amp_embed = nn.Embedding.from_pretrained(amp_embed, padding_idx=0, freeze=freeze)
        self.amp_embed.weight.add_contraint(UnitNormConstraint())
        self.mix_embed = nn.Embedding.from_pretrained(mix_embed, padding_idx=0, freeze=freeze)
        pha_embed = np.pi * (1 - sign_mat) / 2 # based on phase belongs to [0, 2*pi]
        # when amp is 0, pha shoule be randomly initialized
        pha_embed[sign_mat == 0] = (2 * np.pi * torch.rand_like(pha_embed))[sign_mat == 0]
        self.pha_embed = nn.Embedding.from_pretrained(pha_embed, padding_idx=0, freeze=freeze)
        self.pha_embed.weight.add_contraint(RangeConstraint(0, 2 * np.pi))

    def forward(self, x):
        amp_embed = self.amp_embed(x)
        pha_embed = self.pha_embed(x)
        mix_embed = self.mix_embed(x)

        return (amp_embed, pha_embed), mix_embed