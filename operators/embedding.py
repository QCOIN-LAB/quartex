# -*- coding: utf-8 -*-

from operations import *
from utils import *

class Embedding(nn.Module):
    def __init__(self, embed_mat, freeze=False):
        super(GloVe, self).__init__()
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

    @classmethod
    def from_glove(cls, path, token2idx, embed_size=50, freeze=False):
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embed_mat = np.random.uniform(-1/np.sqrt(embed_size), 1/np.sqrt(embed_size), (len(token2idx), embed_size))
        # <pad>
        embed_mat[0, :] = np.zeros((1, embed_size)) 
        for line in fin:
            elements = line.rstrip().split()
            token, vec = ' '.join(elements[:-embed_size]), elements[-embed_size:]
            if token in token2idx:
                embed_mat[token2idx[token]] = np.asarray(vec, dtype=np.float32)
        fin.close()

        return cls(embed_mat, freeze=freeze)

    def forward(self, x):
        amp_embed = self.amp_embed(x)
        pha_embed = self.pha_embed(x)
        mix_embed = self.mix_embed(x)

        return (amp_embed, pha_embed), mix_embed