# -*- coding: utf-8 -*-

from operations import *

class NGram(nn.Module):
    def __init__(self):
        super(NGram, self).__init__()

    def forward(self, x, n=3):
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