# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import operation as op
import parameter as param

from utils.data import Tokenizer

def build_tokenizer(data_path):
    tokenizer = Tokenizer()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            _, text = line.rstrip().split('####')
            for token in text.split():
                tokenizer.add(token)

    return tokenizer

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def pad(tokens, max_len):
        len_ = len(tokens)
        return tokens + [0] * (max_len - len_) 

    def __call__(self, batch):
        batch_input_ids = []
        batch_label = []
        max_len = max([len(item['tokens']) for item in batch])

        for item in batch:
            batch_input_ids.append(self.pad(self.tokenizer.convert_tokens_to_ids(item['tokens']), max_len))
            batch_label.append(item['label'])

        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'label': torch.tensor(batch_label, dtype=torch.long),
        }

def build_data_loader(data_path, tokenizer, data_collator):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.rstrip().split('####')
            data.append({
                'tokens': tokenizer.tokenize(text),
                'label': int(label),
            })
    data_loader = DataLoader(data, batch_size=2, collate_fn=data_collator, shuffle=True, pin_memory=True)
    
    return data_loader

class BaseClassifier(nn.Module):
    def __init__(self, tokenizer):
        super(BaseClassifier, self).__init__()
        self.embed_param, self.mix_param = param.embedding_param(tokenizer.token2id, './glove/glove.6B.50d.txt')
        self.measurement_param = param.measurement_param(50, 2)
    
    def forward(self, x):
        x, mix  = op.embedding(x, self.embed_param, self.mix_param)
        weight = F.softmax(mix, dim=1)
        x = op.density(x)

        x = op.mixture(x, weight=weight)
        p, _ = op.measurement(x, self.measurement_param)

        return torch.log(p.squeeze() + 1e-7)

    def apply_constraint(self):
        for param in self.parameters():
            if getattr(param, 'constraint', None):
                param.apply_constraint()

if __name__ == '__main__':
    tokenizer = build_tokenizer('datasets/dummy.txt')
    data_collator = DataCollator(tokenizer)
    data_loader = build_data_loader('datasets/dummy.txt', tokenizer, data_collator)

    model = BaseClassifier(tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(50):
        for batch in data_loader:
            model.train()

            p = model(batch['input_ids'])
            loss = F.nll_loss(p, batch['label'])
            print('loss', loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()
            model.apply_constraint()
        

