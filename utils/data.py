# -*- coding:utf-8 -*-

import os

import torch

class Tokenizer:
    def __init__(self):
        self.token2id = {'<pad>': 0, '<unk>': 1}
        self.id2token = {0: '<pad>', 1: '<unk>'}

    def add(self, token):
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)
            self.id2token[len(self.id2token)] = token

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    @staticmethod
    def tokenize(text):
        return text.split()
    
    def convert_tokens_to_ids(self, tokens):
        return [self.token2id[token] if token in self.token2id else self.token2id['<unk>'] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id2token[id] for id in ids]



