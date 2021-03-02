# -*- coding:utf-8 -*-

import os

import torch

class Tokenizer:
    def __init__(self):
        self.token2id = {'[PAD]': 0, '[UNK]': 1}
        self.id2token = {0: '[PAD]', 1: '[UNK]'}

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
        ids = []
        for token in tokens:
            ids.append(self.token2id[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            tokens.append(self.id2token[id])

        return tokens



