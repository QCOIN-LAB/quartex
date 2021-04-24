# -*- coding: utf-8 -*-


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import operation as op
import parameter as param

from utils.config import Config
from utils.data import Tokenizer

def build_tokenizer(data_path):
    tokenizer = Tokenizer()
    clean_set = ['train', 'test', 'dev']
    for data_name in clean_set: 
        data_path = os.path.join(data_path,data_name+".txt")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                question, answer, _ = line.rstrip().split('\t')
                for token in question.split():
                    tokenizer.add(token)
                for token in answer.split():
                    tokenizer.add(token)

    return tokenizer

class DataCollator:
    def __init__(self, opt, tokenizer):
        self.tokenizer = tokenizer
        self.dir_path = os.path.join(opt.dataset_dir, 'QA', opt.dataset_name.lower())

    @staticmethod
    def pad(tokens, max_len):
        len_ = len(tokens)
        return tokens + [0] * (max_len - len_) 

    def __call__(self, batch):
        question_input_ids = []
        answer_input_ids = []
        batch_label = []
        max_len1 = max([len(item['question']) for item in batch])
        max_len2 = max([len(item['answer']) for item in batch])
        max_len = max(max_len1, max_len2)
        #max_len = max([len(item['tokens']) for item in batch])

        for item in batch:
            question_input_ids.append(self.pad(self.tokenizer.convert_tokens_to_ids(item['question']), max_len))
            answer_input_ids.append(self.pad(self.tokenizer.convert_tokens_to_ids(item['answer']), max_len))
            batch_label.append(item['label'])

        return {
            'question_ids': torch.tensor(question_input_ids, dtype=torch.long),
            'answer_ids': torch.tensor(answer_input_ids, dtype=torch.long),
            'label': torch.tensor(batch_label, dtype=torch.long),
        }
# def build_data_loader(data_path, tokenizer, data_collator):
#     data_loader = dict()
#     data = []
#     clean_set = ['test','dev'] if self.train_verbose else ['train','test','dev']
#     for data_name in clean_set: 
#         data_path = os.path.join(self.dir_path,data_name+".txt")
#         with open(data_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 question, answer, label = line.rstrip().split('\t')
#                 data.append({
#                     'question': tokenizer.tokenize(question),
#                     'answer': tokenizer.tokenize(answer),
#                     'label': int(label),
#                 })
#         data_loader[data_name] = DataLoader(data, batch_size=2, collate_fn=data_collator, shuffle=True, pin_memory=True)
    
#     return data_loader
class DataReader:
    def __init__(self, config, tokenizer, data_collator):
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.dir_path = os.path.join(config.dataset_dir, 'QA', config.dataset_name.lower())

    #def build_train_data_loader(self, tokenizer, data_collator):
    def build_train_data_loader(self):
        data_loader = dict()
        data = []

        data_path = os.path.join(self.dir_path, "train.txt")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                question, answer, label = line.rstrip().split('\t')
                data.append({
                    'question': self.tokenizer.tokenize(question),
                    'answer': self.tokenizer.tokenize(answer),
                    'label': int(label),
                })
        data_loader = DataLoader(data, batch_size=2, collate_fn=self.data_collator, shuffle=True, pin_memory=True)
        #print(data_loader)
        return data_loader

    # def build_test_data_loader(self, tokenizer, data_collator):
    def build_test_data_loader(self):
        data_loader = dict()
        data = []

        data_path = os.path.join(self.dir_path, "test.txt")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                question, answer, label = line.rstrip().split('\t')
                data.append({
                    'question': self.tokenizer.tokenize(question),
                    'answer': self.tokenizer.tokenize(answer),
                    'label': int(label),
                })
        data_loader = DataLoader(data, batch_size=16, collate_fn=self.data_collator, shuffle=True, pin_memory=True)
        
        return data_loader

    # def build_dev_data_loader(self, tokenizer, data_collator):
    def build_dev_data_loader(self):
        data_loader = dict()
        data = []

        data_path = os.path.join(self.dir_path, "dev.txt")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                question, answer, label = line.rstrip().split('\t')
                data.append({
                    'question': self.tokenizer.tokenize(question),
                    'answer': self.tokenizer.tokenize(answer),
                    'label': int(label),
                })
        data_loader = DataLoader(data, batch_size=2, collate_fn=self.data_collator, shuffle=True, pin_memory=True)
        
        return data_loader

    #def evaluate(self,predicted,mode="test",acc=False):
    #    return evaluation.evaluationBypandas(self.datas[mode],predicted,acc=acc)
        

class CNM(nn.Module):
    def __init__(self, config, tokenizer):
        super(CNM, self).__init__()
        self.embed_param, self.mix_param = param.embedding_param(tokenizer.token2id, './glove/glove.6B.50d.txt')
        self.measurement_param = param.measurement_param(50, 5)
        self.ngram_values = config.ngram_value
        self.pooling_type = config.pooling_type
        self.similarity = nn.CosineSimilarity()

    def routine(self, inp):
        feat = []
        for ngram in [op.ngram(inp, n=int(n_value)) for n_value in self.ngram_values.split(',')]:        
            x, mix  = op.embedding(ngram, self.embed_param, self.mix_param) # b,s,n,e;b,s,n,1
            mix = F.softmax(mix, -2)
            x = op.density(x) # b,s,n,e,e
            x = op.mixture(x, weight=mix).squeeze() # b,s,e,e
            p, _ = op.measurement(x, self.measurement_param, collapse=False) # b,s,m
            feat.append(p)
        feat = torch.cat(feat, -1)
        feat, _ = torch.max(feat, 1, False)

        return feat

    def forward(self, inp_a, inp_b):
        feat_a = self.routine(inp_a)
        feat_b = self.routine(inp_b)
        out = self.similarity(feat_a, feat_b)

        return out


def run_point(config, tokenizer):
    model = CNM(config, tokenizer)
    model = model.to(config.device)
    
    optimizer = torch.optim.RMSprop(list(model.parameters()), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    max_dev_map = 0.
    for i in range(config.epochs):
        print('epoch: ', i)
        train_accs = []
        train_losses = []
        for _i, train_batch in enumerate(config.reader.build_train_data_loader()):
            model.train()
            optimizer.zero_grad()
            batch_q = train_batch['question_ids'].to(config.device)
            batch_a = train_batch['answer_ids'].to(config.device)
            batch_y = train_batch['label'].to(config.device)
            sim = model(batch_q, batch_a)
            # loss = criterion(sim,batch_y[:,0])
            loss = criterion(sim,batch_y[:,0])
            loss.backward()
            optimizer.step()
            n_correct = (sim > 0.5).sum().item()
            n_total = len(sim)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if (_i+1) % 10 == 0:
                model.eval()
                #Evaluate
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                
                scores = []
                for _i, test_batch in enumerate(config.reader.build_dev_data_loader()):
                     batch_q = test_batch['question_ids'].to(config.device)
                     batch_a = test_batch['answer_ids'].to(config.device)
                     score = model(batch_q, batch_a).detach().tolist()
                     scores = scores + score
                     
                map_val, mrr_val, prec_1 = config.reader.evaluate(scores, mode = 'test', acc=False)
                print('average_train_acc: {:.4f},average_train_loss: {:.4f}, dev_map: {:.4f}, dev_mrr: {:.4f}, dev_prec@1: {:.4f}'
                      .format(avg_train_acc, avg_train_loss, map_val,mrr_val,prec_1))

                if map_val > max_dev_map:
                    print('save model!')
                    max_dev_map = map_val
                    torch.save(model.state_dict(), 'temp/best')
    #Loading the best model 
    print('Computing test performance for the best model:')
    model.load_state_dict(torch.load('temp/best')) 
    scores = []
    for _i, test_batch in enumerate(config.reader.build_test_data_loader()):
         batch_q = test_batch['question_ids'].to(config.device)
         batch_a = test_batch['answer_ids'].to(config.device)
         score = model(batch_q, batch_a).detach().tolist()
         scores = scores + score
         
    map_val, mrr_val, prec_1 = config.reader.evaluate(scores, mode = 'test', acc=False)
    print('test_map: {:.4f}, test_mrr: {:.4f}, test_prec@1: {:.4f}'
                      .format(map_val,mrr_val,prec_1))

if __name__ == '__main__':

    config = Config()
    config_file = 'configs/qa.ini'    # define dataset in the config
    config.parse_config(config_file)    
    print('Dataset_name = {}'.format(config.dataset_name))
    tokenizer = build_tokenizer('data\\QA\\trec')
    data_collator = DataCollator(config, tokenizer)
    reader = DataReader(config, tokenizer, data_collator)
    config.reader = reader
    
    if torch.cuda.is_available():
        config.device = torch.device('cuda')
        torch.cuda.manual_seed(config.seed)
    else:
        config.device = torch.device('cpu')
        torch.manual_seed(config.seed)


    run_point(config, tokenizer)
        

