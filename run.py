# -*- coding: utf-8 -*-

from params import Params
import numpy as np
import torch
import models
from loss import rank_hinge_loss
import dataset
import torch.nn as nn

def run_pair(params):
    model = models.setup(params)
    model = model.to(params.device)
    
    optimizer = torch.optim.RMSprop(list(model.parameters()), lr=params.lr)
    criterion = rank_hinge_loss(params)
    max_dev_map = 0.
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        train_losses = []
        for _i, train_batch in enumerate(params.reader.get_train_pair()):
            print('training batch {}'.format(_i))
            model.train()
            optimizer.zero_grad()
            batch_q = train_batch[0].to(params.device)
            batch_a = train_batch[1].to(params.device)
            batch_neg_a = train_batch[2].to(params.device)
            pos_sim = model(batch_q, batch_a)
            neg_sim = model(batch_q, batch_neg_a)
            loss = criterion(pos_sim, neg_sim)
            loss.backward()
            optimizer.step()
            n_correct = (pos_sim > neg_sim).sum().item()
            n_total = len(pos_sim)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if _i % 10 == 0:
                model.eval()
                #Evaluate
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                
                scores = []
                for _i, dev_batch in enumerate(params.reader.get_dev()):
                     batch_q = dev_batch[0].to(params.device)
                     batch_a = dev_batch[1].to(params.device)
                     score = model(batch_q, batch_a).detach().tolist()
                     scores = scores + score
                     
                map_val, mrr_val, prec_1 = params.reader.evaluate(scores, mode = 'dev', acc=False)
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
    for _i, test_batch in enumerate(params.reader.get_test()):
         batch_q = test_batch[0].to(params.device)
         batch_a = test_batch[1].to(params.device)
         score = model(batch_q, batch_a).detach().tolist()
         scores = scores + score
         
    map_val, mrr_val, prec_1 = params.reader.evaluate(scores, mode = 'test', acc=False)
    print('test_map: {:.4f}, test_mrr: {:.4f}, test_prec@1: {:.4f}'
                      .format(map_val,mrr_val,prec_1))
    
def run_point(params):
    model = models.setup(params)
    model = model.to(params.device)
    
    optimizer = torch.optim.SGD(list(model.parameters()), lr=params.lr, weight_decay = 1e-7)
    criterion = nn.BCELoss()
    
    max_dev_map = 0.
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        train_losses = []
        for _i, train_batch in enumerate(params.reader.get_train_point()):
            model.train()
            optimizer.zero_grad()
            batch_q = train_batch[0].to(params.device)
            batch_a = train_batch[1].to(params.device)
            batch_y = train_batch[2].to(params.device)
            score = model(batch_q, batch_a)
            loss = criterion(score,batch_y[:,-1].unsqueeze(dim=-1))
            loss.backward()
            optimizer.step()
            
            n_correct = ((score > 0.5)*batch_y[:,-1].unsqueeze(dim=-1)).sum().item()
            n_total = len(score)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if (_i+1) % 10 == 0:
                model.eval()
                #Evaluate
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                
                scores = []
                for _i, dev_batch in enumerate(params.reader.get_dev()):
                     batch_q = dev_batch[0].to(params.device)
                     batch_a = dev_batch[1].to(params.device)
                     score = model(batch_q, batch_a).detach().tolist()
                     scores = scores + score
                     
                map_val, mrr_val, prec_1 = params.reader.evaluate(scores, mode = 'dev', acc=False)
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
    for _i, test_batch in enumerate(params.reader.get_test()):
         batch_q = test_batch[0].to(params.device)
         batch_a = test_batch[1].to(params.device)
         score = model(batch_q, batch_a).detach().tolist()
         scores = scores + score
         
    map_val, mrr_val, prec_1 = params.reader.evaluate(scores, mode = 'test', acc=False)
    print('test_map: {:.4f}, test_mrr: {:.4f}, test_prec@1: {:.4f}'
                      .format(map_val,mrr_val,prec_1))

# Running MSRPC dataset
# The dataset is paraphrase detection
# So the accuracy is the metric    
# And only pointwise matching is supported
def run_msrpc(params):
    model = models.setup(params)
    model = model.to(params.device)
    
    optimizer = torch.optim.RMSprop(list(model.parameters()), lr=params.lr)
    criterion = nn.MSELoss()
    
    max_dev_acc = 0.
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        train_losses = []
        for _i, train_batch in enumerate(params.reader.get_train_point()):
            model.train()
            optimizer.zero_grad()
            batch_q = train_batch[0].to(params.device)
            batch_a = train_batch[1].to(params.device)
            batch_y = train_batch[2].to(params.device)
            sim = model(batch_q, batch_a)
            loss = criterion(sim,batch_y[:,0])
            loss.backward()
            optimizer.step()
            n_correct = (sim > 0.5).sum().item()
            n_total = len(sim)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if _i % 50 == 0:
                model.eval()
                #Evaluate
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                
                scores = []
                for _i, test_batch in enumerate(params.reader.get_test()):
                     batch_q = test_batch[0].to(params.device)
                     batch_a = test_batch[1].to(params.device)
                     score = model(batch_q, batch_a).detach().tolist()
                     scores = scores + score
                     
                _,_,_, acc_val = params.reader.evaluate(scores, mode = 'test', acc=True)
                print('average_train_acc: {:.4f},average_train_loss: {:.4f}, dev_acc: {:.4f}'
                      .format(avg_train_acc, avg_train_loss, acc_val))

                if acc_val > max_dev_acc:
                    print('save model!')
                    max_dev_acc = acc_val
                    torch.save(model.state_dict(), 'temp/best')
    #Loading the best model 
    print('Computing test performance for the best model:')
    model.load_state_dict(torch.load('temp/best')) 
    scores = []
    for _i, test_batch in enumerate(params.reader.get_test()):
         batch_q = test_batch[0].to(params.device)
         batch_a = test_batch[1].to(params.device)
         score = model(batch_q, batch_a).detach().tolist()
         scores = scores + score
         
    _,_,_, acc_val = params.reader.evaluate(scores, mode = 'test', acc=True)
    print('test_acc: {:.4f}'.format(acc_val))

if __name__=="__main__":
  
    params = Params()
    config_file = 'config/qa.ini'    # define dataset in the config
    params.parse_config(config_file)    
    print('Dataset name = {}'.format(params.dataset_name))
    reader = dataset.setup(params)
    params.reader = reader


    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        torch.cuda.manual_seed(params.seed)
    else:
        params.device = torch.device('cpu')
        torch.manual_seed(params.seed)
    
    if params.dataset_name == 'msrpc':
        print('Matching type = Pointwise.')
        print('Evaluation Metric = Accuracy.')
        run_msrpc(params)
    elif params.match_type == 'point':
        print('Matching type = Pointwise.')
        print('Evaluation Metric = MAP, MRR and PREC@1.')
        run_point(params)
    elif params.match_type == 'pair':
        print('Matching type = Pairwise.')
        print('Evaluation Metric = MAP, MRR and PREC@1.')
        run_pair(params)