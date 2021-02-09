# -*- coding: utf-8 -*-

#-*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import sklearn
import pickle
import sys
sys.path.append('../../')
import preprocess
from tools.timer import log_time_delta
from nltk.corpus import stopwords

OVERLAP_INDEX = 237
from tools import evaluation
from preprocess.dictionary import Dictionary
from preprocess.embedding import Embedding

class DataReader(object):
    def __init__(self,opt):
        self.onehot = True
        self.unbalanced_sampling = False
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)        
      
        self.dir_path = os.path.join(opt.datasets_dir, 'QA', opt.dataset_name.lower())
        self.preprocessor = preprocess.setup(opt)
        self.datas = self.load(do_filter = opt.remove_unanswered_question)
        self.get_max_sentence_length()
        self.nb_classes = 2
        #self.dict_path = os.path.join(self.bert_dir,'vocab.txt')

        if 'bert' in self.network_type:
            loaded_dic = Dictionary(dict_path =self.dict_path)
            self.embedding = Embedding(loaded_dic,self.max_seq_len)
        else:
            self.alphabet=self.get_dictionary(self.datas.values())
            self.embedding = Embedding(self.alphabet,self.max_seq_len)
            
        print('loading word embedding...')
        if opt.dataset_name=="NLPCC":     # can be updated
            self.embedding.get_embedding(dataset_name = self.dataset_name, language="cn",fname=opt.wordvec_path) 
        else:
            self.embedding.get_embedding(dataset_name = self.dataset_name, fname=opt.wordvec_path)

        

        self.opt_callback(opt) 
        
       
    def opt_callback(self,opt):
        opt.nb_classes = self.nb_classes            
        opt.embedding_size = self.embedding.lookup_table.shape[1]        
        opt.max_sequence_length= self.max_seq_len
        
        opt.lookup_table = self.embedding.lookup_table 
            
    def load(self, do_filter = True):
        datas = dict()
        clean_set = ['test','dev'] if self.train_verbose else ['train','test','dev']
        for data_name in ['train','test','dev']:           
            data_file = os.path.join(self.dir_path,data_name+".txt")
            data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"]).fillna('0')
            if do_filter == True and data_name in clean_set:
                data = self.remove_unanswered_questions(data)
                
            data['question'] = data['question'].apply(lambda x : self.preprocessor.run(x,output_type = 'string'))
            data['answer'] = data['answer'].apply(lambda x : self.preprocessor.run(x,output_type = 'string'))
            datas[data_name] = data
        return datas
    
    @log_time_delta
    def remove_unanswered_questions(self,df):
        counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
        questions_have_correct = counter[counter>0].index
    
        return df[df["question"].isin(questions_have_correct)].reset_index()  #&  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)
    
    def get_max_sentence_length(self):
        q_max_sent_length = max(map(lambda x:len(x),self.datas["train"]['question'].str.split()))
        a_max_sent_length = max(map(lambda x:len(x),self.datas["train"]['answer'].str.split()))    
        self.max_seq_len = max(q_max_sent_length,a_max_sent_length)
        if self.max_seq_len > self.max_len:
            self.max_seq_len = self.max_len
        print('max sequene length: {}'.format(self.max_seq_len))
                
    def get_dictionary(self,corpuses = None,dataset="",fresh=True):
        pkl_name="temp/"+self.dataset_name+".alphabet.pkl"
        if os.path.exists(pkl_name) and not fresh:
            return pickle.load(open(pkl_name,"rb"))
        dictionary = Dictionary(start_feature_id = 0)
        dictionary.add('[UNK]')  
#        alphabet.add('END') 
        for corpus in corpuses:
            for texts in [corpus["question"].unique(),corpus["answer"]]:    
                for sentence in texts:                   
                    tokens = sentence.lower().split()
                    for token in set(tokens):
                        dictionary.add(token)
        print("alphabet size = {}".format(len(dictionary.keys())))
        if not os.path.exists("temp"):
            os.mkdir("temp")
        pickle.dump(dictionary,open(pkl_name,"wb"))
        return dictionary   
    
    
#    @log_time_delta
    def transform(self,flag):
        if flag == 1:
            return torch.tensor([0, 1],dtype = torch.float32)
        else:
            return torch.tensor([1, 0],dtype = torch.float32)
        
    def cut(self,sentence, isEnglish=True):
        if isEnglish:
            tokens =sentence.lower().split()
        else:
            tokens = [word for word in sentence.split() if word not in stopwords]
        return tokens
    
    def encode_to_split(self,sentence, max_sentence):
        indices = []
        tokens = self.cut(sentence)
        for word in tokens:
            indices.append(self.alphabet[word])
        if len(indices) < max_sentence:
            indices += [0]* (max_sentence - len(indices))
        return torch.tensor(indices[:max_sentence],dtype = torch.int64)
    
    # Generate pointwise training samples
    # (Q, Ans, Label) for each sample
    # Produce balanced data
    def get_train_point(self,overlap_feature=False):
        input_num = 3
        pairs = []
        for question,group in self.datas["train"].groupby("question"):
            pos_answer=group[group["flag"]==1]["answer"]
            neg_answer=group[group["flag"]==0]["answer"]
            if len(pos_answer)==0 or len(neg_answer)==0:
                continue
            for pos in pos_answer:
                neg_index=np.random.choice(neg_answer.index)
                neg=neg_answer.loc[neg_index,]
                label_neg=self.transform(0)
                seq_neg_a=self.encode_to_split(neg,self.max_seq_len)
                label_pos=self.transform(1)
                seq_pos_a=self.encode_to_split(pos,self.max_seq_len)
                question_seq=self.encode_to_split(question,self.max_seq_len)
                pairs.append((question_seq, seq_neg_a,label_neg))
                pairs.append((question_seq, seq_pos_a, label_pos))
        
        n_batches = int(len(pairs) * 1.0 / self.batch_size)
        pairs = sklearn.utils.shuffle(pairs, random_state=121)
        print(len(pairs))
        for i in range(n_batches):
            batch = pairs[i * self.batch_size:(i + 1) * self.batch_size]
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
            
        batch = pairs[n_batches * self.batch_size:] 
        if len(batch)>0:
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
        
    # Generate pairwise training samples
    # (Q, Pos_ans, Neg_ans) Triplet for each sample
    def get_train_pair(self,shuffle = True, overlap_feature= False,iterable=True,max_sequence_length=0):
        pairs = []
        
        #q,a,neg_a,overlap1,overlap2 = [],[],[],[],[]
        for question,group in self.datas["train"].groupby("question"):
            pos_answers = group[group["flag"] == 1]["answer"]
            neg_answers = group[group["flag"] == 0]["answer"]
            if len(pos_answers)==0 or len(neg_answers)==0:
                continue
            for pos in pos_answers:  
                
                 #sampling with model
                 #if model is not None and sess is not None:                    
                 #    pos_sent = self.embedding.text_to_sequence(pos)
                 #    q_sent,q_mask = self.prepare_data([pos_sent])                             
                 #    neg_sents = [self.embedding.text_to_sequence(sent) for sent in neg_answers]
                 #    a_sent,a_mask = self.prepare_data(neg_sents)                   
                 #    scores = model.predict(sess,(np.tile(q_sent,(len(neg_answers),1)),a_sent))
                 #    neg_index = scores.argmax()   
                 #    seq_neg_a = neg_sents[neg_index]
                    
                 #just random sampling
                 #else:    
 #                    if len(neg_answers.index) > 0:
                neg_index = np.random.choice(neg_answers.index)
                neg = neg_answers.loc[neg_index,]
                
                seq_neg_a=self.encode_to_split(neg, self.max_seq_len)
                seq_a=self.encode_to_split(pos, self.max_seq_len)                
                seq_q = self.encode_to_split(question, self.max_seq_len)

                if overlap_feature:
                    overlap1 = self.overlap_index(seq_q,seq_a)
                    overlap2 = self.overlap_index(seq_q,seq_neg_a)
                    pairs.append((seq_q,seq_a, seq_neg_a, overlap1,overlap2))
                    input_num = 5
                
                else:
                    pairs.append((seq_q,seq_a,seq_neg_a))
                    input_num = 3

        n_batches = int(len(pairs) * 1.0 / self.batch_size)
        pairs = sklearn.utils.shuffle(pairs, random_state=121)

        for i in range(n_batches):
            batch = pairs[i * self.batch_size:(i + 1) * self.batch_size]
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
        
        batch = pairs[n_batches * self.batch_size:] 
        if len(batch)>0:
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]    
        batch = pairs[n_batches * self.batch_size:] + [pairs[n_batches * self.batch_size]] * (self.batch_size - len(pairs) + n_batches * self.batch_size)
        
    
    # calculate the overlap_index
    def overlap_index(self,question,answer):

        qset = set(question.tolist())
        aset = set(answer.tolist())
        a_len = len(answer)
    
        # q_index = np.arange(1,q_len)
        a_index = np.arange(1,a_len + 1)
    
        overlap = qset.intersection(aset)
        for i,a in enumerate(answer.tolist()):
            if a in overlap:
                a_index[i] = OVERLAP_INDEX
        return torch.tensor(a_index,dtype = torch.int64)
            
    def get_test(self,overlap_feature = False, iterable = True):
        
        if overlap_feature:
            process = lambda row: [self.encode_to_split(row["question"],self.max_seq_len),
                               self.encode_to_split(row["answer"],self.max_seq_len), 
                               self.overlap_index(row['question'],row['answer'])]
            input_num = 3
        else:
            process = lambda row: [self.encode_to_split(row["question"],self.max_seq_len),
                               self.encode_to_split(row["answer"],self.max_seq_len)]
            input_num = 2
        
        samples = self.datas['test'].apply(process,axis=1)
        n_batches = int(len(samples) * 1.0 / self.batch_size)
        for i in range(n_batches):
            batch = samples[i * self.batch_size:(i + 1) * self.batch_size]
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
            
        batch = samples[n_batches * self.batch_size:]
        if len(batch)>0:
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]


    def get_dev(self,overlap_feature = False, iterable = True):
        
        if overlap_feature:
            process = lambda row: [self.encode_to_split(row["question"],self.max_seq_len),
                               self.encode_to_split(row["answer"],self.max_seq_len), 
                               self.overlap_index(row['question'],row['answer'])]
            input_num = 3
        else:
            process = lambda row: [self.encode_to_split(row["question"],self.max_seq_len),
                               self.encode_to_split(row["answer"],self.max_seq_len)]
            input_num = 2
        
        samples = self.datas['dev'].apply(process,axis=1)
        n_batches = int(len(samples) * 1.0 / self.batch_size)
        for i in range(n_batches):
            batch = samples[i * self.batch_size:(i + 1) * self.batch_size]
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
            
        batch = samples[n_batches * self.batch_size:]
        if len(batch)>0:
            yield [torch.stack([pair[j] for pair in batch],dim=0) for j in range(input_num)]
                    
    def prepare_data(self,seqs):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = np.max(lengths)
    
        x = np.zeros((n_samples, max_len)).astype('int32')
        x_mask = np.zeros((n_samples, max_len)).astype('float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
        return x, x_mask
    
    def evaluate(self,predicted,mode="test",acc=False):
        return evaluation.evaluationBypandas(self.datas[mode],predicted,acc=acc)
        

