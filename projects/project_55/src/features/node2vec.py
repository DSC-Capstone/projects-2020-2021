import os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from scipy import sparse
import gensim.models

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for line in self.sentences:
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')

class Node2Vec():
    def __init__(self,label_path, mat_dict, num_sentence, num_tokens, vec_size,type_A, output_path,p,q):
        self.label_path = label_path
        self.mat_dict = mat_dict
        self.num_sentences = num_sentence
        self.num_tokens = num_tokens
        self.type_A = type_A
        self.vec_size = vec_size
        self.output_path = output_path
        self.p = p 
        self.q = q

    def perform_walk(self):
        sentences = []
        self.A_train = sparse.load_npz(self.mat_dict['A']).tocsr()
        if 'B' not in self.mat_dict:
            self.type = 'AA'
            for _ in tqdm(range(self.num_sentences)):  
                sentence_len = np.random.choice(self.num_tokens)
                app_row = self.A_train.shape[0]
                api_col = self.A_train.shape[1]
                sentence = ''
                app = np.random.choice(app_row)
                api = np.random.choice(np.nonzero(self.A_train[app,:])[1])
                sentence += f'app{app} api{api} ' 
                cur = 'api'
                for i in range(sentence_len-1):
                    if cur == 'api':
                        app = self.A_only_generate_with_probability(app,api,'api')
                        cur = 'app'
                        sentence += f'app{app} '
                    else:
                        api = self.A_only_generate_with_probability(app,api,'app')
                        cur = 'api'
                        sentence += f'api{api} '
                end_app = self.A_only_generate_with_probability(app,api,'api')
                sentence += f'app{end_app}' 
                sentences.append(sentence)
        else:
            self.type = 'ALL'
            self.app_row = self.A_train.shape[0]
            self.matrix = sparse.vstack([self.A_train,(sparse.load_npz(self.mat_dict['B']).tocsr()+sparse.load_npz(self.mat_dict['P']).tocsr())])
            self.matrix = sparse.hstack([sparse.vstack([sparse.csr_matrix((self.app_row,self.app_row), dtype=np.int8),self.A_train.T]),self.matrix]).astype(bool).astype(int).tocsr()
            self.matrix.setdiag(0)
            self.potential_list = np.arange(self.matrix.shape[0])
            for _ in tqdm(range(self.num_sentences)):
                sentence_len = np.random.choice(self.num_tokens)
                sentence = ''
                prev = np.random.choice(self.app_row)
                curr = np.random.choice(np.nonzero(self.matrix[prev,:])[1])
                sentence += f'app{prev} api{curr} ' 
                for i in range(sentence_len):
                    curr,prev = self.generate_with_probability(curr,prev)
                    sentence += f'{self.convert_to_string(curr)} '
                sentence = sentence[:-1]
                sentences.append(sentence)
        corpus = MyCorpus(sentences)
        model = gensim.models.Word2Vec(sentences=corpus, size=self.vec_size, min_count=1)
        model.save(f'{self.output_path}/node2vec_{self.type}_vec{self.vec_size}_tok{self.num_tokens}_sen{self.num_sentences}_{self.type_A}.model')
        print('saved')
    
    def convert_to_string(self,row_num):
        if row_num < self.app_row:
            return 'app'+str(row_num)
        else:
            return 'api'+str(row_num - self.app_row)


    def A_only_generate_with_probability(self,orig_app, orig_api,start='app'):
        if start == 'app':
            api_list = np.nonzero(self.A_train[orig_app,:])[1]
            prob_list = [1/self.p if i == orig_api else 1/self.q for i in api_list]
            total_prob = sum(prob_list)
            prob_list = [i/total_prob for i in prob_list]
            return np.random.choice(api_list,p=prob_list)
        else:
            app_list = np.nonzero(self.A_train[:,orig_api])[0]
            prob_list = [1/self.p if i==orig_app else 1/self.q for i in app_list]
            total_prob = sum(prob_list)
            prob_list = [i/total_prob for i in prob_list]
            return np.random.choice(app_list,p=prob_list)

    def generate_with_probability(self,curr,prev):
        curr_list = np.nonzero(self.matrix[curr,:])[1]
        prob_list = self.matrix[prev,:].toarray()[0]
        prob_list[prev] = 1/self.p
        prob_list = prob_list[curr_list]
        prob_list = np.where(prob_list==0,1/self.q,prob_list)
        total_prob = sum(prob_list)            
        prob_list = [i/total_prob for i in prob_list]
        return_val = np.random.choice(curr_list,p = prob_list)
        return return_val, curr
            

def build_n2v(label_path, mat_dict, num_sentences, num_tokens, vec_size, type_A, output_path,p,q):
    node2vec = Node2Vec(label_path, mat_dict, num_sentences, num_tokens, vec_size, type_A, output_path,p,q)
    print('start')
    node2vec.perform_walk()
