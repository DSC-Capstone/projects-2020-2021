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

class Metapath2Vec():
    def __init__(self,label_path, mat_dict,num_sentences, input_path, num_tokens, vec_size,type_A,output_path):
        self.label_path = label_path
        self.mat_dict = mat_dict
        self.num_sentences = num_sentences
        self.input_path = input_path
        self.type_A = type_A
        self.num_tokens = num_tokens
        self.vec_size = vec_size 
        self.output_path = output_path

    def perform_walk(self):
        sentences = []
        self.A_train = sparse.load_npz(self.mat_dict['A']).tocsr()
        self.B_matrix = sparse.load_npz(self.mat_dict['B']).tocsr()
        self.P_matrix = sparse.load_npz(self.mat_dict['P']).tocsr()
        self.type_path = [i for i in self.input_path[1:]]
        for _ in tqdm(range(self.num_sentences)):
            app_row = self.A_train.shape[0]
            curr_ind = np.random.choice(app_row)
            max_length = max(len(self.input_path),self.num_tokens)
            min_length = min(len(self.input_path),self.num_tokens)
            sentence_len = np.random.choice(np.arange(min_length, max_length+1))
            sentence = f'app{curr_ind} '
            curr_ind = self.generate_step(self.A_train,'app',curr_ind)
            curr_type = 'api'
            for i in range(sentence_len):
                path = self.type_path[i%len(self.type_path)]
                if path == 'A':
                    if curr_type == 'app':
                        curr_ind = self.generate_step(self.A_train,curr_type,curr_ind)
                        curr_type = 'api'
                    else:
                        curr_ind = self.generate_step(self.A_train,curr_type,curr_ind)
                        curr_type = 'app'
                elif path == 'B':
                    curr_ind = self.generate_step(self.B_matrix,curr_type,curr_ind)
                    curr_type = 'api'
                else:
                    curr_ind = self.generate_step(self.P_matrix,curr_type,curr_ind)
                    curr_type = 'api'
                sentence += f'{curr_type}{curr_ind} '
            sentence = sentence[:-1]
            sentences.append(sentence)
        
        corpus = MyCorpus(sentences)
        model = gensim.models.Word2Vec(sentences=corpus, size=self.vec_size,min_count = 1)
        model.save(f'{self.output_path}/metapath2vec_{self.input_path}_vec{self.vec_size}_tok{self.num_tokens}_sen{self.num_sentences}_{self.type_A}.model')
    
    def generate_step(self,matrix,curr_type,curr_ind):
        if curr_type == 'app':
            return np.random.choice(np.nonzero(matrix[curr_ind,:])[1])
        else:
            return np.random.choice(np.nonzero(matrix[:,curr_ind])[0])
        

        
def build_m2v(label_path, mat_dict,num_sentences, input_path, num_tokens, vec_size,type_A, output_path):
    metapath2vec = Metapath2Vec(label_path, mat_dict,num_sentences, input_path, num_tokens, vec_size,type_A, output_path)
    print('start')
    metapath2vec.perform_walk()

                
            