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


def build_w2v(label_path, mat_dict, num_sentences, num_tokens, vec_size, type_A, output_path):
    labels = json.load(open(label_path))

    # generate word2vec based on AA matrix 
    if 'AA' in mat_dict:
        A_train = sparse.load_npz(mat_dict['AA']).tocsr()

        sentences = []
        for _ in tqdm(range(num_sentences)):  
            sentence_len = np.random.choice(num_tokens)
            A_row = A_train.shape[0]
            A_col = A_train.shape[1]
            start_app = np.random.choice(A_row)
            sentence = f'app{start_app}'
            for i in range(sentence_len):
                api = np.random.choice(np.nonzero(A_train[start_app,:])[1])
                end_app = np.random.choice(np.nonzero(A_train[:, api])[0])
                sentence += f' api{api} app{end_app}' 
            sentence = sentence[:-1]
            sentences.append(sentence)

        corpus = MyCorpus(sentences)
        model = gensim.models.Word2Vec(sentences=corpus, size=vec_size)
        model.save(f'{output_path}/word2vec_AA_vec{vec_size}_tok{num_tokens}_sen{num_sentences}_{type_A}.model')
        


