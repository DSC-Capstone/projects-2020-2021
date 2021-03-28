import numpy as np
import pandas as pd
import re
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import sparse
import os



def tokenize(text):
    tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    encode_text = tokenizer.encode(text, add_special_tokens = True)
    return encode_text


def tokenization(df):
    tokenized_text = df['text'].apply(tokenize)
    return tokenized_text


def padding(tokenized_text):
    max_length = 0
    for i in range(len(tokenized_text)):
        length = len(tokenized_text[i])
        if length > max_length:
            max_length = length
    for i in range(len(tokenized_text)):
        tokenized_text[i] = tokenized_text[i] + (max_length - len(tokenized_text[i]))*[0]
    lis = [tokenized_text[0]]
    for i in range(1, len(tokenized_text)):
        lis = lis + [tokenized_text[i]]
    trained_matrix = sparse.csr_matrix(lis)
    return trained_matrix, max_length

def padding_test(tokenized_text, max_length):
    for i in range(len(tokenized_text)):
        if len(tokenized_text[i]) <= max_length:
            tokenized_text[i] = tokenized_text[i] + (max_length - len(tokenized_text[i]))*[0]
        elif len(tokenized_text[i]) > max_length:
            tokenized_text[i] = tokenized_text[i][:max_length]
    lis = [tokenized_text[0]]
    for i in range(1, len(tokenized_text)):
        lis = lis + [tokenized_text[i]]
    trained_matrix = sparse.csr_matrix(lis)
    return trained_matrix


def model_trained(trained_matrix, df):
    labels = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(trained_matrix, labels, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    value = model.predict(X_test)
    accuracy = accuracy_score(value, y_test)
    return accuracy,model


def build_logreg(**kwargs):
    path,cleanpath,outpath = kwargs['data_path'],kwargs['cleaned_csv'],kwargs['out_path']

    cleaned = pd.read_csv(path+cleanpath)
    tokenized_text = tokenization(cleaned)
    trained_matrix,max_length = padding(tokenized_text)
    results = model_trained(trained_matrix, cleaned)
    print('accuracy of Log Reg model on the labeled dataset:', results[0])

    if(kwargs['model'] == 'logreg'):
        score_list,date_list = [],[]
        for i in os.listdir(path):
            if '-clean' in i:
                test_df = pd.read_csv(path+i)
                text = padding_test(test_df['clean_text'].apply(tokenize), max_length)
                score = np.mean(results[1].predict(text))
                score_list.append(score)
                date_list.append(i[:10])

        df_score = pd.DataFrame({'date':date_list, "score":score_list})
        df_score['date'] = pd.to_datetime(df_score['date'])
        df_score.sort_values(by=['date'], inplace=True)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        df_score.to_csv(outpath+kwargs['score_csv'],index=False)
        print('predictions of LogReg are saved in `data/final`')
    return 