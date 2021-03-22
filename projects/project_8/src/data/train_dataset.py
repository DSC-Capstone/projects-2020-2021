from dotenv import load_dotenv, find_dotenv
import os
# get the keys for Kaggle api
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
KAGGLE_KEY = os.environ.get('KAGGLE_KEY')
KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME')

import numpy as np
import pandas as pd
import re
import shutil

def download_train(url,filename,path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(url, filename, path=path)
    shutil.unpack_archive(path+filename+'.zip', extract_dir='./'+path)
    return

def data_cleaning(csv_file):
    trained = pd.read_csv(csv_file, encoding = "ISO-8859-1", header = None)
    trained.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    trained = trained[['text', 'sentiment']]
    trained['sentiment'] = trained['sentiment'].replace(0, -1).replace(4, 1).replace(2, 0)
    trained['text'] = trained['text'].str.lower()
    labels = trained['sentiment']
    return trained

def remove_hashtags(text):
    text = re.sub(r"^#\S+|\s#\S+", '', text)
    return text

def remove_urls(text):
    text = re.sub(r'http\S+', '', text)
    return text

def remove_ats(text):
    text = re.sub(r"^@\S+|\s@\S+", '', text)
    return text

def text_cleaning(df):
    df['text'] = df['text'].apply(remove_hashtags)
    df['text'] = df['text'].apply(remove_urls)
    df['text'] = df['text'].apply(remove_ats)
    return df

def get_training_dataset(**kwargs):
    print('downloading training dataset from Kaggle...')
    if kwargs['test']:
        df = data_cleaning(kwargs['path_test']+kwargs['kaggle_data'])
    else:
        download_train(kwargs['kaggle_url'], kwargs['kaggle_data'], kwargs['data_path'])
        df = data_cleaning(kwargs['data_path']+kwargs['kaggle_data'])

    df = text_cleaning(df)
    df.to_csv(kwargs['out_path']+kwargs['train_csv'],index=False)
    print('training dataset is saved in `data/interim`!')
    return