import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
import string
import datetime

# set up sets for cleaning
stopword = stopwords.words('english')
letters = set(string.ascii_lowercase)
digits = set(string.digits)

# clean the tweets posts
def clean_text(x):
    text = x['text'].lower().split()
    # remove retweet usernames
    if x['retweeted'] == False:
        text = text[2:]
    text = ' '.join(text)
    
    # remove punctuation
    cleaned = []
    for c in text:
        if (c in letters) or (c in digits) or (c == " "):
            cleaned.append(c)
    text = ''.join(cleaned)
    
    # remove stopwords
    result = ''
    for word in text.split():
        if word not in stopword and len(word) >= 3:
            result += word + ' '
    
    return result


# combine all in functions
def clean_csv(path, filename, topath):
    df = pd.read_csv(path+filename)
    if 'test' not in path:
        df = pd.read_csv(path+filename, usecols=range(1,9))
    df = df[df['language'] == 'en']
    df['clean_text'] = df.apply(clean_text,axis=1)
    df = df.dropna(subset=['clean_text'])
    df.to_csv(topath+filename[:10]+'-clean.csv',index=False)
    return

# clean all the csv data
def clean_all_csv(datapath, outpath):
    for fn in os.listdir(datapath):
        if ('2020' in fn) and ('csv' in fn):
            clean_csv(datapath, fn, outpath)
    print('tweet datasets are saved in `data/interim`!')
    return
