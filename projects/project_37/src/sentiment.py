import json
import os
import numpy as np
import pandas as pd
import re
import time

import nltk
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pdb

# Helper Functions:

def sentiment_score(reviews_list):
    '''Attach a sentiment score to each sentence of a review. Return DataFrame.'''
    print(" --------Computing Sentiment Scores for Reviews ------")
    start_time = time.time()
    scores = []
    seen_sentences = []
    index = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sid = SentimentIntensityAnalyzer()
    for i, review in enumerate(reviews_list):
        sentences = tokenizer.tokenize(review)
        for sentence in sentences:
            index.append(i)
            seen_sentences.append(sentence)
            ss_intermediate = sid.polarity_scores(sentence)
            scores.append(ss_intermediate)
    df = pd.DataFrame({'index': index, 'sentence': seen_sentences}).join(pd.DataFrame(scores))
    df = df.assign(phrases = df.sentence.apply(lambda x: re.findall('<phrase>.+?</phrase>', x)))
    print("--- %s seconds ---" % (time.time() - start_time))
    return df

# def clean_phrases(sentences):
#     '''Remove the phrase tags'''
#     return sentences.str.replace('<phrase>', '').str.replace('</phrase>', '')

def clean_phrases(sentences):
    '''Remove the phrase tags'''
    if isinstance(sentences, str):
        sentences = eval(sentences)
    return sentences.str.replace('<phrase>', '').str.replace('</phrase>', '')

def make_positive_phrases(df, val):
    '''Return positive phrases'''
    return clean_phrases(df[['compound', 'phrases']][df['compound']>=val].phrases.explode().dropna()).value_counts()

def make_negative_phrases(df, val):
    '''Return positive phrases'''
    return clean_phrases(df[['compound', 'phrases']][df['compound']<=-val].phrases.explode().dropna()).value_counts()

def make_sentimented_restaurant(reviews_list, VAL):
    '''Return a dataframe '''
    df = sentiment_score(reviews_list)
    positive_phrases = make_positive_phrases(df, VAL)
    negative_phrases = make_negative_phrases(df, VAL)
    df = df.assign(positive)
    
def clean_restaurant(df, restaurant_dir):
    starting_ind = list(df[(df['index']==0) & (df['Unnamed: 0']==0)].index)
    i = 0
    add = 0
    df_list = []

    while i < len(starting_ind)-1:
        a = df[starting_ind[i]:starting_ind[i+1]]['index'] + add
        df_list.append(a)
        add += 10000
        i+=1

    df_list.append(df[starting_ind[i]:]['index'] + add)
    df.assign(index=pd.concat(df_list).values).to_csv(restaurant_dir)
    
def process(df):
    val = 0.5
    tmp = df[['compound', 'phrases']].phrases\
    .apply(lambda x: eval(str(x).lower()\
                          .replace('<phrase>', '').replace('</phrase>', '')) if len(x)>2 else None)
    return tmp
    
# Functions

def make_sentiment_table(reviews_list, restaurant_dir):
    '''Return a table with sentiment per sentence; positive phrases; negative phrases'''
    print(" -------- Preparing Restaurant Phrases ------")
    VAL = 0.3
    ## change back later
    df = sentiment_score(reviews_list)
    # df = pd.read_csv(restaurant_dir)
    ## ---
    positive_phrases = make_positive_phrases(df, VAL)
    negative_phrases = make_negative_phrases(df, VAL)
    df.to_csv(restaurant_dir)
    return df, positive_phrases, negative_phrases
    
def make_website_table(df, restaurant_dir, subset_dir):
    '''Create the dataframe used for the website'''
    print(" -------- Building Restaurant Table ------")
    chunksize = 100000
    data_iterator = pd.read_csv(restaurant_dir, index_col=0, chunksize=chunksize)
    lv = pd.read_csv(subset_dir)
    chunk_list = list()
    val = 0.3
    i = 0
    start_time = time.time()
    for data_chunk in data_iterator:
        print("Chunks processing: {}".format(i+1))
        filtered_chunk = process(data_chunk[['index', 'compound', 'phrases']])
        tmp = data_chunk[['index', 'compound', 'phrases']].assign(phrases=filtered_chunk)
        tmp = tmp[tmp['compound']>=val].dropna()
        chunk_list.append(tmp)
        i += 1
    df = pd.concat(chunk_list)
    print("--- %s seconds ---" % (time.time() - start_time))

    a = df.groupby('index')['phrases'].sum().reset_index()
            
#     b = a.merge(lv.rename({'Unnamed: 0': 'index'}, axis=1, on='index')
    b = lv.rename({'Unnamed: 0': 'index'}, axis=1).merge(a, on='index')
                        
    c = b.groupby(['name', 'business_id']).phrases.sum()\
    .apply(lambda x: pd.Series(x).value_counts().to_dict()).reset_index()
    d = lv[['business_id', 'city', 'categories', 'stars']]\
    .groupby(['business_id', 'city', 'categories'])['stars'].mean().reset_index()
    e = c.merge(d, on=['business_id'])
    e = e.assign(phrases = e['phrases'].apply(lambda x: json.dumps(x)))
    e.to_csv(restaurant_dir)
    #clean_restaurant(e, restaurant_dir).to_csv(restaurant_dir)