import sklearn.metrics as me
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import re

def create_docs_NE(df):
    '''
    Clean the given dataframe name entity column by dropping off location entity.
    Merge same documents entity in to one list.
    '''
    for i in df.NE.values:
        removes=[]
        for j in range(len(i)):
            if len(i)>0:
                if i[j].split(':')[1]==' Location':
                    removes.append(i[j])
                else:
                    i[j]=i[j].split(':')[0]
            else: i[j]=i[j].split(':')[0]
        for k in removes:
            i.remove(k)
    s=df[['doc','NE']]
    docs=[]
    sets=set()
    for i in range(1,len(s.doc.values)):
        if s.doc.values[i]==s.doc.values[i-1]:
            for j in set(s.NE.values[i-1]):
                sets.add(j)
        else:
            for j in set(s.NE.values[i-1]):
                sets.add(j)
            docs.append(list(sets))
            sets=set()
    docs.append(list(sets))
    return docs


def evaluate(true, pred):
    '''
    Evaluate the F1, Precision, Recall and Accuracy for the given prediction and true labels.
    '''
    print("F1 Score: ",me.f1_score(true, pred,average = 'weighted'))
    print("Precision: ",me.precision_score(true, pred,average = 'weighted'))
    print("Recall: ", me.recall_score(true, pred,average = 'weighted'))
    print("Accuracy: ",me.accuracy_score(true, pred))
    
    
def ner_vect(lst):
    '''
    Vectorize the Name Entity list by uni gram
    '''
    se = pd.Series(lst)
    text_for_vect = ' '.join(se.apply(lambda x: str(x)))
    count_vect = CountVectorizer().fit([text_for_vect]) 
    vocab_lst = np.unique(list(count_vect.vocabulary_))
    return vocab_lst

def ner_preprocess(path):
    '''
    Read the ner csv file and clean it by using previously defined fucntions.
    '''
    df = pd.read_csv(path,converters={'NE': eval})
    toreturn = create_docs_NE(df)
    toreturn = ner_vect(toreturn)
    return list(toreturn)

def phrase_preprocess(path):
    '''
    Read the autophrase result and obtain only phrases has quality better than .5
    '''
    df = pd.read_csv(path,error_bad_lines=False, delimiter= '\t',names = ['p','phrase'])
    df = df[df.p >= 0.5]
    return list(df.phrase)

def regex_condi(string):
    if ".com" in string:
        return False
    if ".edu" in string:
        return False
    if "@" in string:
        return False
    if 'Host'in string:
        return False
    if ".gov" in string:
        return False

    return True

def clean_text(string):
    '''
    Clean the 20 news group dataset by regex and helper function regex_condi
    '''
    string = " ".join([i for i in string.split() if regex_condi(i)])
    string = re.sub(r"Re:","", string)
    string = re.sub(r"Reply-To:","", string)
    toreturn = string.strip()
    return toreturn

def load_20_news():
    '''
    load the 20 news group and train, validation, test split the data set for training.
    '''
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    newsgroups_train_X,newsgroups_train_y = newsgroups_train.data, newsgroups_train.target
    newsgroups_val_X, newsgroups_test_X, newsgroups_val_y, newsgroups_test_y = train_test_split(newsgroups_test.data, newsgroups_test.target, test_size=0.5, random_state=42)


    # Clean train 
    newsgroups_train_X = [clean_text(i) for i in newsgroups_train_X]
    newsgroups_val_X = [clean_text(i) for i in newsgroups_val_X]
    newsgroups_test_X =  [clean_text(i) for i in newsgroups_test_X]
    return newsgroups_train_X,newsgroups_test_X,newsgroups_val_X, newsgroups_train_y,newsgroups_test_y,newsgroups_val_y

def load_bbc_news(path):
    '''
    load the BBC news group and train, validation, test split the data set for training.
    '''
    bbc_df = pd.read_csv(path)[['summary','type_code']]
    bbc_X_train, bbc_X_test, bbc_y_train, bbc_y_test = train_test_split(bbc_df.summary, bbc_df.type_code, test_size=0.4, random_state=2021)
    bbc_X_val,bbc_X_test,bbc_y_val,bbc_y_test = train_test_split(bbc_X_test, bbc_y_test, test_size=0.5, random_state=42)
    
    return bbc_X_train, bbc_X_test,bbc_X_val, bbc_y_train, bbc_y_test, bbc_y_val

def load_20_news_test(n = None):

    '''
    load the 20 news group and train, validation, test split the data set for training.
    Input:
    n: The number of instances for test target.
    '''
    if n is None:
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

        newsgroups_train_X,newsgroups_train_y = newsgroups_train.data, newsgroups_train.target
        newsgroups_val_X, newsgroups_test_X, newsgroups_val_y, newsgroups_test_y = train_test_split(newsgroups_test.data, newsgroups_test.target, test_size=0.5, random_state=42)


        # Clean train 
        newsgroups_train_X = [clean_text(i) for i in newsgroups_train_X]
        newsgroups_val_X = [clean_text(i) for i in newsgroups_val_X]
        newsgroups_test_X =  [clean_text(i) for i in newsgroups_test_X]
    else:
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

        newsgroups_train_X,newsgroups_train_y = newsgroups_train.data, newsgroups_train.target
        newsgroups_val_X, newsgroups_test_X, newsgroups_val_y, newsgroups_test_y = train_test_split(newsgroups_test.data, newsgroups_test.target, test_size=0.5, random_state=42)


        # Clean train 
        newsgroups_train_X = [clean_text(i) for i in newsgroups_train_X]
        newsgroups_val_X = [clean_text(i) for i in newsgroups_val_X]
        newsgroups_test_X =  [clean_text(i) for i in newsgroups_test_X]
    return newsgroups_train_X[:n],newsgroups_test_X[:n],newsgroups_val_X[:n], newsgroups_train_y[:n],newsgroups_test_y[:n],newsgroups_val_y[:n]
