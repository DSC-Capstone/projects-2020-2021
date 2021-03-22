from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


import src.utils as utils
import pandas as pd
import numpy as np


def BoG_model(X,y, clf = 'Logistic', vocab = None, combining = False):
    """
    Parameters:
    X: Training Features
    y: Training Label
    clf: Choice of classifier. Default Logistics. Options: Logistic / SVM
    vocab: Specified vocabulary list for BoG.
    combining: Boolean for deciding include unigram feature from raw. Default False.

    This function is a building pipeline for our task.
    Data will be precessed with BagOfWord with specified vocab list.
    Then feed into the Logistic / SGD Classifier(SVM).

    Return:
        pipe: The fitted Classifier.
    """
    if vocab is None:
        if clf == 'Logistic':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer()),
               ('clasiffier', LogisticRegression())
               ])
        if clf == 'SVM':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer()),
               ('clasiffier', SGDClassifier())
               ])
    else:
        if combining == True:
            count_vect = CountVectorizer().fit(X)   
            vocab_lst = np.unique(list(count_vect.vocabulary_) + list(vocab))
        if combining == False:
            vocab_lst = vocab
        
        if clf == 'Logistic':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
               ('clasiffier', LogisticRegression())
               ])
        if clf == 'SVM':
            pipe = Pipeline([
               ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
               ('clasiffier', SGDClassifier())
               ])
    pipe.fit(X, y)
    return pipe

def Tfidf_model(X,y, clf = 'Logistic', vocab = None, combining = False):
    """
    Parameters:
    X: Training Features
    y: Training Label
    clf: Choice of classifier. Default Logistics. Options: Logistic / SVM
    vocab: Specified vocabulary list for BoG.
    combining: Boolean for deciding include unigram feature from raw. Default False.

    This function is a building pipeline for our task.
    Data will be precessed with TF-IDF with specified vocab list.
    Then feed into the Logistic / SGD Classifier(SVM).

    Return:
        pipe: The fitted Classifier.
    """
    if vocab is None:
        if clf == 'Logistic':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer()),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', LogisticRegression())
                ])
        if clf == 'SVM':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer()),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', SGDClassifier())
                ])
    else:
        if combining == True:
            count_vect = CountVectorizer().fit(X)   
            vocab_lst = np.unique(list(count_vect.vocabulary_) + list(vocab))
        if combining == False:
            
            vocab_lst = vocab
            
        if clf == 'Logistic':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', LogisticRegression())
                ])
        if clf == 'SVM':
            pipe = Pipeline([
                ('BagOfWord', CountVectorizer(vocabulary =vocab_lst)),
                ('TfIdf',TfidfTransformer()),
                ('clasiffier', SGDClassifier())
                ])
    pipe.fit(X, y)
    return pipe

def build_model(X,y,model = 'SVM', vectorizing = 'tfidf', vocab_lst = None, combining = False):

    """
    Parameters:
    X: Training Features
    y: Training Label
    model: Choice of classifier. Default SVM. Options: Logistic / SVM
    vectorizing: Choose the feature processing procedure. "tfidf" (default) or default "bag of word"
    vocab: Specified vocabulary list for feature processing.
    combining: Boolean for deciding include unigram feature from raw. Default False.

    This function is a building pipeline for our task.
    Data will be precessed with model of choice, vectorizing step with specified vocab list.
    Then feed into the Logistic / SGD Classifier(SVM).

    Return:
        pipe: The fitted Classifier.
    """
    
    if vectorizing == 'tfidf':
        model = Tfidf_model(X,y, clf = model, vocab = vocab_lst, combining = combining)
    else:
        model = BoG_model(X,y, clf = model, vocab = vocab_lst, combining = combining)
        
    return model
