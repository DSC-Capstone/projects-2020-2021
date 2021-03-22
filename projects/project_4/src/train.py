import pandas as pd
import numpy as np
import requests
import os
import gzip
import shutil
import json
from flatten_dict import flatten
from twarc import Twarc
import logging
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def train_model(logger, df, training_data_path, model_path, dims, fit_priors, max_dfs, min_dfs, n_splits, outdir):
    '''Trains model'''
    # Clean Data
    df['text'] = df['text'].apply(lambda x: x.replace('#', '').replace('@', ''))

    # Loop through dimensios
    results = {}
    i = 0
    for dim in dims:
        # Create model
        count_vect = CountVectorizer(stop_words='english', max_df=max_dfs[i], min_df=min_dfs[i])
        tfidf_transformer = TfidfTransformer()
        clf = MultinomialNB(fit_prior=fit_priors[i])
        text_clf = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', clf)])
        # text_clf = Pipeline([('vect', count_vect), ('clf', clf)])

        i += 1

        # Cross validation
        X, y = df['text'], df[dim]

        kf = KFold(n_splits=n_splits, shuffle=True)
        kf.get_n_splits(X)

        class_accs = []
        reg_errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            text_clf.fit(X_train, y_train)
            y_pred = text_clf.predict(X_test)
            class_acc = np.mean(y_pred == y_test)
            reg_error = np.mean(abs(y_pred - y_test))
            class_accs.append(class_acc)
            reg_errors.append(reg_error)

        results = pd.DataFrame({'class_acc': class_acc, 'reg_errors': reg_errors})
        results.to_csv(os.path.join(model_path, '{}_results.csv'.format(dim)))

        # Train on all data
        text_clf.fit(X, y)

        # Save model
        fp = os.path.join(model_path, '{}.mdl'.format(dim))
        f = open(fp, 'wb')
        pickle.dump(text_clf, f)
        
    return 



