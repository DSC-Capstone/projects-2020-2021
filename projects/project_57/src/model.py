#Imports
from sklearn.svm import SVC
import os,glob
import re
from pathlib import Path
import pandas as pd 
import numpy as np
from collections import defaultdict 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.sparse import coo_matrix
from sklearn.metrics import f1_score

def AAT(matrix_a):
    AAT = matrix_a.dot(matrix_a.transpose()).todense()
    return AAT
def ABAT(matrix_a, matrix_b):
    AB = matrix_a.dot(matrix_b)
    ABAT = AB.dot(matrix_b.transpose()).todense()
    return ABAT

def APAT(matrix_a, matrix_p):
    AP = matrix_a.dot(matrix_p)
    APAT = AP.dot(matrix_p.transpose()).todense()
    return APAT
def APBPAT(matrix_a, matrix_p, matrix_b):
    AP = matrix_a.dot(matrix_p)
    APB = AP.dot(matrix_b)
    APBP = APB.dot(matrix_p.transpose())
    APBPTAT = APBP.dot(matrix_a.transpose()).todense()
    return APBPTAT
def build_svm(kernel, k,  malware_seen, benign_seen):
    df = pd.DataFrame(kernel)
    benign_seen = list(benign_seen)#[:-1]
    malware_seen = list(malware_seen)#[:-1]
    to_remove = []
    temp_arr = []
    
    lab = ['malware', 'benign']
    if k == 'aat':
        for index, elem in df.iterrows():
            for index, elem in enumerate(elem): 
                if elem == 0:
                    to_remove.append(index)
            break
        df = df.drop(to_remove,1)
        df = df.drop(to_remove)
        x = [*malware_seen, *benign_seen]
    else:
        x = [*malware_seen, *benign_seen, *lab]
        temp_arr = [*temp_arr, *lab]
    x = dict(zip(x, range(0,len(x))))
    df['label'] = x.keys()
    for elem in list(df['label']):
        if elem in benign_seen: 
            temp_arr.append('benign')
        elif elem in malware_seen:
            temp_arr.append('malware')
    df['label'] = temp_arr
    X = df.iloc[:,:-1] 
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    clf = SVC(kernel = 'linear')
    fitted = clf.fit(X_train,y_train)
    pred = fitted.predict(X_test)
    accuracy = fitted.score(X_test, y_test)
    return (accuracy, f1_score(y_test, pred, average='micro'))
