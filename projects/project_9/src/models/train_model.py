'''
train_model.py is used to train models for classification.
'''
## import library
import pandas as pd
import os
import sys
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
def svc_model(X_train, y_train, gamma_value):
    '''
    return trained svc model. 
    :param: X_train: a list contains all names of network-stats records.
    :param: y_train: file directory where raw data stored.
    :param: gamma_value:the gamma value used for svc model.
    '''
    clf = svm.SVC(gamma = gamma_value)
    clf.fit(X_train, y_train)
    return clf

def linear_svc_model(X_train,y_train):
    '''
    return trained linear svc model. 
    :param: X_train: a list contains all names of network-stats records.
    :param: y_train: file directory where raw data stored.
    '''
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    return clf

def kneighbors_model(X_train,y_train,n_neighbor):
    '''
    return trained KNNeighbors model. 
    :param: X_train: a list contains all names of network-stats records.
    :param: y_train: file directory where raw data stored.
    :param: n_neighbors:the n_neighbors value used for KNNeighbors model.
    '''
    clf = KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(X_train, y_train)
    return clf
        
def logistic_model(X_train,y_train,solver_method):
    '''
    return trained logistic model. 
    :param: X_train: a list contains all names of network-stats records.
    :param: y_train: file directory where raw data stored.
    :param: solver_method:the solver value used for logistic model.
    '''
    clf = LogisticRegression(solver = solver_method)
    clf.fit(X_train, y_train)
    return clf

def random_forest_model(X_train,y_train,estimators_num):
    '''
    return trained random forest model. 
    :param: X_train: a list contains all names of network-stats records.
    :param: y_train: file directory where raw data stored.
    :param: estimators_num:the estimators value used for random forest model.
    '''
    clf = RandomForestClassifier(n_estimators = estimators_num)
    clf.fit(X_train, y_train)
    return clf

        
def train_model(indir,outdir,testsize,randomstate,method,method_parameters):
    '''
    return a trained model with a json file contains report of this model.
    :param: indir: file directory where extracted features stored.
    :param: outdir: file directory where output of this funcition stored.
    :param: testsize: the portion of train dataset used for validation.
    :param: randomstate: the randomstate number to random split train and valid set.
    :param: method: the classifier name used for training.
    :param: method_parameters: the parameter used for training.
    '''
    df=pd.read_csv(indir)
    features_name = ["valid_package_rate","peaks_gap","peaks_number","max_prom_norm", "peak_0p1Hz_norm", "peak_0p2Hz_norm", "pct_zeros"]
    y = np.array(df["data_label"])
    x = np.array(df[features_name])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=testsize, shuffle = True,random_state=randomstate)
    if method=="SVC":
        clf=svc_model(X_train, y_train, method_parameters)
    if method=="Linear_SVC":
        clf=linear_svc_model(X_train, y_train)
    if method=="KNeighbors":
        clf=kneighbors_model(X_train, y_train, method_parameters)
    if method=="Logistic":
        clf=logistic_model(X_train, y_train, method_parameters)
    if method=="Random_Forest":
        clf=random_forest_model(X_train, y_train, method_parameters)
    
    train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(train_pred, y_train)
    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(test_pred, y_test)
    model_report={}
    model_report["Using Features"]="valid_package_rate,peaks_gap,peaks_number,max_prom_norm, peak_0p1Hz_norm, peak_0p2Hz_norm, pct_zeros"
    model_report["Using Classifier"]=method
    model_report["Train Accuracy"]= str(train_accuracy)
    model_report["Valid Accuracy"]= str(test_accuracy)
    tn, fp, fn, tp=0,0,0,0
    for i in np.arange(len(y_test)):
        if y_test[i]==0 and test_pred[i]==0:
            tn+=1
        if y_test[i]==1 and test_pred[i]==0:
            fp+=1
        if y_test[i]==0 and test_pred[i]==1:
            fn+=1
        if y_test[i]==1 and test_pred[i]==1:
            tp+=1
    model_report["Validation Set True Negative"]=str(tn)
    model_report["Validation Set False Positive"]=str(fp)
    model_report["Validation Set False Negative"]=str(fn)
    model_report["Validation Set True Positive"]=str(tp)

    
    #importance = clf.feature_importances_
    # summarize feature importance
    #print(importance)
    
    filename = os.path.join(outdir, 'model.joblib')
    pickle.dump(clf, open(filename, 'wb'))
    filename2 = os.path.join(outdir, 'training_report.json')
    with open(filename2, "w") as outfile:  
        json.dump(model_report, outfile)