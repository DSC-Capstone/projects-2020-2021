'''
predict_model.py is used to predict an web activity is generated using vpn or not using trained model.
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
from sklearn.metrics import confusion_matrix
import pickle
import json
from ..features.build_features import features_build


def predict_model(indir,indir2, outdir):
    '''
    return two table contains predictions of each test data and analysis.
    :param: indir: file directory where trained model stored
    :param: indir2: file directory where test data stored
    :param: outdir: file directory where generated predictions stored
    '''
    filename = os.path.join(indir, 'model.joblib')
    loaded_model = pickle.load(open(filename, 'rb'))
    df = features_build(indir2,outdir,2)
    features_name = ["valid_package_rate,peaks_gap,peaks_number,max_prom_norm, peak_0p1Hz_norm, peak_0p2Hz_norm, pct_zeros"]
    predictions = loaded_model.predict(df)
    df['predictions']=predictions
    df['predictions']=["live" if i == 1 else "streaming" for i in df['predictions']]
    df2=pd.DataFrame()
    entries = os.listdir(indir2)
    df2['file_name']=entries
    df2['predictions']=df['predictions']
    df2.to_csv (outdir+'/predictions.csv', index = False, header=True)
    