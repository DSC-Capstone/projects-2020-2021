import pandas as pd
import os,glob
import re
from pathlib import Path
import pandas as pd 
import numpy as np
from collections import defaultdict 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def create_features(def_dict):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []
    for i in def_dict.keys():
        col1.append(i)
        col2.append(len(def_dict[i]['invoke_type']['invoke-static']))
        col3.append(len(def_dict[i]['invoke_type']['invoke-virtual']))
        col4.append(len(def_dict[i]['invoke_type']['invoke-direct']))
        col5.append(len(def_dict[i]['invoke_type']['invoke-super']))
        col6.append(len(def_dict[i]['invoke_type']['invoke-interface']))
        col7.append(len(def_dict[i]['Combined']['APIs']))
    df = pd.DataFrame([col1,col2,col3,col4, col5, col6,col7]).T
    df['App Name'] = df[0]
    df['invoke-static'] = df[1]
    df['invoke-virtual'] = df[2]
    df['invoke-direct'] = df[3]
    df['invoke-super'] = df[4]
    df['invoke-interface'] = df[5]
    df['All APIs'] = df[6]
    df = df.drop([0,1,2,3,4,5,6],1)
    return df

def scatter_matrix(data, outdir):
    data = create_features(data)
    pd.plotting.scatter_matrix(data)
    plt.suptitle('Independent Gaussians')
    plt.savefig(os.path.join(outdir, 'scatter_matrix.png'))

def basic_stats(data, outdir):
    data = create_features(data)
    out = pd.concat(
        [data.mean().rename('means'), 
         data.std().rename('standard deviations'),
         data.median().rename('medians')], 
        axis=1)
    out.to_csv(os.path.join(outdir, 'basic_stats.csv'))

def run_model(data_dict):
    #Split Data
    df = create_features(data_dict)
    #y = df['App Name'].astype('category').cat.codes
    #X = df.drop(['App Name'],1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    #run GB classifier for baseline
    #df = df[['App Category','All API','invoke-static','invoke-virtual','invoke-direct','invoke-super', 'invoke-interface']]
    #grandientboosting = GradientBoostingClassifier()
    #grandientboosting.fit(X_train,y_train)
    #grandientboosting.predict(X_test)
    print("Model Accuracy - ")
    return df.head() #grandientboosting.score(X_test, y_test)


