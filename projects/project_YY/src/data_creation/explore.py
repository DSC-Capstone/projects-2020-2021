import os
import re
import numpy as np
import pandas as pd
import json
import random
import threading
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    progress_bar=f'{prefix} |{bar}| {percent}% {suffix}'
    print(progress_bar, end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print(" "*len(progress_bar), end="\r")

def create_feature_frame(directory, apps, feature):
    """Builds dataframe for specified apps based on given feature within a directory
       Returned dataframe will have features as rows and app names as columns
    
    :param directory : str
        Filepath to directory containing apps
        
    :param apps : list
        List of app names (strings) within directory
        
    :param feature : str
        Feature to parse app data for
    """
    directory_frame = pd.DataFrame()
    count=0
    for app in apps:
        app_frames = []
        app_frame = build_local_frame(directory, app, feature)
        directory_frame = directory_frame.join(app_frame, how='outer').fillna(0)
        count+=1
        printProgressBar(
            iteration=count,
            total=len(apps),
            prefix="Progress: ",
            suffix="Complete"
        )
    return directory_frame 

def count_codeblocks(directory, apps):
    """Returns list of counts, where each count is the number of codeblocks within an app
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param apps : list
        List of app names (strings) within directory
    """
    cb_cts = []
    for app in apps:
        with open(directory + "/" + app, "r") as f:
            codeblocks = json.load(f)
        count = 0
        for codeblock in codeblocks:
            count += 1
        cb_cts.append(count)
    
    return cb_cts

def build_local_frame(directory, app, feature):
    """Builds dataframe for feature within an app and returns it
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param app : str
        Name of app
    """        
    app_dict = build_global_freq_dict(directory, app, feature, {})
    
    #features are rows and app name is column 
    return pd.DataFrame(app_dict.values(), columns=[app], index=app_dict.keys())
    
    
def build_global_freq_dict(directory, app, feature, global_dict):
    """Builds or augments global dictionary for specified feature from app data and returns it
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param app : str
        Name of app
        
    :param feature : str
        Feature to extract (['api', 'invoke'])   
        
    :param global_dict: dict, optional
        Global dictionary of feature to build or augment 
        If none given, initialize empty dictionary
    """
    if feature == 'api':
        return build_api_dict(directory, app, global_dict) 
        
    elif feature == 'invoke':
        return build_invoke_method_dict(directory, app, global_dict)
    
    elif feature == 'package':
        return build_package_dict(directory, app, global_dict)
            
    return global_dict

def build_api_dict(directory, app, global_dict):
    """Adds api calls and frequencies for a single app to global api dictionary and returns it
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param app : str
        Name of app
        
    :param global_dict : dict
        Global api dictionary to augment
    """
    with open(directory + "/" + app, "r") as f:
            codeblocks = json.load(f)

    for codeblock in codeblocks:
        for method in codeblock:
            api_call = method.split("}, ")
            if api_call[-1] not in global_dict:
                global_dict[api_call[-1]] = 1
            else:
                global_dict[api_call[-1]] += 1
    return global_dict


def build_invoke_method_dict(directory, app, global_dict):
    """Adds invoke methods and frequencies for a single app to global invoke method dictionary and returns it
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param app : str
        Name of app
        
    :param global_dict : dict
        Global api dictionary to augment
    """
    with open(directory + "/" + app, "r") as f:
            codeblocks = json.load(f)

    for codeblock in codeblocks:
        for method in codeblock:
            try:
                invoke = re.findall('(?<=\-)(.+?)(?=\{)', method)[0].strip()
            except: 
                continue
            if invoke not in global_dict.keys():
                global_dict[invoke] = 1
            else:
                global_dict[invoke] += 1
    return global_dict


def build_package_dict(directory, app, global_dict):
    """Adds method packages and frequencies for a single app to global package dictionary and returns it
    
    :param directory : str
        Filepath to directory containing app data 
        
    :param app : str
        Name of app
        
    :param global_dict : dict
        Global package dictionary to augment
    """
    with open(directory + "/" + app, "r") as f:
            codeblocks = json.load(f)

    for codeblock in codeblocks:
        for method in codeblock:
            api_call = method.split("}, ")[-1]
            package_list = re.findall('^[/a-zA-z]+;{1}', api_call)
            if len(package_list) > 0:
                package = package_list[0].strip(';')
                if package not in global_dict.keys():
                     global_dict[package] = 1
                else:
                    global_dict[package] += 1
    return global_dict

def log_model(df_feats):
    print("MODEL: ")
    print("-------")

    X = df_feats.drop("label",axis = 1).to_numpy().reshape(-1, len(df_feats.columns) -1)
    y = df_feats["label"].to_numpy().reshape(-1, 1)
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X,y)
    preds = model.predict_proba(X)
    y_pred = []
    for i in preds:
        if i[0] > i[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)


    print("COEFS: " + str(model.coef_))  
    print("MODEL SCORE: " + str(model.score(X, y.ravel())))
    print("BALANCED ACC SCORE: "+str(balanced_accuracy_score(y,y_pred)))
    print()
    print("CONFUSION MATRIX: ")
    print(confusion_matrix(y, y_pred))
