import pandas as pd
import numpy as np
import os
import json

def users_by_subreddit(sci_path, poli_path, myth_path):
    paths = [sci_path, poli_path, myth_path]
    users_by_sub = dict()
    
    for data_path in paths:
        df = pd.read_csv(data_path)
        for sub in df['subreddit'].unique():
            users_by_sub[sub] = df['author'].loc[df['subreddit'] == sub]
    return users_by_sub

def shared_users(users_by_sub):
    cross_counts = dict()
    for keys1, values1 in users_by_sub.items():
        for keys2, values2 in users_by_sub.items():
            cross_counts[str((keys1, keys2))] = [pd.Series(list(set(values1).intersection(set(values2)))), len(list(set(values1).union(set(values2))))]
    return cross_counts

def count_matrix(shared_users, save_path, sci, myth, poli, save_name):
    matrix_counts = dict()
    
    for keys, values in shared_users.items():
        matrix_counts[eval(keys)] = len(values[0])/values[1]
        
    ser = pd.Series(list(matrix_counts.values()),
                    index = pd.MultiIndex.from_tuples(matrix_counts.keys()))
    df = ser.unstack().fillna(0)
    
    df = df[sci + myth + poli]
    df = df.reindex(sci + myth + poli)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    df.to_csv(save_path + '/' + save_name)

def polarity_matrix(shared_users, polarity_path, save_path, sci, myth, poli, save_name):
    matrix_polarities = dict()
    polarities = pd.read_csv(polarity_path)
    
    for keys, values in shared_users.items():
        df = pd.DataFrame(values[0]).merge(polarities, how='left', left_on=0, right_on='Unnamed: 0')
        avg_science = df['science (%)'].mean()
        avg_myth = df['myth (%)'].mean()
        avg_politics = df['politics (%)'].mean()
        
        all_means = [avg_science, avg_myth, avg_politics]
        
        for i in range(3):
            if pd.isnull(all_means[i]):
                all_means[i] = 0
        matrix_polarities[eval(keys)] = all_means

    ser = pd.Series(list(matrix_polarities.values()),
                    index = pd.MultiIndex.from_tuples(matrix_polarities.keys()))
    df = ser.unstack().fillna(0)
    
    df = df[sci + myth + poli]
    df = df.reindex(sci + myth + poli)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    df.to_csv(save_path + '/' + save_name)
