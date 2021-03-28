import sys
import json
import os
import pandas as pd
import requests
import spotipy
from collections import defaultdict
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
import ipywidgets
from ipywidgets import FloatProgress
from src.models.task2_utils import *
from sklearn import metrics


def get_train_test(history, pct_test = 0.2):
    test_data = history.copy() # Make a copy of the original data to be test data
    test_data[test_data != 0] = 1 
    
    training_data = history.copy() # Make a copy of the original data to alter for training data
    
    nonzero_inds = training_data.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) 

    
    random.seed(0) # For reproducibility
    
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Get the number of samples
    samples = random.sample(nonzero_pairs, num_samples) # Sample without replacement

    artist_indices = [index[0] for index in samples] 

    user_indices = [index[1] for index in samples] 

    
    training_data[artist_indices, user_indices] = 0 
    training_data.eliminate_zeros() 
    
    return training_data, test_data, list(set(user_indices))


def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_avg_auc(training_set, altered_users, predictions, test_set):
    user_auc = [] # User AUC 
    most_popular_auc = [] # Popular AUC
    popular_artists = np.array(test_set.sum(axis = 1)).reshape(-1) 
    artist_vecs = predictions[1]
    for user in altered_users: 
        training_column = training_set[:,user].toarray().reshape(-1) 
        zero_inds = np.where(training_column == 0) 
        
        user_vec = predictions[0][user,:]
        predicted_user_val = user_vec.dot(artist_vecs).toarray()[0,zero_inds].reshape(-1)
        
        actual = test_set[:,user].toarray()[zero_inds,0].reshape(-1)
        
        artist_popularity = popular_artists[zero_inds] 
        
        user_auc.append(auc_score(predicted_user_val, actual)) # Calculate AUC for the given user and store
        
        most_popular_auc.append(auc_score(artist_popularity, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(user_auc)), float('%.3f'%np.mean(most_popular_auc))

def run_auc(item_user_interactions, user_vecs, artist_vecs):
    
    training_data, test_data, artists_users_altered = get_train_test(item_user_interactions, pct_test = 0.2)
    user_vecs = sparse.csr_matrix(user_vecs)
    artist_vecs = sparse.csr_matrix(artist_vecs)
    # Calculate scores
    auc_scores = calc_avg_auc(training_data, artists_users_altered,
              [user_vecs, artist_vecs.T], test_data)
    auc_final_score = auc_scores[0]
    auc_df = pd.DataFrame([auc_final_score], columns=['AUC Score'])
    return auc_df
    