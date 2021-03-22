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

def compute_user_stats(logger, tweets, mdls, dims, user_data_path, flagged):
    '''
    Computes user polarities using their tweets and text prediction models.
    '''
    user_entries = []

    # Loop through tweets
    for tweet_id, tweet in tweets.items():
        # Loop through users
        for user_id in tweet['user_ids'].keys():
            df = tweet['user_ids'][user_id]
            user_entry = [tweet_id, tweet['text'], tweet['user/name'], user_id, df.shape[0], flagged[tweet_id]]
            for mdl in mdls:
                if len(df['text']) > 0:
                    y_pred = mdl.predict(df['text'])
                    user_entry.append(np.mean(y_pred))
                else:
                    user_entry.append('N/A')
            user_entries.append(user_entry)

    # Save results
    cols = ['tweet_id', 'tweet_text', 'user/name', 'user_id', 'n_tweets', 'flagged'] + dims
    df = pd.DataFrame(user_entries, columns=cols).round(2)
    df.to_csv(os.path.join(user_data_path, 'polarities.csv'))
    return