import os
import pandas as pd
import numpy as np


# Count the number of occurences of every hashtag in the JSON
def hashtag_counts(json, normalize = False):
    print(json)
    df = pd.read_json(json, lines = True)
    if len(df) == 0:
        return pd.Series([])
    ht = df['entities'].apply(lambda e: [x['text'] for x in e['hashtags']])
    return pd.Series(ht.sum()).value_counts(normalize=normalize)


# Count the number of posts every user has made in the JSON
def user_counts(json, normalize = False):
    df = pd.read_json(json, lines=True)
    us = df['user'].apply(lambda x: x['screen_name'])
    return us.value_counts(normalize=normalize)


# Count either hashtags or users in all available JSON files
def count_features(jsons, top_k = None, mode = 'hashtag', normalize = False):
    # Decide whether to count hashtags or users
    if mode == 'hashtag':
        method = hashtag_counts
    elif mode == 'user':
        method = user_counts
        
    # Compile count of first JSON in list
    total_series = method(jsons[0], normalize)
    print(f'vc shape {total_series.shape}')
    
    if len(jsons) > 1:
        # Append counts to every subsequent JSON
        for json in jsons[1:]:
            vc_series = method(json, normalize)
            total_series = total_series.add(vc_series, fill_value = 0)
            print(f'vc shape {total_series.shape}')

    # Return the top users/hashtags in all of the data
    return total_series.sort_values().sort_values(ascending=False)