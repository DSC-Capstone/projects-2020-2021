import pandas as pd
import numpy as np
import json
from src.utils import get_project_root
import os 

root = get_project_root()
raw_data_path = os.path.join(root, 'data', 'raw', 'election')
json_data_path = os.path.join(root, 'data', 'processed', 'election')
jsons = [os.path.join(json_data_path,name) for name in sorted(os.listdir(json_data_path)) if 'jsonl' in name]

# Count the number of occurences of every hashtag in the JSON
def hashtag_counts(json_path, case_sensitive=False):
#     print('in method')
    chunksize = 1
#     print('preparing reader....!')
    with open(json_path) as reader:
#         print('reader ready')
        ht_counts = dict({})
        if not case_sensitive:
            for line in reader:
                hts = [h['text'] for h in json.loads(line)['entities']['hashtags']]
                for ht in hts:
                    try:
                        ht_counts[ht.lower()] += 1
                    except KeyError:
                        ht_counts[ht.lower()] = 0 
        else:
            for line in reader:
                hts = [h['text'] for h in json.loads(line)['entities']['hashtags']]
                for ht in hts:
                    try:
                        ht_counts[ht] += 1
                    except KeyError:
                        ht_counts[ht] = 1
#     print('finished calculating hashtag counts')
    return pd.Series(ht_counts)


# Count the number of posts every user has made in the JSON
def user_counts(json, case_sensitive=False):
    df = pd.read_json(json, lines=True)
    us = df['user'].apply(lambda x: x['screen_name'])
    return us.value_counts()



# Count either hashtags or users in all available JSON files
def count_features(jsons = jsons, mode = 'hashtag', normalize=True, case_sensitive=False, top_k=300):
    # Decide whether to count hashtags or users
    if mode == 'hashtag':
        method = hashtag_counts
    elif mode == 'user':
        method = user_counts
        
    # Compile count of first JSON in list
    total_series = method(jsons[0], case_sensitive=case_sensitive)
#     print(f'vc shape {total_series.shape}')
    
    if len(jsons) > 1:
        # Append counts to every subsequent JSON
        for json in jsons[1:]:
            vc_series = method(json)
            total_series = total_series.add(vc_series, fill_value = 0)
#             print(f'vc shape {total_series.shape}')

    # Return the top users/hashtags in all of the data
    if top_k is None:
        return total_series.sort_values().sort_values(ascending=False)
    else:
        return total_series.sort_values().sort_values(ascending=False).iloc[:top_k]