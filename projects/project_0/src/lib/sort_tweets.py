import os 
import json
from datetime import datetime

def date_to_datetime(tweet):
    '''
    takes in a tweet 
    outputs a datetime object with day, month, year, time to use for sorting
    '''
    created_at = tweet['created_at'].split()
    parsed = ' '.join([created_at[1], created_at[2], created_at[-1], created_at[3]])
    
    return datetime.strptime(parsed, '%b %d %Y %H:%M:%S')

def sort_files(path):
    '''
    takes in a datapath to the folder where politician's tweets are stored in jsonl format
    
    sorts each politician's tweets in their files and saves them into a new file 
    which is located in a newly created folder 'sorted' located in the same folder 
    '''
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(subdir, file)
            
            if 'sorted' in filepath:
                continue
            
            if not file.endswith('.jsonl'):
                continue 
                
            tweets = []
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    tweets.append(tweet)
            
            tweets.sort(key=date_to_datetime)
            
            if not os.path.exists(os.path.join(subdir, 'sorted')):
                os.makedirs(os.path.join(subdir, 'sorted'))
            
            outpath = os.path.join(subdir, 'sorted', file)
            with open(outpath, 'w+', encoding='utf-8') as f:
                for t in tweets:
                    f.write(json.dumps(t) + '\n')