import os
import subprocess
import numpy as np
import pandas as pd
import requests
import gzip
import shutil
import json

from twarc import Twarc
from datetime import datetime, date, timedelta
from utils import get_project_root


root = get_project_root()
raw_data_path = os.path.join(root, 'data', 'raw', 'election')
json_data_path = os.path.join(root, 'data', 'processed', 'election')

def configure_twarc():
    """Passes api credentials into Twarc"""
    t = Twarc(
        os.getenv('CONSUMER_KEY'),
        os.getenv('CONSUMER_SECRET'),
        os.getenv('ACCESS_TOKEN'),
        os.getenv('ACCESS_TOKEN_SECRET')
    )
    return t

def generate_dataset(trange='all'):
    """Downloads tweet ID's in the specified date range"""
    
    date_range_fp = os.path.join(root, 'config', 'election_range.json')
    date_range_dict = json.load(open(date_range_fp))
    
    from_time = date_range_dict['from_time']
    to_time = date_range_dict['to_time']
    
    # Get days of data between range
    days_between = pd.date_range(from_time, to_time, freq='d')
    
    # Download the data from each day
    for date in days_between:
#         print(date)
        get_data_of_day(date, raw_data_path, trange=trange)


def get_data_of_day(date, raw_data_path, trange = 'all'):
    year = str(date.year)
    month = '%02d'% date.month
    day = '%02d'%date.day
    folder = month + '/' + day
    f_date = '-'.join([year, month, day])
    print(f'downloading {f_date}')
    save_path = os.path.join(raw_data_path, folder)
    try:
        # Make directories if they don't already exist
        os.makedirs(save_path)
    except Exception as e:
        print(e)
        pass
    year = date.year
    
    if trange == 'all':
        for hour in range(24):  
            hour ='%02d'%hour
            download_url = f'https://raw.githubusercontent.com/echen102/us-pres-elections-2020/master/{year}-{month}/us-presidential-tweet-id-{f_date}-{hour}.txt'
#             print(f'downloading from {download_url}')
            try:
                filename = f'{f_date}-{hour}.txt'
                if filename in os.listdir(save_path):
                    print(f'{filename} already downloaded, skipping')
                else:
                    filepath = os.path.join(save_path, filename)
                    with open(filepath, 'wb') as f:
                        r = requests.get(download_url)
                        f.write(r.content)
            except Exception as e:
                print(f'error encountered while downloading data for {date}-{hour}', e)

# For a day's worth of tweets, sample one out of every number of tweets
def sample_file(day_folder, sample_rate):
    result = pd.Series([])
    # for every hour in the folder
    for file in os.listdir(day_folder):
#         print(f'at hour {file}')
        file_path = os.path.join(day_folder, file)
        s = pd.read_csv(file_path, header = None)[0].iloc[::sample_rate]
        result = result.append(s, ignore_index=True)
    return result

def rehydrate_tweets():
    # Start and configure Twarc
    t = configure_twarc()
    date_range_fp = os.path.join(root, 'config', 'election_range.json')
    date_range_dict = json.load(open(date_range_fp))
    sample_rate = date_range_dict["sample_rate"]
    
    try:
        os.makedirs(json_data_path)
    except:
        pass
        
    # Find out which days of Twitter data we haven't sampled from
    all_dates = set({})
    for month in os.listdir(raw_data_path):
        for day in os.listdir(os.path.join(raw_data_path, month)):
            all_dates.add('-'.join([str(month), str(day)]))
    json_names = set([name.split('.')[0] for name in os.listdir(json_data_path) if '-' in name])
    missing_dates = all_dates - json_names
    print(f'Here are the missing JSONs: {missing_dates}')
        
    # Rehydrate data from days we haven't rehydrated from yet
    for date in sorted(missing_dates):
        # Sample a subset of data from our raw ID's
        print(f'rehydrating tweets on {date}')
        month, day = date.split('-')[0], date.split('-')[1]
        date_path = os.path.join(*[raw_data_path, month, day])
        data_sample = sample_file(date_path, sample_rate)
        
        # Generate a directory/filename to save our hydrated tweets
        name = date + '.jsonl'
        target_path = os.path.join(json_data_path, name)
        print(f'saving to {target_path}')
        
        # Write to file
        with open(target_path, 'w') as outfile:
            for tweet in t.hydrate(data_sample):
                outfile.write(json.dumps(tweet) + '\n')