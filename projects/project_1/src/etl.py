import numpy as np
import pandas as pd
import json
import os
import requests
BASE_REPO_2020 = "https://raw.githubusercontent.com/echen102/us-pres-elections-2020/master/"

def get_data(dates_path, start_date_2016, end_date_2016, start_date_2020, end_date_2020, raw_data_dir, tweet_dir, sample_prop, filenames_2016):
    '''Download data for dates of interest from GitHub, unzip tweet id files'''
    # Make separate directories for tweet ids and hydrated tweet files
    os.system("mkdir " + tweet_dir + "/2016/tweet_ids")
    os.system("mkdir " + tweet_dir + "/2016/hydrated_tweets")
    os.system("mkdir " + tweet_dir + "/2020/tweet_ids")
    os.system("mkdir " + tweet_dir + "/2020/hydrated_tweets")
    
    get_2016_data(raw_data_dir, tweet_dir, filenames_2016, sample_prop)
    get_2020_data(tweet_dir, start_date_2020, end_date_2020, sample_prop)
    return

def get_2020_data(tweet_dir, start_date, end_date, sample_prop):
    start_month = start_date[5:7]
    end_month = end_date[5:7]
    for month in range(int(start_month), int(end_month)+1): # Each month has a different folder
        ids_this_month = np.array([])
        if month == int(start_month): # Don't want to fetch all hours in that month's folder if start in middle of month
            start_day = start_date
            end_month = str(int(month) + 1)
            if len(end_month) == 1:
                end_month = "0" + end_month
            end_day = "2020-" + end_month
        elif month == int(end_month): # Don't want to fetch all hours in that month's folder if end in middle of month
            end_day = end_date
            start_month = str(month)
            if len(start_month) == 1:
                start_month = "0" + start_month
            start_day = "2020-" + start_month
        else:
            end_month = str(month + 1)
            start_month = str(month)
            if len(start_month) == 1:
                start_month = "0" + start_month
            if len(end_month) == 1:
                end_month = "0" + end_month
            start_day = "2020-" + start_month
            end_day = "2020-" + end_month

        hours = pd.date_range(start_day, end_day, freq="H").tolist()[:-1]
        month_folder = BASE_REPO_2020 + start_day[:7] + "/"
        # os.system("mkdir " + tweet_dir + "/2020/tweet_ids/" + start_day[:7])
        for this_hour in hours:
            hour = str(this_hour.hour)
            if len(hour) == 1:
                hour = "0" + hour
            hour_str = str(this_hour.date()) + "-" + hour
            url = month_folder + "us-presidential-tweet-id-" + hour_str + ".txt"
            resp_txt = requests.get(url).text # Get ids for this day
            twt_ids = np.array(resp_txt.split('\n'))[:-1]
            num_to_sample = int(len(twt_ids)/sample_prop)
            sampled_this_day = np.random.choice(twt_ids, num_to_sample, replace=False) # Sample
            ids_this_month = np.concatenate([ids_this_month, sampled_this_day]) # Add to array for this month
            
        np.savetxt(tweet_dir + "/2020/tweet_ids/" + start_day[:7] + "_tweet_ids.txt", ids_this_month, fmt='%s')

    for month in range(int(start_month), int(end_month)+1): # Each month has a different folder
        ids_this_month = np.array([])
        if month == int(start_month): # Don't want to fetch all hours in that month's folder if start in middle of month
            start_day = start_date
            end_month = str(int(month) + 1)
            if len(end_month) == 1:
                end_month = "0" + end_month
            end_day = "2020-" + end_month
        elif month == int(end_month): # Don't want to fetch all hours in that month's folder if end in middle of month
            end_day = end_date
            start_month = str(month)
            if len(start_month) == 1:
                start_month = "0" + start_month
            start_day = "2020-" + start_month
        else:
            end_month = str(month + 1)
            start_month = str(month)
            if len(start_month) == 1:
                start_month = "0" + start_month
            if len(end_month) == 1:
                end_month = "0" + end_month
            start_day = "2020-" + start_month
            end_day = "2020-" + end_month

        tweet_dir += "/2020"
        ids_dir = tweet_dir + "/tweet_ids/"
        hydrated_dir = tweet_dir + "/hydrated_tweets/"
        # Hydrate sampled ids, save to hydrated directory
        print(start_day[:7])
        os.system("twarc hydrate " + ids_dir + start_day[:7] + "_tweet_ids.txt > " + hydrated_dir + start_day[:7] + "_hydrated.jsonl")
    return

def get_2016_data(raw_data_dir, tweet_dir, filenames_2016, sample_prop):
    tweet_dir += "/2016/"
    ids_dir = tweet_dir + "/tweet_ids/"
    hydrated_dir = tweet_dir + "/hydrated_tweets/"
    for f_name in filenames_2016:
        twt_ids = np.loadtxt(tweet_dir + f_name, dtype="str", usecols=0)
        num_to_sample = int(len(twt_ids)/sample_prop)
        sampled = np.random.choice(twt_ids, num_to_sample, replace=False) # Sample
        np.savetxt(ids_dir + f_name[:-4] + "_tweet_ids.txt", sampled, fmt='%s')
        os.system("twarc hydrate " + ids_dir + f_name[:-4] + "_tweet_ids.txt > " + hydrated_dir + f_name[:-4] + "_hydrated.jsonl")
    return