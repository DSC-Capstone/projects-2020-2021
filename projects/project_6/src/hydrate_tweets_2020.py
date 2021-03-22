import os
import subprocess
import numpy as np
import requests
import sys
import datetime
import re

"""

Script to hydrate the tweets for the 2020 election

Run this command by doing the following: python hydrate_tweets_2020.py <tweet_data_folder> <output_folder>

Example: python hydrate_tweets_2020.py ../data/election_data_2020 ../data/hydrated_tweets/2020

File structure:
root
    data
        democrats
        republicans
        election_data_2020
        hydrated_tweets
            2016
            2020
    scripts
    src
    notebooks

"""

# dates of interest
# START_DATE = '2020-7-13'
# END_DATE = '2020-11-10'


# def get_valid_files(file_list):
#     """Returns the txt files within the given date range"""
    
#     sdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d').date()
#     edate = datetime.datetime.strptime(END_DATE, '%Y-%m-%d').date()

#     valid_files = []
#     for file in file_list:
#         day = file.split('_')[0]
#         try:
#             date = datetime.datetime.strptime(day, '%Y-%m-%d').date()
#             if sdate <= date <= edate:
#                 valid_files.append(file)
#             else:
#                 pass
#         except ValueError:
#             pass
#     # getting a subset of the data - only the 0th file per date
#     valid_dates = []
#     for elem in valid_files:
#         valid_dates.append(elem.split('_')[0])
        
#     to_hydrate = []
#     for elem in valid_dates:
#         to_hydrate.append(elem + '_0_ids.txt')
    
#     return to_hydrate

def hydrate_ids(txt_path, jsonl_path):
    """Retrieves tweets given IDs in txt_path using twarc."""
    hydrate_cmd = 'twarc hydrate ' + txt_path + ' > ' + jsonl_path
    subprocess.run(hydrate_cmd, shell=True)

if __name__ == '__main__':
    
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]
        
    file_list = os.listdir(data_folder)
    existing_jsonls = os.listdir(output_folder)
#     valid_files = get_valid_files(file_list)
    print("Already have: ", existing_jsonls)
    for file in file_list:
        txt_path = os.path.join(data_folder, file)
        
        jsonl_file = file.replace('txt', 'jsonl')
        if jsonl_file not in existing_jsonls:
            jsonl_path = os.path.join(output_folder, jsonl_file)

            print('hydrating ' + txt_path)
            hydrate_ids(txt_path, jsonl_path)
        else:
            pass
    print('\n Done Hydrating! \n')
