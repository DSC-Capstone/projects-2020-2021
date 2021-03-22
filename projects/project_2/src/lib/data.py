import tweepy
import config
import time 
import datetime
import numpy as np
import os
import json
import pandas as pd

def rehydrate(directory_path):
    '''
    rehydrates the tweet ids in all txt files in the current directory 
    tweets are saved into a separate file with the same name into jsonl format 
    '''
    #Stores the current path so we can change back to it later
    cwd = os.getcwd()
    
    #Changes the path to the folder containing the files we wish to rehydrate
    os.chdir(directory_path)
    
    #Lists the files so we can iterate through them
    folder = os.listdir()
    
    for file in range(len(folder)):

        #Checks to make sure the jsonl file doesn't already exist for the politician
        politician_check = folder[file][:-4] + '.jsonl'
        if politician_check in folder:
            print("The data for " + folder[file][:-4] + " already exists.")
            continue

        #Checks to make sure file ends in a txt extension 
        if folder[file][-4:] == '.txt':
            #Rehydrates the tweet ids using twarc
            os.system('twarc hydrate ' + folder[file] + ' > ' + folder[file][:-4] + '.jsonl')
            print("Data for " + folder[file][:-4] + " has been obtained.")
        else:
            continue
            
    #Changes the directory to the original path
    os.chdir(cwd)
        

def get_data(scientific_path, misinformation_path):
    '''
    setup for rehyrating tweets for both our scientific and misinformation groups
    '''
    rehydrate(scientific_path)
    rehydrate(misinformation_path)

def get_metrics(politician_file, filepath):
    '''
    obtains the engagement metrics for all the tweet ids in our txt files
    engagement metrics are saved into a separate file with the same name in csv format
    '''
    
    working_path = os.getcwd()
    os.chdir(filepath)
    
    check = politician_file + '.csv'
    
    #Checks to make sure the csv file does not already exist before calling on the API
    if check in os.listdir():
        print("The csv for " + politician_list + " already exists.")
        os.chdir(working_path)
        return
    
    #Opens the text file of tweet ids found in the filepath
    text_file = open(politician_file + ".txt", "r")
    lines = text_file.read().splitlines()
    text_file.close()

    #List containing dataframes consisting of 100 rows
    metrics_list = []

    for i in range(0, len(lines), 100):
        #Gets 100 tweet ids at a time and passes it to the call to the API
        #The API can only look up metrics for 100 tweets at a time
        partition = lines[i:i+100]
        partition_string = ','.join(partition)

        request = os.popen("curl 'https://api.twitter.com/2/tweets?ids=" + partition_string + "&tweet.fields=public_metrics' --header 'Authorization: Bearer " + config.bearer_token + "'").read()

        #Check for rate limit, if exceeded we sleep for 15 minutes to reset the limit
        #Then continues where it left off for the tweet ids
        if 'exceeded' in request:
            print("Rate Limit Exceeded. Sleeping for 15 Minutes.")
            time.sleep(60*15)
            request = os.popen("curl 'https://api.twitter.com/2/tweets?ids=" + partition_string + "&tweet.fields=public_metrics' --header 'Authorization: Bearer " + config.bearer_token + "'").read()

        #A string with json is returned from the API call, this converts it to Json
        to_json = json.loads(request)

        #Takes the data portion of the output and turns it into a dataframe
        #Appends the dataframe of 100 rows to a list containing the smaller dataframes
        metrics_data = pd.DataFrame.from_dict(to_json['data'])
        metrics_list.append(metrics_data)

    #Appends the small dataframes of 100 into a larger dataframe
    metrics = pd.concat(metrics_list, ignore_index=True)
    
    #Saves the metrics dataframe to a csv file so we can reduce the number of API calls we make
    metrics.to_csv(politician_file + '.csv', index=True)
    print("The csv for " + politician_file + " has been completed.")
    
    os.chdir(working_path)
    
    return metrics

def get_csvs(scientific_path, misinformation_path, scientific_list, misinformation_list):
    '''
    setup for obtaining engagement metrics for both our scientific and misinformation groups
    '''

    for i in range(len(scientific_list)):
        get_metrics(scientific_list[i], scientific_path)

    for i in range(len(misinformation_list)):
        get_metrics(misinformation_list[i], misinformation_path)