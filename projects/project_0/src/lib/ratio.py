import json
import pandas as pd
import time
import os

import config

def get_ratios(politician_file, filepath):
    '''
    transforms the csv files from the data pipeline and calculates the ratios
    rewrites the existing csv file with the new information
    '''
    
    working_path = os.getcwd()
    os.chdir(filepath)

    csv_file = politician_file + '.csv'
    
    #Checks to make sure the csv file exists in order to calculate ratios
    if csv_file not in os.listdir():
        print("The csv file for " + politician_file + " does not exist.")
        os.chdir(working_path)
        return

    #Reads in the metrics csv obtained from the data pipeline
    metrics = pd.read_csv(csv_file, index_col=0)
    
    #Calculates and gets the ratios using our formula of (2 * replies) / (retweets + likes)
    #Adds the ratio column to the dataframe
    metrics['public_metrics'] = metrics['public_metrics'].apply(lambda x: eval(x))
    ratios_table = pd.concat([metrics, metrics['public_metrics'].apply(pd.Series)], axis=1)
    ratios = (ratios_table['reply_count'] * 2) / ((ratios_table['retweet_count']) + (ratios_table['like_count']))
    ratios = ratios.rename('ratio')
    ratios_table = pd.concat([ratios_table, ratios], axis=1)

    #Only keep relevant columns that we need for ratio analysis
    output = ratios_table[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'ratio']]
    
    #Saves the ratios dataframe to a csv file so we can reduce the number of API calls we make
    output.to_csv(politician_file + '.csv', index=True)
    print("The ratios for " + politician_file + "'s tweets have been calculated.")
    
    os.chdir(working_path)
    
    return ratios_table

def get_ratio_csv(scientific_path, misinformation_path, scientific_list, misinformation_list):
    '''
    setup for calculating csv for both our misinformation and scientific group
    '''

    for i in range(len(scientific_list)):
        get_ratios(scientific_list[i], scientific_path)

    for i in range(len(misinformation_list)):
        get_ratios(misinformation_list[i], misinformation_path)