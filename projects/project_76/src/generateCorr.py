import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
import os
import sys
from pandas_datareader import data as pdr

def buildCorrGraph(filepath,threshhold):
    dirs = os.listdir(filepath)
    combined_df = pd.DataFrame()
    for x in dirs:
        if '.csv' in x:
            df = pd.read_csv(filepath + '/' + x, error_bad_lines=False);
            #remove the csv to get the ticker tag
            combined_df[x.replace('.csv','')] = abs(df['Close'].diff())

    combined_df = combined_df.iloc[2:]
    corr_df = combined_df.corr()
    corr4 = corr_df.applymap(lambda x: 0 if x < threshhold else 1)
    if 'dow' in filepath:
        corr4.to_csv('./data/dowJonescorrelation0.4graph.csv')
    else:
        print('in here');
        corr4.to_csv('./data/sp500correlation0.4graph.csv')

