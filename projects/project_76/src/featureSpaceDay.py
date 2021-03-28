import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
import sys
import os
import json

from itertools import chain
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\src")
import glob
from pandas_datareader import data as pdr


''' 
This method would give you the feature Space of all the stocks for a certain number of days for 
and starting at at a certain day.For example, for day 3 and 3 numDays, we will give a feature
space of all the stocks on the third day + 3 more days so from day three through day six. It will
also return a label specifying a 1, if on the next day the Closing price closes higher than on the
opening price, and a -1 if the opening is higher.

Parameters:
day (int): The starting day of the feature space
numDays (int) : The number of days that we would use on top of the specified day

Return:
featureSpace (DataFrame) : Columns will be the numDays * 4 and the len(rows) will have the number of 
stocks in the datasets.
'''

def featureDaySpace(day,numDays):
    labels = [] # array of 30 labels for each stock
    with open('./config/model-params.json') as f:
        p = json.loads(f.read())
        filepath = p['filepath']
    
    featureVals = []
    #temp add
    #for x in tickers:
    dirs = os.listdir(filepath)
    for x in dirs:
        if '.csv' in x:
            df = pd.read_csv(filepath + x, error_bad_lines=False);
            #get the labels by seeing if the next day closing > opening 
            dayAfterRow = df.iloc[day+numDays]
            #open price at index 0 
            openPrice = dayAfterRow.iloc[0]
            #close at index 3
            closePrice = dayAfterRow.iloc[3]

            if closePrice > openPrice:
                labels.append(1)
            else:
                labels.append(0)
                
            #get the feature space
            df = df.drop(columns = ['Adj Close','Volume','Date'])


            featureVals.append(list(chain.from_iterable(df.iloc[day:day+numDays].values.tolist())))

    tempCol = ['Open','High','Low','Volume']
    col = []
    for x in range(numDays):
        for y in tempCol:
            col.append(y + ' Day ' + str(x) )
    
    featurespace = pd.DataFrame(featureVals, columns = col) 
    
    return featurespace, labels
