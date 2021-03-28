# imports and dependencies
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import datetime as dt
import numpy as np

'''
This function is preprocessing the raw data with extracting first hour after market open. (9:15-10:15)
'''

def preprocessing(fileName):
    '''
    fileName: name of the file for the raw data
    '''
    raw = pd.read_csv('../data/raw data/raw_NIFTY100.csv')

    df = raw.copy()
    df.timestamp = pd.to_datetime(df.timestamp)
    df['date'] = df.timestamp.apply(lambda x: x.date)
    df['minute'] = df.timestamp.apply(lambda x: x.time)

    # Getting rid of dates where #data points are insufficient
    a = pd.DataFrame(df.groupby('date').close.count())
    drop_dates = list(a[a.close < 300].index)
    df = df[~df.date.isin(drop_dates)]
    
    first = df.copy()
    first['temp'] = first.minute.apply(first_hour)
    first = first[first.temp==True]
    first = first.drop(['timestamp', 'open', 'high', 'low', 'volume', 'temp'], axis=1)
    
    first.to_csv('../data/first_combined.csv',index=False)
    

def first_hour(time):
    if time.hour == 9:
        if time.minute >=15:
            return True
    elif time.hour == 10:
        if time.minute < 15:
            return True
    else:
        return False