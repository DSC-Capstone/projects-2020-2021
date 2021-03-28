# imports and dependencies
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import datetime as dt
import numpy as np

'''
This function is labeling the image Whether the price goes up or down that day.
If the price goes up, label is 1. If the price goes down or does not change, label is 0
'''

def data_label(fileName):
    '''
    fileName: name of the file for raw data
    '''
    raw = pd.read_csv('../data/raw data/' + fileName)

    label = pd.DataFrame()
    for date in df.date.unique():
        day = df[df.date==date]
        day = day.sort_values('minute')
        first_open = day.iloc[0].open
        last_close = day.iloc[-1].close
        temp = pd.DataFrame([[date, first_open, last_close]])
        label = label.append(temp)


    label.columns = ['date','open','close']
    label['diff'] = label.close - label.open

    def label_convert(diff):
        if diff > 0:
            return '1'
        else:
            return '0'

    label['label'] = label['diff'].apply(label_convert)
    label = label.drop(['open','close','diff'],axis=1)
    
    path = '../data/'
    label.to_csv(path+'/label_dir_2.csv', index=False)
