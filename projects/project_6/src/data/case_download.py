import numpy as np
import pandas as pd
import os
import datetime

def total_case(tofilename, topath, sdate, edate, url):
    # set up start date and end date
    start_date = datetime.datetime.strptime(sdate, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(edate, '%Y-%m-%d')
    
    # read and clean dataset
    df_case = pd.read_csv(url)
    df_case['date'] = pd.to_datetime(df_case['date'], format='%Y-%m-%d')
    
    # filter out dates
    df_case = df_case[(df_case['date'] >= start_date) & (df_case['date'] < end_date)]
    df_case_sum = df_case[['date', 'new_cases']].groupby(by=['date']).sum().reset_index()
    df_case_sum.to_csv(topath+tofilename,index=False)
    
    return
