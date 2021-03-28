import os
import numpy as np
import pandas as pd

# Reset Repo 
def reset():

    check_files = ['training.csv', 'test_training.csv']

    for fp in check_files:
        if os.path.exists(fp):
            os.remove(fp)

# Chunk Data
def chunk_data(df, interval=120):
    """
    takes in a filepath to the data you want to chunk and feature engineer
    chunks our data into a specified time interval
    each chunk is then turned into an observation to be fed into our classifier
    """
    df_list = []
    
    df['Time'] = df['Time'] - df['Time'].min()
    
    total_chunks = np.floor(df['Time'].max() / interval).astype(int)

    for chunk in np.arange(total_chunks):
      
        start = chunk * interval
        end = (chunk+1) * interval

        temp_df = (df[(df['Time'] >= start) & (df['Time'] < end)])
        df_list.append(temp_df)
        
    return df_list

# Extended Column Cleaning 
def explode_extended(df):
    """
    takes in a network-stats df and explodes the extended columns.
    time is converted from seconds to milliseconds.
    drop the ip address columns and the aggregate columns.
    """
    ext_col = ['packet_times', 'packet_sizes', 'packet_dirs']
    
    pre_explode = df[ext_col].apply(lambda x: x.str.split(';').str[:-1])
    
    exploded = pre_explode[ext_col].apply(lambda x: x.explode(ignore_index=True).astype(np.int64))
    exploded.columns = ['Time', 'pkt_size', 'pkt_dir']

    _sorted = exploded.sort_values('Time')
    _sorted['Time'] = pd.to_datetime(_sorted['Time'], unit='ms')

    return _sorted

# Peak Related 
def get_peak_loc(df, col, strict=1):
    """
    takes in a dataframe, column, and strictness level. threshold is determined
    by positive standard deviations from the average. strictness is default at 1.

    returns an array of peak locations (index).
    """
    threshold = np.mean(df[col]) + (strict * np.std(df[col]))
    return np.array(df[col] > threshold)

# Add Resolution Column
def add_resolution(fp, res):
    temp_df = pd.read_csv(fp)
    temp_df['resolution'] = res
    return temp_df

# Read in all Data
def load_data(resolution, path):
    data_filepath = path + resolution + "/"
    data_dir = os.listdir(data_filepath)
    return [add_resolution(data_filepath + fp, resolution) for fp in data_dir]

