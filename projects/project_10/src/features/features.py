import pandas as pd
import glob
import os
import itertools
import numpy as np
import scipy
from scipy.signal import peak_prominences
import warnings
warnings.filterwarnings("ignore")

from src.features.feature_creation import split
from src.features.feature_creation import roll
from src.features.feature_creation import longest_dir_streak

import multiprocessing
import time

import logging

from src.utils import ensure_path_exists

def _engineer_features(args):
    return engineer_features(*args)
    
def engineer_features(
    df, label, rolling_window_1, rolling_window_2, resample_rate,
    frequency
    ):
    print('beginning features for ' + str(label))
    df['dt_time'] = pd.to_timedelta(df['dt_time'])
    df = df.set_index('dt_time')
    df = df.drop(columns=['binned'])

    #drop rows with any nan
    df = df.dropna(how='any')

    #flow level statistics
    sent_bytes = df[df['dir'] == 1]['size'].sum()
    received_bytes = df[df['dir'] == 2]['size'].sum()
    sent_packets = len(df[df['dir'] == 1])
    received_packets = len(df[df['dir'] == 2])

    #if there are no received bytes or packets, skip this chunk
    if received_bytes == 0 or received_packets == 0:
        print('no received bytes/packets')
        return None
    print('multiple bytes')
    received_mean_size = df.loc[df.dir==2, 'size'].mean()
    sent_mean_size = df.loc[df.dir==1, 'size'].mean()

    #ratio of sent bytes over received bytes
    bytes_ratio = sent_bytes / received_bytes

    #ratio of sent packets over received packets
    count_ratio = sent_packets / received_packets

    #ratios
    #large packet is defined as any packet size over 1200 byes
    #small packet is defined as any packet under 200 bytes
    sent_large = df[(df['dir']==1) & (df['size'] > 1200)]
    sent_small = df[(df['dir']==1) & (df['size'] < 200)]
    received_large = df[(df['dir']==2) & (df['size'] > 1200)]
    received_small = df[(df['dir']==2) & (df['size'] < 200)]
    
    try:
        #ratio of large, uploaded packets over all uploaded packets 
        sent_large_prop = len(sent_large) / len(df[(df['dir']==1)])
    except:
        sent_large_prop = 0
    try:
        #ratio of small, uploaded packets over all uploaded packets
        sent_small_prop = len(sent_small) / len(df[(df['dir']==1)])
    except:
        sent_small_prop = 0 
        
    try:
        #ratio of large, downloaded packets over all downloaded packets
        received_large_prop = len(received_large) / len(df[(df['dir']==2)])
    except:
        received_large_prop = 0
    try:
        #ratio of small, downloaded packets over all downloaded packets
        received_small_prop = len(received_small) / len(df[(df['dir']==2)])
    except:
        received_small_prop = 0

    
    #interpacket delay
    df['ip_delay'] = df.index.to_series().diff().dt.total_seconds() * 1000
    df = df.dropna(how='any')

    #signal peak prominence using welch's method
    df_rs = df.resample(resample_rate).sum()
    f, Pxx_den = scipy.signal.welch(df_rs['size'], fs = frequency)
    peaks, _ = scipy.signal.find_peaks(np.sqrt(Pxx_den))
    prominences = peak_prominences(np.sqrt(Pxx_den), peaks)[0]
    try:
        df_max_prom = prominences.max()
    except:
        df_max_prom = 0
    
    #interpacket delay means over rolling windows of 10 seconds and 60 seconds
    #
    #need to convert milliseconds to seconds here
    rolling_delays_10 = roll(df, 'ip_delay', int(rolling_window_1/1000))['mean'].mean()
    rolling_delays_60 = roll(df, 'ip_delay', int(rolling_window_2/1000))['mean'].mean()

    
    #----------GROUP FEATURES-----------
    #Arely's
    #ft1: ratio of (number of packet sizes from 0-200 size)/(number of all packets)
    small_packet_ratio = len(df[(df['size'] <200)]) / len(df)
    #ft2: ratio of (upload packet sizes in range [200-400])/(number of all packets)
    medium_packet_ratio = len(df[(df['size'] >200) & (df['size'] <400)]) / len(df)
    #ft3: ratio of (number of pksize >1200)/ (number of all packets)
    large_packet_ratio = len(df[(df['size'] >1200)]) / len(df)
    
    #Chang's
    download_bytes_cv = np.std(df[df['dir'] == 2]['size']) / np.mean(df[df['dir'] == 2]['size'])
    upload_bytes_cv = np.std(df[df['dir'] == 1]['size']) / np.mean(df[df['dir'] == 1]['size'])
    
    features = [
                rolling_delays_10,
                rolling_delays_60,
                received_mean_size,
                sent_mean_size,
                sent_small_prop,
                received_large_prop,
                received_small_prop,
                df_max_prom,
                small_packet_ratio,
                medium_packet_ratio,
                large_packet_ratio,
                download_bytes_cv,
                upload_bytes_cv,
                
                label
            ]
    print('finished features for ' + str(label))
    return features

def create_features(source_dir, out_dir, out_file, chunk_size, rolling_window_1, rolling_window_2, resample_rate, frequency):

    # Ensure that the output directory exists.
    ensure_path_exists(source_dir, is_dir=True)
    ensure_path_exists(out_dir, is_dir=True)

    #splitting dataframe into chunk_size'd chunks
    #chunk size is in milliseconds
    preprocessed_dfs = glob.glob(os.path.join(source_dir, 'preprocessed*'))
    split_df_groups = [split(f, chunk_size) for f in preprocessed_dfs]
    
    #flattening list
    merged = list(itertools.chain.from_iterable(split_df_groups))
    
    #streaming provider label
    merged_keys = [m[0] for m in merged]
    
    #the actual dataframes
    merged_dfs = [m[1] for m in merged]
#     cols = [
#         'bytes_sr_ratio',
#         'count_sr_ratio',
#         'smoothed_mean_delay_10s',
#         'smoothed_mean_delay_60s',
#         'received_mean_size',
#         'sent_mean_size',
#         'sent_large_prop',
#         'sent_small_prop',
#         'received_large_prop',
#         'received_small_prop',
#         'sent_longest_streak',
#         'received_longest_streak',
#         'max_frequency_prominence',

#         'provider'
#     ]
    
    cols = [
        'smoothed_mean_delay_10s',
        'smoothed_mean_delay_60s',
        'received_mean_size',
        'sent_mean_size',
        'sent_small_prop',
        'received_large_prop',
        'received_small_prop',
        'max_frequency_prominence',
        'small_packet_ratio',
        'medium_packet_ratio',
        'large_packet_ratio',
        'download_bytes_cv',
        'upload_bytes_cv',
        'provider']


    args = [
        (merged_dfs[i], merged_keys[i], rolling_window_1,
        rolling_window_2, resample_rate, frequency)
        for i in range(len(merged_dfs))
    ]

    workers = multiprocessing.cpu_count()
    # print(f'Starting a processing pool of {workers} workers.')
    logging.info(f'Starting a processing pool of {workers} workers')
    start = time.time()
    pool = multiprocessing.Pool(processes=workers)
    results = pool.map(_engineer_features, args)
    # print(f'Time elapsed: {round(time.time() - start)} seconds.')
    logging.info(f'Time elapsed: {round(time.time() - start)} seconds.')
    
    features = np.vstack(list(filter(lambda x: x is not None, results)))

    features_df = pd.DataFrame(features, columns=cols).dropna()
    # print(f'{features_df.shape[0]} chunks of data feature engineered.')
    logging.info(f'{features_df.shape[0]} chunks of data feature engineered.')

    features_df.to_csv(os.path.join(out_dir, out_file), index=False)
    # print('Features created: ', list(features_df.columns))
    logging.info(f'Features created: {list(features_df.columns)}')
