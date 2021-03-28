import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import find_peaks

sys.path.insert(0, '../src')
from utils import explode_extended, chunk_data

mbit_rate = 1/125000 

def hard_threshold_peaks(df, col, thresh):
    """
    takes in a dataframe and a column. finds the location of spikes given by
    thresh. thresh is a megabit value
    """
    x = df[col]
    peaks, _ = sp.signal.find_peaks(x, height=thresh)
    return peaks 
  
def peak_features(df, col, threshold):
    """
    threshold is a megabit value that is converted to bytes
    """
    mbps_thresh = threshold * 125000
    #peaks = df[get_peak_loc(df, col)]
    peaks = df.iloc[hard_threshold_peaks(df, col, mbps_thresh)]
    
    if len(peaks) <= 0:
        return [0, 0, 0, 120]

    return [np.mean(peaks)[col] * mbit_rate, np.std(peaks)[col] * mbit_rate, len(peaks), 120 / len(peaks)]

def power_density(df, bins, sampling):
    f, psd = sp.signal.welch(df['pkt_size'], fs=sampling) 
    psd = np.sqrt(psd)
    freq = np.linspace(0, np.max(f) + .01, num=bins) - .001
    total = np.trapz(y=psd, x=f)
    psd_den_lst = []

    for i in np.arange(len(freq) - 1):

        f_lower = np.where(f >= freq[i])
        f_upper = np.where(f < freq[i+1] )
        selected_range = np.intersect1d(f_lower, f_upper)

        psd_den = np.trapz(y=psd[selected_range], x=f[selected_range]) / total
        psd_den_lst.append(psd_den)

    return psd_den_lst
  
def spectral_features(df, col, sampling):

    """
    welch implemention of spectral features
    resample the data before inputting (might change prereq depending on
    resource allocation)
    """

    _, Pxx_den = sp.signal.welch(df[col] / 125000, fs=sampling)
    Pxx_den = np.sqrt(Pxx_den)

    peaks = sp.signal.find_peaks(Pxx_den)[0]
    prominences = sp.signal.peak_prominences(Pxx_den, peaks)[0]

    idx_max = prominences.argmax()

    return [np.std(prominences), prominences[idx_max]]

def normalized_std(df, col):
    df_avg = np.mean(df[col])
    df_std = np.std(df[col])

    return df_std / df_avg

def rolling_normalized_std(df, sample_size):
    df_roll = df.rolling(sample_size).mean()
    roll_cv = normalized_std(df_roll, 'pkt_size')
    return roll_cv
  
# Create Features 
def create_features(path, interval, threshold, prominence_fs, binned_fs):
    vals = []

    mbit = 125000

    df = pd.read_csv(path)
    df_chunks = chunk_data(df, interval)

    for chunk in df_chunks:

        extended = explode_extended(chunk)
        resample_1s = extended.resample('1000ms', on='Time').sum()
        resample_2s = extended.resample('2000ms', on='Time').sum() 

        # average amount of bytes/second in a chunk
        download_avg = np.mean(chunk['2->1Bytes']) / mbit
        download_std = np.std(chunk['2->1Bytes']) / mbit
        # diff_pkts = np.mean(chunk['2->1Pkts'] - chunk['1->2Pkts'])

        # peak - average, # peaks, seconds:peak ratio
        peak_feats = peak_features(chunk, '2->1Bytes', threshold)

        # spectral - standard deviation of densities, prominence 
        # psd_density = power_density(resample_2s, 3, binned_fs)
        # psd_density_stdev = np.std(psd_density)
        prominence_feat = spectral_features(resample_1s, 'pkt_size', prominence_fs)

        # coefficient of variation
        rolling_cv = rolling_normalized_std(resample_1s, 8)

        chunk_feat = np.hstack((
            download_avg,
            download_std,
            # diff_pkts,
            peak_feats,
            # psd_density_stdev,
            prominence_feat,
            rolling_cv
        ))

        vals.append(chunk_feat)
    
    return vals

# Create Training Data
def create_training_data(folder_path, interval, threshold, prominence_fs, binned_fs):
    resolutions = ['144p', '240p', '360p', '480p', '720p', '1080p']

    features = [
        "download_avg", 
        "download_std",
        # "diff_pkts",
        "peak_avg",
        "peak_std",
        "peak_amount", 
        "seconds_per_peak", 
        # "psd_std", 
        "prominence_std",
        "max_prominence",
        "rolling_cv" 
    ]   

    training = []

    for res in resolutions:
        full_path = folder_path + res +'/'
        data_dir = [full_path + fp for fp in os.listdir(full_path)]

        data_feats = np.vstack(
            ([create_features(fp, interval, threshold, prominence_fs, binned_fs) for fp in data_dir]))

        temp_df = pd.DataFrame(data=data_feats, columns=features)
        
        if res in ['144p', '240p']:
            temp_df['resolution'] = 'low'

        if res in ['360p', '480p']:
            temp_df['resolution'] = 'medium'

        if res in ['720p', '1080p']:
            temp_df['resolution'] = 'high'

        training.append(temp_df)
    
    return pd.concat((training))