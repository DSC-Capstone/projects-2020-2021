'''
build_features.py is used to do feature extractions and will return a dataframe for model training.
'''
## import library
import pandas as pd
import os
import sys
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from scipy import signal


def extract_valid_package_rate(entries,raw_path):
    '''
    return the valid package rate feature for each network-stats record.
    
    :param: entries: a list contains all names of network-stats records.
    :param: raw_path: file directory where raw data stored.
    '''
    feature = []
    for i in entries:
        temp = raw_path +"/"+ i
        file = pd.read_csv(temp, index_col=0).reset_index()
        group_file = file.groupby("Time").sum().reset_index()
        vpr = (len(group_file)-1)/(group_file["Time"][len(group_file)-1]-group_file["Time"][0])
        feature.append(vpr)
    return feature

def extract_peaks_gap(entries,raw_path):
    '''
    return the peaks gap feature for each network-stats record.
    
    :param: entries: a list contains all names of network-stats records.
    :param: raw_path: file directory where raw data stored.
    '''
    feature = []
    for i in entries:
        temp = raw_path +"/"+ i
        file = pd.read_csv(temp, index_col=0).reset_index()
        peaks, _ = find_peaks(file['2->1Bytes'], height=np.mean(file['2->1Bytes']))
        lst = file["Time"][peaks]
        gap = 0
        for i in range(len(lst)-1):
            sub = lst.iloc[i+1] - lst.iloc[i] - 1
            gap = sub + gap
        feature.append(gap)
    return feature

def extract_peaks_number(entries,raw_path):
    '''
    return the peak number feature for each network-stats record.
    
    :param: entries: a list contains all names of network-stats records.
    :param: raw_path: file directory where raw data stored.
    '''
    feature = []
    for i in entries:
        temp = raw_path +"/"+ i
        file = pd.read_csv(temp, index_col=0).reset_index()
        peaks, _ = find_peaks(file['2->1Bytes'], height=np.mean(file['2->1Bytes']))
        feature.append(len(peaks))
    return feature

def extract_max_prominence(entries,raw_path):
    '''
    return the max prominence feature for each network-stats record.
    
    :param: entries: a list contains all names of network-stats records.
    :param: raw_path: file directory where raw data stored.
    '''
    max_prominence_feature=[]
    for i in entries:
        temp = raw_path +"/"+ i
        df=pd.read_csv(temp, index_col=0).reset_index()
        df_temp = df[['Time', '2->1Bytes']].set_index('Time')
        df_temp.index = pd.to_datetime(df_temp.index,unit='s')
        df_temp = df_temp.resample('500ms').sum()
        mean1 = df_temp['2->1Bytes'].mean()
        s = df_temp['2->1Bytes'] - mean1
        s.loc[s < 0] = 0
        fs = 2
        f, Pxx_den = signal.welch(s, fs, nperseg=len(s))
        peaks, properties = signal.find_peaks(np.sqrt(Pxx_den), prominence=1)
        max_prominence = properties['prominences'].max()
        #appends the created value to feature list
        max_prominence_feature.append(max_prominence)
    return max_prominence_feature

def extended_2to1(df):
    df = df[['packet_times', 'packet_sizes', 'packet_dirs']]
    df = df.apply(lambda x: x.str.split(';').explode())
    df = df.loc[df['packet_dirs'] == '2'].reset_index() 
    df = df.dropna(subset=['packet_sizes'])
    df['packet_sizes'] = df['packet_sizes'].astype(int)
    return df
    
    


def spectral_features(entries, raw_path):
    max_prom_norm, peak_0p1Hz_norm, peak_0p2Hz_norm, pct_zeros = [], [], [], []
    for i in entries:
        temp = raw_path + "/" + i
        df=pd.read_csv(temp, index_col=0).reset_index()
        #working with dataset #1
        df1 = extended_2to1(df)
        df1 = df1[['packet_times', 'packet_sizes']].set_index('packet_times')
        df1.index = pd.to_datetime(df1.index,unit='ms')
        df1 = df1.resample('200ms').sum()
        s1 = df1['packet_sizes']/1e6    
        fs = 5
        num_windows = 3
        f1, Pxx_den1 = signal.welch(s1, fs, nperseg=len(s1)/num_windows)
        peaks1, properties1 = signal.find_peaks(np.sqrt(Pxx_den1), prominence=0.001)
        max_prominence_feature1 = properties1['prominences'].max()
        # Some interesting features
        max_prom_norm1 = max_prominence_feature1/np.mean(np.sqrt(Pxx_den1))
        peak_0p1Hz_norm1 = Pxx_den1[np.where(abs(f1-0.1) == min(abs(f1-0.1)))][0]/np.mean(Pxx_den1)
        peak_0p2Hz_norm1 = Pxx_den1[np.where(abs(f1-0.2) == min(abs(f1-0.2)))][0]/np.mean(Pxx_den1)
        zero_thres=0.01
        pct_zeros1 = 100*np.sum(s1<zero_thres)/len(s1)
        max_prom_norm.append(max_prom_norm1)
        peak_0p1Hz_norm.append(peak_0p1Hz_norm1)
        peak_0p2Hz_norm.append(peak_0p2Hz_norm1)
        pct_zeros.append(pct_zeros1)
    return max_prom_norm, peak_0p1Hz_norm, peak_0p2Hz_norm, pct_zeros



def features_build(indir,outdir,output):
    '''
    return a table to outdir contains features from all records generated by network-stats from indir.
    :param: indir: file directory where raw data stored
    :param: outdir: file directory where generated data stored
    :param: output: whether this function output the table as a csv file or not, 2 for predicting, 1 means output, 0 means not
    '''
    entries = os.listdir(indir)
    if ".ipynb_checkpoints" in entries:
        entries.remove(".ipynb_checkpoints")
    features_name = ["valid_package_rate","peaks_gap","peaks_number"]
    feat1 = extract_valid_package_rate(entries,indir)
    feat2 = extract_peaks_gap(entries,indir)
    feat3 = extract_peaks_number(entries,indir)

    #feat4 = extract_max_prominence(entries,indir)
    label = []
    for i in entries:
        if "live" in i:
            label.append(1)
        if "streaming" in i:
            label.append(0)
    data_tuples = list(zip(feat1,feat2,feat3,label))
    tab = pd.DataFrame(data_tuples, columns=['valid_package_rate','peaks_gap','peaks_number','data_label'])
    tab["valid_package_rate"] = tab["valid_package_rate"].fillna(1)
    if output == 1:
        tab.to_csv (outdir+'/features.csv', index = False, header=True)

    feat4, feat5, feat6, feat7 = spectral_features(entries, indir)
    if output!=2:
        label = []
        for i in entries:
            if "live" in i:
                label.append(1)
            if "streaming" in i:
                label.append(0)
        data_tuples = list(zip(feat1,feat2,feat3,feat4,feat5,feat6,feat7,label))
        tab = pd.DataFrame(data_tuples, columns=["valid_package_rate","peaks_gap","peaks_number","max_prom_norm", "peak_0p1Hz_norm", "peak_0p2Hz_norm", "pct_zeros",'data_label'])
        tab["valid_package_rate"] = tab["valid_package_rate"].fillna(1)
        if output == 1:
            tab.to_csv (outdir+'/features.csv', index = False, header=True)
        else:
            return tab

    else:
        data_tuples = list(zip(feat1,feat2,feat3,feat4,feat5,feat6,feat7))
        tab = pd.DataFrame(data_tuples, columns=["valid_package_rate","peaks_gap","peaks_number","max_prom_norm", "peak_0p1Hz_norm", "peak_0p2Hz_norm", "pct_zeros"])    
        tab["valid_package_rate"] = tab["valid_package_rate"].fillna(1)
        return tab   
    

