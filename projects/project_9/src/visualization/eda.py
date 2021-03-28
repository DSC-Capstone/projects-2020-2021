import numpy as np 
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy import signal


#TODO FILEPATH for the files we want
filepath = '../test/testdata/'
#grabbing data for streaming and live to do EDA on
twitch_vod = pd.read_csv(filepath + 'maqader-twitch-streaming-1102021-8.csv')
twitch_live = pd.read_csv(filepath + 'maqader-twitch-live-1112021-5.csv')

twitch_vod['Time'] = twitch_vod['Time'] - twitch_vod['Time'][0]
twitch_live['Time'] = twitch_live['Time'] - twitch_live['Time'][0]

def fig1():
    '''
    Creates figure 1 in EDA notebook displaying the number 
    of bytes downloaded for video on demand data
    '''
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Twitch VOD", "Twitch Live"))

    fig.add_trace(go.Scatter(x=twitch_vod['Time'], 
                             y=twitch_vod['2->1Bytes'],
                             name='2->1Bytes', 
                             line=dict(color='rgb(68,0,84)', width = 1)),
                               row=1, col=1)
    
    fig.add_trace(go.Scatter(x=twitch_live['Time'], 
                             y=twitch_live['2->1Bytes'],
                             name='2->1Bytes', 
                             line=dict(color='rgb(50,104,142)', width = 1)),
                               row=1, col=2)
    fig.update_layout(
        title="Number of Bytes Downloaded",
        
   
        yaxis_title="Bytes",
        font=dict(family="Courier New, monospace",size=16,color="Black"))
    fig.update_yaxes(range=[0, 4000000])
    fig.update_xaxes(title_text="Time (Seconds)")
    return fig

#Function to find peaks!
def findPeaks(Data, height_min):
    """
    Finds the peak heights in 2->1 Bytes, and then evaluates the seconds in each interval 
    between peaks
    
    data: takes in a dataframe of network stats data 
    height_min: takes in minimum peak height to create peaks for 
    
    returns array of interval lengths
    """
    x = Data['Time']
    y = Data['2->1Bytes']
    peaks, _ = find_peaks(y, height=height_min)
    s = x[peaks]
    return peaks

#Function to find threshold to use in find-intervalPeaks uses mean! 
def find_threshold(Data):
    """
    Finds the threshold we want to use in finding interval peaks
    
    Data: takes in dataframe of network stats data 
    """
    x = Data['Time']

    y = Data['2->1Bytes']
    peaks, _ = find_peaks(y, height=0)
    mean = y[peaks].mean() 
    return mean

def fig2():
    '''
    Creates figure 2 in EDA notebook displaying the peaks 
    for twitch video on demand 
    '''
    indices = findPeaks(twitch_vod, find_threshold(twitch_live))
    x = twitch_vod['Time']
    y = twitch_vod['2->1Bytes']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Original Plot',line=dict(color='rgb(50,104,142)', width = 1)
    ))

    fig.add_trace(go.Scatter(
        x=x[indices],
        y=[y[j] for j in indices],
        mode='markers',
        marker=dict(
            size=7,
            color='red',
            symbol='cross'
        ),
        name='Detected Peaks'
    ))

    fig.update_layout(
        title="Peaks in Bytes Downloaded Twitch VOD",
        xaxis_title="Time (Seconds)",
        yaxis_title="Bytes",
        font=dict(family="Courier New, monospace",size=16,color="Black"))

    return fig

def fig3():
    '''
    Creates figure 3 in EDA notebook displaying the peaks 
    for twitch live video
    '''
    indices = findPeaks(twitch_live, find_threshold(twitch_live))
    x = twitch_live['Time']
    y = twitch_live['2->1Bytes']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Original Plot',line=dict(color='rgb(68,0,84)', width = 1)
    ))

    fig.add_trace(go.Scatter(
        x=x[indices],
        y=[y[j] for j in indices],
        mode='markers',
        marker=dict(
            size=7,
            color='red',
            symbol='cross'
        ),
        name='Detected Peaks'
    ))

    fig.update_layout(
        title="Peaks in Bytes Downloaded Twitch Live",
        xaxis_title="Time (Seconds)",
        yaxis_title="Bytes",
        font=dict(family="Courier New, monospace",size=16,color="Black"))

    return fig

def fig4():
    vod = pd.DataFrame()
    indices = findPeaks(twitch_vod, find_threshold(twitch_vod))
    x = twitch_vod['Time']
    vod['times'] = x[indices]
    vod['peak'] = 'VOD Peak'

    live = pd.DataFrame()
    indices_2 = findPeaks(twitch_live, find_threshold(twitch_live))
    x_2 = twitch_live['Time']
    live['times'] = x_2[indices_2]
    live['peak'] = 'Live Peak'


    trace1 = go.Scatter(
        x=vod.times,
        y=vod.peak,
        mode='markers',
        marker=dict(
            color='rgb(50,104,142)',
            size=4))

    trace2 = go.Scatter(
        x=live.times,
        y=live.peak,
        mode='markers',
        marker=dict(
            color='rgb(68,0,84)',
            size=2),
        yaxis='y2')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace2,secondary_y=False)
    fig['layout'].update(height = 300, width = 1100, title = 'Time of Peak in 5 minute Chunk of Live vs VOD Video',xaxis=dict(
       tickangle=90, title = 'Time (seconds)'), font=dict(family="Courier New, monospace",size=16,color="Black"))

    fig.update_layout(showlegend=False)
    return fig

def fig5():
    group_file = twitch_vod.groupby("Time").sum().reset_index()
    vpr_vod = (len(group_file)-1)/(group_file["Time"][len(group_file)-1]-group_file["Time"][0])
    
    group_file2 = twitch_live.groupby("Time").sum().reset_index()
    vpr_live = (len(group_file2)-1)/(group_file2["Time"][len(group_file2)-1]-group_file2["Time"][0])
    
    print('The VOD has a valid packet ' + str(vpr_vod * 100) + '% of the time!')
    print('The Live has a valid packet ' + str(vpr_live * 100) + '% of the time!')
    return 

def extended_2to1(df):
    '''
    Helper Function for fig5 and fig6 to transform the extended column data.
    '''
    df = df[['packet_times', 'packet_sizes', 'packet_dirs']]
    df = df.apply(lambda x: x.str.split(';').explode())
    df = df.loc[df['packet_dirs'] == '2'].reset_index() 
    df = df.dropna(subset=['packet_sizes'])
    df['packet_sizes'] = df['packet_sizes'].astype(int)
    return df


def fig6():
    '''
    Creates figure 5 in EDA notebook displaying the time domain data for twitch live and twitch VOD.
    '''
    #working with dataset #1
    df1 = extended_2to1(twitch_live)
    df1 = df1[['packet_times', 'packet_sizes']].set_index('packet_times')
    df1.index = pd.to_datetime(df1.index,unit='ms')
    df1 = df1.resample('200ms').sum()
    s1 = df1['packet_sizes']/1e6
    
    #working with dataset #2
    df2 = extended_2to1(twitch_vod)
    df2 = df2[['packet_times', 'packet_sizes']].set_index('packet_times')
    df2.index = pd.to_datetime(df2.index,unit='ms')
    df2 = df2.resample('200ms').sum()
    s2 = df2['packet_sizes']/1e6
    
    
    zero_thres = 0.01
    pct_zeros1 = 100*np.sum(s1<zero_thres)/len(s1)
    pct_zeros2 = 100*np.sum(s2<zero_thres)/len(s2)
    
    
    print(f'feature'.ljust(20),'live'.ljust(10),'streaming'.ljust(10))
    print(f'pct_zeros'.ljust(20),f'{pct_zeros1:.1f}'.ljust(10),f'{pct_zeros2:.1f}'.ljust(10))
    

    max_value_time = np.max([np.max(s1),np.max(s2)])
    
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Twitch VOD", "Twitch Live"))

    fig.add_trace(go.Scatter(x=twitch_vod['Time'], 
                             y=s2,
                             name='2->1Packet_Sizes', 
                             line=dict(color='rgb(68,0,84)', width = 1)),
                               row=1, col=1)
    
    fig.add_trace(go.Scatter(x=twitch_live['Time'], 
                             y=s1,
                             name='2->1Packet_Sizes', 
                             line=dict(color='rgb(50,104,142)', width = 1)),
                               row=1, col=2)
  
    fig.add_shape(go.layout.Shape(type="line", yref="paper", xref="x", 
                                        x0=1, y0=0.01, x1=300,
                                        y1=0.01, line=dict(color='red', width=1)), row=1, col=1)
    fig.add_shape(go.layout.Shape(type="line", yref="paper", xref="x", 
                                        x0=1, y0=0.01, x1=300,
                                        y1=0.01, line=dict(color='red', width=1)),row=1, col=2)
    fig.update_layout(
        title='Time series, data binned in 200 ms intervals',
        xaxis_title="Time (Seconds)",
        yaxis_title="Packet_Sizes",
        font=dict(family="Courier New, monospace",size=16,color="Black"))
    fig.update_yaxes(range=[0, max_value_time*1.25])
    
    
    return fig
    
def fig7():
    '''
    Creates figure 6 in EDA notebook displaying the frequency domain data for twitch live and twitch VOD.
    '''
    #working with dataset #1
    df1 = extended_2to1(twitch_live)
    df1 = df1[['packet_times', 'packet_sizes']].set_index('packet_times')
    df1.index = pd.to_datetime(df1.index,unit='ms')
    df1 = df1.resample('200ms').sum()
    
    s1 = df1['packet_sizes']/1e6    
    fs = 5
    num_windows = 3
    f1, Pxx_den1 = signal.welch(s1, fs, nperseg=len(s1)/num_windows)
    peaks1, properties1 = signal.find_peaks(np.sqrt(Pxx_den1), prominence=.01)
    max_prominence_feature1 = properties1['prominences'].max()
    
    
    
    #working with dataset #2
    df2 = extended_2to1(twitch_vod)
    df2 = df2[['packet_times', 'packet_sizes']].set_index('packet_times')
    df2.index = pd.to_datetime(df2.index,unit='ms')
    df2 = df2.resample('200ms').sum()
    #df2 = df2.resample('2s').sum()
    s2 = df2['packet_sizes']/1e6

    
    fs = 5
    num_windows = 3
    f2, Pxx_den2 = signal.welch(s2, fs, nperseg=len(s2)/num_windows)
    peaks2, properties2 = signal.find_peaks(np.sqrt(Pxx_den2), prominence=.01)
    max_prominence_feature2 = properties2['prominences'].max()
    
    
    # Some interesting features
    
    max_prom_norm1 = max_prominence_feature1/np.mean(np.sqrt(Pxx_den1))
    max_prom_norm2 = max_prominence_feature2/np.mean(np.sqrt(Pxx_den2))
    
    
    peak_0p1Hz_norm1 = Pxx_den1[np.where(abs(f1-0.1) == min(abs(f1-0.1)))][0]/np.mean(Pxx_den1)
    peak_0p1Hz_norm2 = Pxx_den2[np.where(abs(f2-0.1) == min(abs(f2-0.1)))][0]/np.mean(Pxx_den2)
    peak_0p2Hz_norm1 = Pxx_den1[np.where(abs(f1-0.2) == min(abs(f1-0.2)))][0]/np.mean(Pxx_den1)
    peak_0p2Hz_norm2 = Pxx_den2[np.where(abs(f2-0.2) == min(abs(f2-0.2)))][0]/np.mean(Pxx_den2)


    
    print(f'feature'.ljust(20),'live'.ljust(10),'streaming'.ljust(10))
    print(f'max_prom/mean'.ljust(20),f'{max_prom_norm1:.1f}'.ljust(10),f'{max_prom_norm2:.1f}'.ljust(10))
    print(f'0.1Hz/mean'.ljust(20),f'{peak_0p1Hz_norm1:.1f}'.ljust(10),f'{peak_0p1Hz_norm2:.1f}'.ljust(10))
    print(f'0.2Hz/mean'.ljust(20),f'{peak_0p2Hz_norm1:.1f}'.ljust(10),f'{peak_0p2Hz_norm2:.1f}'.ljust(10))

 

    
    max_value = np.max([np.max(np.sqrt(Pxx_den1)),np.max(np.sqrt(Pxx_den2))])
    
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Twitch VOD", "Twitch Live"))

    fig.add_trace(go.Scatter(x=f2, 
                             y=np.sqrt(Pxx_den2),
                             name='2->1Packet_Sizes', 
                             line=dict(color='rgb(68,0,84)', width = 1)),
                               row=1, col=1)
    
    fig.add_trace(go.Scatter(x=f1, 
                             y=np.sqrt(Pxx_den1),
                             name='2->1Packet_Sizes', 
                             line=dict(color='rgb(50,104,142)', width = 1)),
                               row=1, col=2)
    
    
    fig.add_trace(go.Scatter(
        x=f2[peaks2],
        y=np.sqrt(Pxx_den2)[peaks2],
        mode='markers',
        marker=dict(
            size=7,
            color='red',
            symbol='cross'
        ),
        name='Detected Peaks Streaming'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=f1[peaks1],
        y=np.sqrt(Pxx_den1)[peaks1],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='x'
        ),
        name='Detected Peaks Live'
    ), row=1, col=2)
    fig.update_layout(
        title='Frequency transform (Hz), data binned in 200 ms intervals',
        xaxis_title="Frequency [Hz]",
        yaxis_title="SQRT Power Spectral Density",
        font=dict(family="Courier New, monospace",size=16,color="Black"))
    fig.update_yaxes(range=[0, max_value*1.25])
    
    return fig
