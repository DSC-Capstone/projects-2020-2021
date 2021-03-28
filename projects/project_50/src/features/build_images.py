# imports and dependencies
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import datetime
import numpy as np

from pyts.image import GramianAngularField

import os

'''
This function takes in dataframe transformed by Gramian Angular Field
and converts and saves image coordinates to actual images
@ return: None
'''
def convert_img(img_path, idx, curr_X_gadf, gramian_df):
    
    fig = plt.figure()
    ax = plt.subplot(111)

    # date_str = str(gramian_df.index[idx].date())
    date_str = str(gramian_df.index[idx])
    fname = f'{img_path}/{date_str}.png'

    ax.imshow(curr_X_gadf, cmap='rainbow', origin='lower')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(fname, bbox_inches='tight')

'''
This function wraps transformations of time series data by 
Gramian Angular Field with actual image conversion
@ return: None
'''
def gramian_img(img_path, gramian_df):
    
    # scale inputs to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    # did not flatten the images
    gadf = GramianAngularField(sample_range=(-1,1), method='difference')

    for idx in range(gramian_df.shape[0]):

        curr_row = gramian_df.iloc[idx].dropna().to_frame().T
        # Scale the data to be between -1 and 1
        curr_feat = scaler.fit_transform(curr_row)
        # gramian
        curr_X_gadf = gadf.fit_transform(curr_feat)
        # convert image
        convert_img(img_path, idx, curr_X_gadf[0], gramian_df)

