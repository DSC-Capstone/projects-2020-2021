import glob
import os
import re

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def distance(lat1, lon1, lat2, lon2) -> float:
    """Calculates distance in feet between two pairs 
    of coordinates using Vicenty's algorithm and 
    assuming Earth is spherical. 
    """

    #convert latitiude and longitude to spherical coords
    lat1, lon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2, lon2 = np.deg2rad(lat2), np.deg2rad(lon2)
    
    # average radius of earth in feet
    r = 20890565.9449 
    
    p1 = 0.5*np.pi-lat1
    p2 = 0.5*np.pi-lat2
    a = np.sin(p1)*np.sin(p2)*np.cos(lon1-lon2)+np.cos(p1)*np.cos(p2)
    
    return r * np.arccos(a)

def rmse(expected, actual) -> float:    
    """ Calculates root mean squared error between the
    calcualted distances of GPS outputted coordinates
    and distance of track for each batch.
    """
    total_preds = len(expected)
    sum_error = 0.0
    
    for i in range(total_preds):
        squared_error = (actual[i]-expected[i])**2
        sum_error += squared_error
        
    return sqrt(sum_error/float(total_preds))

def error(expected, actual) -> float:
    """ Calculates error between expected distance
    and actual distance as indicated by GPS outputs
    for a single run.
    """
    total_preds = 1
    squared_error = (actual[0]-expected[0])**2
    
    return sqrt(squared_error)

def calc_distances_batch(path) -> pd.DataFrame():
    """Compiles a dataframe with batch name, run name, start
    position coordinates, end positionn coordinates, length 
    of track, calculated distance, and rmse of the batch.
    """
    
    ground_truth = int(path.split("../data/gps_data/", 1)[1].replace("ft",""))
    files = glob.glob(path + "/*.csv")
    df_dist = pd.DataFrame(columns = ['batch', 'run','start_lat_lon','finish_lat_lon','expected_dist','actual_dist','rmse'])
    
    for file in files:
        df = pd.read_csv(file)
        start = df.iloc[0]
        finish = df.iloc[-1]
        calc_dist = distance(start.lat, start.lon, finish.lat, finish.lon)
        name = re.search('(?:[^/](?!\/))+(?=_cleaned.csv)', file)
        batch = re.search('(?:\/gps_data\/)(.*(?=\/))', file)
        df_dist = df_dist.append({'batch': batch.group(1),'run' : name.group(), 'start_lat_lon': (start.lat, start.lon), 
                                  'finish_lat_lon': (finish.lat, finish.lon), 'expected_dist' : ground_truth, 
                                  'actual_dist' : calc_dist}, 
                                 ignore_index = True)
    df_dist.rmse = rmse(df_dist.expected_dist, df_dist.actual_dist)
    
    return df_dist

def calc_distances_run(path) -> pd.DataFrame():
    """Compiles a dataframe with batch name, run name, start
    position coordinates, end positionn coordinates, length 
    of track, calculated distance, and rmse of each file.
    """
    
    path = "../data/gps_data"
    all_raw_folders = glob.glob(os.path.join(path, '*'))
    df_all_runs = pd.DataFrame(columns = ['batch', 'run','start_lat_lon','finish_lat_lon','expected_dist','actual_dist','error'])


    for folder in all_raw_folders:
        ground_truth = int(folder.split("../data/gps_data/", 1)[1].replace("ft",""))
        files = glob.glob(folder + "/*.csv")
        for file in files:
            df = pd.read_csv(file)
            start = df.iloc[0]
            finish = df.iloc[-1]
            calc_dist = distance(start.lat, start.lon, finish.lat, finish.lon)
            name = re.search('(?:[^/](?!\/))+(?=_cleaned.csv)', file)
            batch = re.search('(?:\/gps_data\/)(.*(?=\/))', file)
            row = pd.DataFrame({'batch': batch.group(1),'run' : name.group(), 'start_lat_lon': [(start.lat, start.lon)], 
                                      'finish_lat_lon': [(finish.lat, finish.lon)], 'expected_dist' : ground_truth, 
                                      'actual_dist' : calc_dist, 'error': "temp"}, index=[0])

            row.error = error(row.expected_dist, row.actual_dist)
            df_all_runs = df_all_runs.append(row)
        
    return df_all_runs

def make_rmse_table(path):
    df = pd.DataFrame(columns = ['ground_truth', 'rmse'])
    all_batches = glob.glob(os.path.join(path, '*'))
    
    for batch in all_batches:
        batch_df = calc_distances(batch)
        df = df.append({'ground_truth' : batch_df.batch.iloc[0], 'rmse' : batch_df.rmse.iloc[0]}, ignore_index=True)
    df = df.set_index('ground_truth')
    df = df.sort_index(ascending=True)
    return df
    

