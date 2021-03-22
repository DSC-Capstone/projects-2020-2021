import os
import warnings


#Shapely Imports
from shapely.geometry import Polygon, Point, LineString
import shapely


#Data Analysis 
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import csv
import sys
import os


#Plot
import matplotlib.pyplot as plt
import gc


#config
import configparser


student_marker = 'o'
student_colordict = {"healthy": '#00d420', 'exposed': '#f0e10c', 'infectious': '#a82e05'}
student_edgedict = {"healthy": '#00d420', 'exposed': '#cf9e19', 'infectious': '#a82e05'}
student_sizedict = {"healthy": 10, 'exposed': 12, 'infectious': 16}


teacher_marker = '^'
teacher_colordict = {"healthy": '#00d420', 'exposed': '#f0e10c', 'infectious': '#a82e05'}
teacher_edgedict = {"healthy": '#009416', 'exposed': '#cf9e19', 'infectious': '#a82e05'}
teacher_sizedict = {"healthy": 14, 'exposed': 16, 'infectious': 20}

plt.rcParams["figure.figsize"] = (15,15)



# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))


school_params = './config/schoolparams.ini'
parser = configparser.ConfigParser()
parser.read(school_params)
population = parser['SCHOOL_POPULATION']
total_population = int(population['total_population'])


def load_map_school1(file_path):
    '''
    This is specific to the current school layout at this point, should be modified later in the future
    assume the input school map file is sufficient
    '''
    school_geometry = gpd.read_file(file_path)

    # move second floor to map bottom
    p = Polygon([(900,-800), (900,-1100), (1650,-1100), ( 1650,-800)])
    sf = school_geometry.geometry.intersects(p)
    new_sf = school_geometry[sf].translate(yoff=-800)
    school_geometry.loc[sf,["geometry"]] = new_sf
    school_gdf = school_geometry.rename(columns={"object": "room_type"})


    # generate recess area
    grd_recess = Polygon([(750, -1000), (750, -710), (1615, -710), (1615, -1480), (1450, -1480), (1450, -1200), 
                                                               (900, -1200), (900, -1000)])
    pkg_recess = Polygon([(430, -1400), (430, -1150), (660, -1150), (660, -1400)])



    school_gdf = school_gdf[['Id', 'room_type', 'geometry']]
    school_gdf.loc[len(school_gdf)] = [90000, 'recess_yard', grd_recess]
    school_gdf.loc[len(school_gdf)] = [90001, 'recess_yard', pkg_recess]
    
    #rescale map to feet unit
    school_gdf.geometry = school_gdf.geometry.apply(lambda x: shapely.affinity.scale(x, xfact=0.37, yfact=0.37, origin=(0,0)))
    return school_gdf







def analyze_model_result(model_df=None, model_df_path = None, write_image = False, username='jleiucsd'):
    """
    generate statistic analysis of the model ouput on number of days vs number of covid positive patients
    if write_image True: saves the analysis plot to local directory of model output (oasis specified foloder)
    inputs:
        model_out_df: model output as a dataframe format
        model_out_path: model output as a file path to a csv file
            model_out_df, and model_out_path cannot be None at the same time!
        write_image: if True, the result will be saved on the local directory of model output
        username: username of oasis account
    """
    if (model_df is None) and (model_df_path is None):
        raise ValueError('All parameter inputs cannot be None!')
    
    
    
    output_path = "/oasis/scratch/comet/{}/temp_project/".format(username) +\
    "output"

    
    if (model_df is None):
        model_df = pd.read_csv(model_df_path)

        
    plt.grid(linestyle="-.")
    plt.plot(list(model_df.drop_duplicates().cov_positive/total_population), linewidth=5)
    plt.xlabel('Days', fontsize=20)
    plt.ylabel('COVID Positive Patients (%)', fontsize=20)
    
    if write_image:
        plt.savefig(fileout + 'output_stats.png')
    
    plt.show()



def output_to_csv(params_key, output_dfs, username='jleiucsd'):

    
    
    param_dict = eval(params_key)
    
    output_path = "/oasis/scratch/comet/{}/temp_project/".format(username) +\
    "output"
    for param, val in param_dict.items():
        if ('path' not in param) and ('N' not in param):
            output_path += '_{}_'.format(param) + str(val)

    try:
        os.mkdir(output_path)
    except OSError:
        pass

    
    
    model_df, agent_df = output_dfs
    agent_df.to_csv(output_path + '/agent_val.csv')
    model_df.to_csv(output_path + '/model_val.csv', index=False)
    
    
    
    
    

def write_output(params_key, output_dfs, map_path, username='jleiucsd'):

    
    
    param_dict = eval(params_key)
    
    output_path = "/oasis/scratch/comet/{}/temp_project/".format(username) +\
    "output"
    for param, val in param_dict.items():
        if ('path' not in param) and ('N' not in param):
            output_path += '_{}_'.format(param) + str(val)

    try:
        os.mkdir(output_path)
    except OSError:
        pass

    
    
    model_df, agent_df = output_dfs
    
    school_gdf = gpd.read_file(map_path)
    school_gdf = school_gdf.rename({'Id': 'AgentID'}, axis=1)
    agent_gdf = gpd.GeoDataFrame(agent_df, geometry=gpd.points_from_xy(agent_df.x, agent_df.y))
    
    df_size = len(agent_df.loc[(slice(None), agent_df.unique_id.values[0]), :])
    for i in range(df_size):
        step_gdf = agent_gdf.loc[(i, slice(None)), :]
        show(i, step_gdf, school_gdf, output_path)
    
    model_df.to_csv(output_path + '/model_val.csv', index=False)
    
    
    
def show(i, step_gdf, school_gdf, output_path):
    humans_gdf = step_gdf[step_gdf["viral_load"].isna()]
    humans_gdf = humans_gdf[humans_gdf["x"].notna()]
    viral_load = step_gdf[step_gdf["viral_load"].notna()]["viral_load"].droplevel("Step")
    
    # school map case
    school_map = school_gdf.merge(pd.DataFrame(viral_load).reset_index())
    basemap = school_map.plot(column = "viral_load", cmap="Reds", alpha = 0.5, vmin = 0, vmax = 5)
    school_map.boundary.plot(ax = basemap, color='k', linewidth=0.2)
        
        
    student_gdf = humans_gdf.loc[humans_gdf['unique_id'].str.startswith("S"), :]
    teacher_gdf = humans_gdf.loc[humans_gdf['unique_id'].str.startswith("T"), :]
    for status in humans_gdf['health_status'].unique():       
        # student case
        student_gdf[student_gdf['health_status'] ==  status].plot(
            ax = basemap, 
             marker= student_marker,
            markersize= student_sizedict[status], 
            edgecolor= student_edgedict[status], 
            color=student_colordict[status], 
            linewidth=1.0
        )



        # teacher case
        teacher_gdf[teacher_gdf['health_status'] ==  status].plot(
            ax = basemap, 
             marker= teacher_marker,
            markersize= teacher_sizedict[status], 
            edgecolor= teacher_edgedict[status], 
            color=teacher_colordict[status], 
            linewidth=1.0
        )

    
    
    day = i//90
    step_count = i%90
    hour = 9 + step_count*5//60 # assume plot start at 9am
    minute = step_count*5%60
    plt.title("Iteration: Day {}, ".format(day + 1) + "%d:%02d" % (hour, minute), fontsize=30)
    
    
    
    img_path = output_path + "/outputimages"
    try:
        os.mkdir(img_path)
    except OSError:
        pass
    fileout = img_path + "/" + str(day + 1) + '_' + ("%03d" % (step_count)) + ".png"

    plt.savefig(fileout)
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    # force clear catch 
    gc.collect()