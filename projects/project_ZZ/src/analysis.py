
## External Module Imports
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
import output_image
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor

def generate_file_names(combinations_list, output_dir_prefix='outputs'):
    
    '''
    name: generate_file_names
    
    purpose: To generate the list of files to look for from the raw outputs.
        
    parameters: 
        combinations_list: dataframe of combinations of variable model parameters
        output_dir: Directory to read raw model outputs from.
        
    returns:
        The list of files from the input parameters to look for.
    '''
    
    files = []
    for combi in combinations_list:
        seat_dist, init_patient, attend_rate, inclass_lunch, mask_prob, iteration = combi
        files.append(f'{output_dir_prefix}/output_seat_dist_{seat_dist}_init_patient_{init_patient}_attend_rate_{attend_rate}_inclass_lunch_{inclass_lunch}_mask_prob_{mask_prob}_iteration_{iteration}')
    return files

def return_files_single_param(parameter_val_tuple, files):
    '''
    name: return_files_single_param
    
    purpose: To filter out a list of files for a specific set of parameters.
    
    parameters: 
        parameter_val_tuple: a tuple with in the form (param_name, value).
        files: List of model_output_files computed from variable params combinations.
        
    returns:
       The list of files containing the specified parameter combination.

    '''
        
    datafiles = []
    for file in files:
        param, val = parameter_val_tuple
        if f'{param}_{val}' in file:
            datafiles.append(file)
    return datafiles

def return_dataframe_params(files, data='model_val'):
    '''
    name: return_dataframe_params
    
    purpose: To return a combined pandas dataframe containing all results from parameter combinations.
    
    parameters: 
        files: List of model_output_files computed from variable params combinations.
        data: the file_name (csv) to analyze.
        
        
    returns:
       A combined pandas dataframe containing all results from parameter combinations
    '''
    
    dict_files = dict(zip(range(1, len(files)+1), files))
    dataframes = []
    for combo, file in dict_files.items():
        file = file + f'/{data}.csv'
        df = pd.read_csv(file)
        df['combo'] = combo
        dataframes.append(df)
    return pd.concat(dataframes)

def generate_facetgrid(data, combinations, x_col='day', y_col='cov_positive', sep_col='combo', filename='comboplot'):
    '''
    name: generate_facetgrid
    
    purpose: To return a grid containing lineplots containing results from specific parameter combinations.
    
    parameters: 
        data: the pandas dataframe output after running return_dataframe_params.
        combinations: a dataframe containing the individual list of combinations of parameters.
        x_col: the variable to plot on the x axis of the lineplot
        y_col: the variable to plot on the y axis of the lineplot
        sep_col: the separating column (in this case it's the combination number)
        filename: the output file name.
        
    returns:
       A combined pandas dataframe containing all results from parameter combinations
    '''
    
    
    
    data = data.copy()
    data[y_col] = data.apply(lambda row: row[y_col]/(500* combinations[combinations['combination_number'] == row['combo']]['attend_rate'].values[0]), axis=1)
    sns.set(font_scale=5)  
    g = sns.FacetGrid(data, col=sep_col, xlim=(0, 14), col_wrap=5, height=10)
    g.map(sns.pointplot, x_col, y_col, order=data[sep_col].unique(), color=".3", ci=None)

    for ax in g.axes.flat:
        labels = ax.get_xticklabels() 
        for i,l in enumerate(labels):
            if(i%5 != 0): labels[i] = '' #Code to set x limits in FacetGrid.
        ax.set_xticklabels(labels, rotation=30) 

    g.savefig(f'results/{filename}.png', dpi=400)
    return True

def generate_heatmaps(files, out_folder='heatmaps/'):
    
    '''
    name: generate_heatmaps
    
    purpose: To return a heatmap containing locations where agents become exposed.
    
    parameters:
        files: List of model_output_files computed from variable params combinations.
        
    returns:
       A combined pandas dataframe containing all results from parameter combinations
    '''
    
    
    for iteration in range(len(files)):
        path = files[iteration]
        path += '/agent_val.csv'
        dataframe_exposed_loc = pd.read_csv(path)
        dataframe_exposed_loc = dataframe_exposed_loc[dataframe_exposed_loc['health_status'] == 'exposed']

        if 'geometry' in dataframe_exposed_loc.columns:
            exposed_locations = dataframe_exposed_loc.groupby(["unique_id"])["geometry"].first()
            exposed_locations = gpd.GeoSeries(exposed_locations.apply(shapely.wkt.loads))
            exposed_locations = exposed_locations.to_frame()
            exposed_locations['x'] = exposed_locations.geometry.apply(lambda p: p.x)
            exposed_locations['y'] = exposed_locations.geometry.apply(lambda p: p.y)

        else:
            exposed_locations = dataframe_exposed_loc.groupby(["unique_id"])["x", "y"].first()
        
        school_geometry = gpd.read_file('layouts/schoollayout1.shp')
        p = shapely.geometry.Polygon([(900,-800), (900,-1100), (1650,-1100), ( 1650,-800)])
        sf = school_geometry.geometry.intersects(p)
        new_sf = school_geometry[sf].translate(yoff=-800)
        school_geometry.loc[sf,["geometry"]] = new_sf
        school_gdf = school_geometry.rename(columns={"object": "room_type"})
        
        ax = school_geometry.plot(color="white", edgecolor='black')
        sns.kdeplot(exposed_locations["x"], exposed_locations["y"], ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'Heatmap of Exposed Locations Combo {iteration+1}', fontdict={'size':20})
        plt.savefig(f'{out_folder}/heatmap_combo_{iteration+1}.png', dpi=400)
        
        
        
def return_significant_difference(group1, group2, groups):
    
    '''
    name: return_significant_difference
    
    purpose: To assess if distributions of COVID-19 transmission are statistically different
    using the Kolmogorov Smirnov test via a p-value cut-off of 0.05.
    
    parameters:
        group1: name of the combination number of group1.
        group2: name of the combination number of group2
        groups: pandas groupby object consisiting of all groups.
    returns:
       True if signifiacantly different.
       False if not significantly different.
    '''
    
    data1 = groups.get_group(group1).groupby('day')['cov_positive'].mean()
    data2 = groups.get_group(group2).groupby('day')['cov_positive'].mean()
    
    test = stats.ks_2samp(data1, data2)[1]
    
    if test < 0.05:
        return True
    else:
        return False

def get_different_column(row):
    '''
    name: get_different_column
    
    purpose: To find which variable parameter is responsible for the significant different.
    
    parameters:
        row: single row of the merged dataframe in the form of <factor_name>_x, <factor_name>_y notation.
    returns:
        a '-' delimited string containing the number of factors which vary.
    '''
    output = []
    if row['attend_rate_x'] != row['attend_rate_y']:
        output += ['attend_rate']
    if row['inclass_lunch_x'] != row['inclass_lunch_y']:
        output += ['inclass_lunch']
    if row['mask_prob_x'] != row['mask_prob_y']:
        output += ['mask_prob']
    return '-'.join(output)
