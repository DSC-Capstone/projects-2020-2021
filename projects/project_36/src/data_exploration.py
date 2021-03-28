"""
Jon Zhang, Keshan Chen, Vince Wong
data_exploration.py
"""
import sys
import json
import os
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
import seaborn as sns
from functools import reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

class data_exploration():
    """
    class data_exploration contains all the relevant methods for data exploration.
    """

    def parse_cpu_data(fname):
        """
        Description: reads a csv file and converts into dataframe
        Parameters: fname -> .csv
        Returns: DataFrame
        """
        return pd.read_csv(fname, usecols=['guid','load_ts', 'batch_id',
                                           'name','instance','nrs', 'mean',
                                           'histogram_min', 'histogram_max','metric_max_val'],
                                  nrows=1000000,
                                  sep='\t')

    def parse_sys_data(fname):
        """
        Description: reads a csv file and converts into dataframe
        Parameters: fname -> .csv
        Returns: DataFrame
        """
        return pd.read_csv(fname, sep=chr(1))

    def parse_app_data(fname1, fname2):
        """
        Description: reads a csv file and converts into dataframe
        Parameters: fname -> .csv
        Returns: DataFrame
        """
        apps = pd.read_csv(fname1, error_bad_lines=False, sep=chr(1))
        app_class = pd.read_csv(fname2, error_bad_lines=False, sep=chr(35))
        combined = apps.join(app_class, lsuffix='frgnd_proc_name', rsuffix='exe_name', how='left')
        return combined

    def optimize_dataframe(data):
        """
        Description: takes in the cpu dataframe and optimizes it through converting columns
        Parameters: df -> DataFrame
        Returns: DataFrame
        """
        df_int = df.select_dtypes(include=['int'])
        converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
        compare_ints = pd.concat([df_int.dtypes, converted_int.dtypes], axis=1)

        df_float = df.select_dtypes(include=['float'])
        converted_float = df_float.apply(pd.to_numeric, downcast='float')
        compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes], axis=1)

        optimized_df = df.copy()
        optimized_df[converted_int.columns] = converted_int
        optimized_df[converted_float.columns] = converted_float
        df_obj = df.select_dtypes(include=['object']).copy()

        converted_obj = pd.DataFrame()

        for col in df_obj.columns:
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if num_unique_values / num_total_values < 0.5:
                converted_obj.loc[:,col] = df_obj[col].astype('category')
            else:
                converted_obj.loc[:,col] = df_obj[col]

        optimized_df[converted_obj.columns] = converted_obj
        return optimized_df

    def get_stats(df, column, value):
        """
        Description: returns the dataframe when the column 'name' is that specified
        Parameters: df -> DataFrame
        Returns: DataFrame
        """
        return df.loc[df[column] == value]

    def get_guid(df, column):
        """
        Description: returns a list consisting of all the guid's
        Parameters: df -> DataFrame, column -> String
        Returns: list
        """
        return list(df[column].value_counts().index)

    def get_cpu_guid(series, list):
        """
        Description: returns a dataframe of only matching GUIDs
        Parameters: series -> SeriesObject, list -> List
        Returns: DataFrame
        """
        hwcpu_match = series.loc[series['guid'].isin(list)]
        hwcpu_match = hwcpu_match[['guid', 'load_ts', 'mean']]
        hwcpu_match['utilization_mean'] = hwcpu_match['mean']
        hwcpu_match = hwcpu_match.drop(columns='mean')
        return hwcpu_match

    def get_temp_guid(series, list):
        """
        Description: returns a dataframe of only matching GUIDs
        Parameters: series -> SeriesObject, list -> List
        Returns: DataFrame
        """
        hwtemp_match = series.loc[series['guid'].isin(list)]
        hwtemp_match = hwtemp_match[['guid', 'load_ts', 'mean']]
        hwtemp_match['temp_mean'] = hwtemp_match['mean']
        hwtemp_match = hwtemp_match.drop(columns='mean')
        return hwtemp_match

    def get_table(df, df2, df3):
        """
        Description: returns a dataframe consisting of left-joined dataframes matching on GUID
        Parameters: df -> DataFrame, df2 -> DataFrame, df3 -> DataFrame
        Returns: DataFrame
        """
        combined = df.join(df2, on=['guid'], how='left')
        combined = combined.join(df3, on=['guid'], how='left')
        #combined = combined.drop(columns=['model_normalized', "processornumber"])
        return combined

    def get_mean_durations(df, df2):
        """
        Description: returns a dataframe consisting of 3 columns matched on guid and mean durations
        Parameters: df -> DataFrame, df2 -> DataFrame
        Returns: DataFrame
        """
        mean_dur = df.pivot_table('event_duration_ms', ['guid', 'app_type'], aggfunc=np.mean).reset_index()
        combined_guid = list(df2['guid'].value_counts().index)
        dur_guid = list(mean_dur['guid'].value_counts().index)
        app_overlap = [x for x in combined_guid if x in dur_guid]
        mean_dur = mean_dur.loc[mean_dur['guid'].isin(app_overlap)]
        return mean_dur
