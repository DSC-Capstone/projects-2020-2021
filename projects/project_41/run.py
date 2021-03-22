"""
Jon Zhang, Keshan Chen, Vince Wong
run.py
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
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

sys.path.insert(0, 'src')
from src.data_exploration import *
from src.model import *

def main(targets):
    """
    this method will run all the methods within class data_exploration.py
    """
    # Parse through the datasets and select only relevant columns
    cpu_df = data_exploration.parse_cpu_data("data/raw/hw_metric_histo.csv000")
    sys_df = data_exploration.parse_sys_data("data/raw/system_sysinfo_unique_normalized.csv000")
    apps_df = data_exploration.parse_app_data('data/raw/frgnd_backgrnd_apps.csv000',
                                              'data/raw/ucsd_apps_execlass.csv000')

    # Create a new reference to the optimized DataFrame
     optimized_df = data_exploration.optimize_dataframe(cpu_df)

    # grab the specific column "HW::CORE:C0:PERCENT" as a feature
    cpu = data_exploration.get_stats(optimized_df, "name", "HW::CORE:C0:PERCENT:")

    # grab the specific column "HW::CORE:TEMPERATURE:CENTIGRADE" as a feature
    temp = data_exploration.get_stats(optimized_df, "name", "HW::CORE:TEMPERATURE:CENTIGRADE:")

    # grab the GUIDs from each dataset and put them into lists
    sys_guid = data_exploration.get_guid(sys_df, 'guid')
    hw_guid = data_exploration.get_guid(cpu_df, 'guid')

    # checking for the GUID overlap in both datasets
    syshw_overlap = [guid for guid in sys_guid if guid in hw_guid]

    # objective is to create a dataframe of only matching GUIDs
    hwcpu_match = data_exploration.get_cpu_guid(cpu, syshw_overlap)

    # only grabbing the relevant columns to be matched on
    hwtemp_match = data_exploration.get_temp_guid(temp, syshw_overlap)

    # instantiating our dataframes to be joined
    hwtemp = pd.DataFrame(hwtemp_match.groupby('guid')['temp_mean'].mean())
    hwcpu = pd.DataFrame(hwcpu_match.groupby('guid')['utilization_mean'].mean())

    # joining our matched dataframes together, only using relevant columns
    table = data_exploration.get_table(sys_df, hwcpu, hwtemp)
    mean_dur = data_exploration.get_mean_durations(apps_df, table)
    combined = model.get_combined_table(mean_dur, table, "app_type")

    # Split into predictor variable and training data
    Y = model.targets(combined)
    X = model.features_df(combined)

    # train the model and plot the scores
    show = model.train_model(X, Y)
    model_scores = model.plot_graphical_model_scores(show)

    print(show)

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
