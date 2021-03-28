"""
Jon Zhang, Keshan Chen, Vince Wong
model.py
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

class model():
    """
    class model contains all methods pertaining to data modelling
    """
    def get_combined_table(df, df2, column):
        """
        Description: Merge combined DataFrame with every relevant feature column
        Parameters: df -> DataFrame, df2 -> DataFrame, column -> String
        Returns: DataFrame
        """
        itdur = df.loc[df[column]=='IT']
        avdur  = df.loc[df[column]=='Anti-Virus']
        commdur = df.loc[df[column]=='Communication']
        gamedur = df.loc[df[column]=='Game']
        iudur = df.loc[df[column]=='Installer/Updater']
        intdur = df.loc[df[column]=='Internet']
        meddur = df.loc[df[column]=='Media/Consumption']
        netdur = df.loc[df[column]=='Network Apps']
        offdur = df.loc[df[column]=='Office']
        sysdur = df.loc[df[column]=='System/Other']
        utdur = df.loc[df[column]=='Utility']
        meditdur = df.loc[df[column]=='Media/Edit']
        udur =  df.loc[df[column]=='*']
        edudur = df.loc[df[column]=='Education']
        appdur = df.loc[df[column]=='Metro/Universal Apps']
        oobedur = df.loc[df[column]=='OOBE']
        gldur = df.loc[df[column]=='Game Launcher']

        types = [itdur, avdur, commdur, gamedur, iudur, intdur, meddur, netdur,
                 offdur, sysdur, utdur, meditdur, udur, edudur, appdur, oobedur, gldur]

        combined = df2
        for x in types:

            string = x[column].iloc[0]
            x = x.drop(columns=[column])
            new_col = string + '_dur_ms'
            x[new_col] = x['event_duration_ms']
            x = x.drop(columns=['event_duration_ms'])
            combined = combined.merge(x, on=['guid'], how='left')

        lst = ['IT_dur_ms', 'Anti-Virus_dur_ms', 'Communication_dur_ms',
               'Game_dur_ms', 'Installer/Updater_dur_ms', 'Internet_dur_ms',
               'Media/Consumption_dur_ms', 'Network Apps_dur_ms', 'Office_dur_ms',
               'System/Other_dur_ms', 'Utility_dur_ms', 'Media/Edit_dur_ms',  '*_dur_ms',
               'Education_dur_ms','Metro/Universal Apps_dur_ms', 'OOBE_dur_ms']

        for i in lst:
            combined[i] = combined[i].fillna(0)

        combined = combined[['chassistype', 'modelvendor_normalized', 'ram',
                             'os','#ofcores', 'age_category',
                             'graphicsmanuf', 'gfxcard', 'graphicscardclass',
                             'cpuvendor', 'cpu_family',
                             'discretegraphics', 'vpro_enabled', 'utilization_mean',
                             'temp_mean','IT_dur_ms', 'Anti-Virus_dur_ms',
                             'Communication_dur_ms', 'Game_dur_ms', 'Installer/Updater_dur_ms',
                             'Internet_dur_ms', 'Media/Consumption_dur_ms', 'Network Apps_dur_ms',
                             'Office_dur_ms', 'System/Other_dur_ms', 'Utility_dur_ms',
                             'Media/Edit_dur_ms',  '*_dur_ms', 'Education_dur_ms',
                             'Metro/Universal Apps_dur_ms', 'OOBE_dur_ms', 'persona']]

        return combined


    def features_df(df):
        """
        Description: Method to separate all the features into a DataFrame
        Parameters: df -> DataFrame
        Returns: DataFrame
        """
        def macro_cats(x):
            """
            Description: Encodes strings to numerical values
            Parameters: x -> String
            Returns: String
            """
            if x=='Web User' or x=='Casual User' or x=='Communication' or x=='Win Store App User' or x=='Entertainment' or x=='File & Network Sharer':
                return 0
            elif x=='Gamer' or x=='Casual Gamer':
                return 1
            elif x=='Office/Productivity' or x=='Content Creator/IT':
                return 2
            elif x == 'Unknown':
                return 3
            else:
                return 4

        morecats = df.dropna(axis=1)
        morecats['persona'] = morecats['persona'].apply(macro_cats)
        morecats = morecats.drop(columns=['persona'])
        morecats = pd.get_dummies(morecats)
        return morecats

    def targets(df):
        """
        Description: Method just returns the persona column
        Parameters: df -> DataFrame
        Returns: DataFrame Column Object
        """
        Y = df['persona']
        return Y

    def train_model(_X_, _Y_):
        """
        Description: train_model essentially runs our classifiers and outputs accuracy scores
        Parameters: _X_ -> feature vectors, _Y_ -> predictor variable
        Returns: DataFrame
        """
        X_train, X_test, Y_train, Y_test = train_test_split(_X_, _Y_, test_size=0.2)
        names = ["Extra_Trees", "AdaBoost", "Gradient_Boosting"]

        classifiers = [
            ExtraTreesClassifier(n_estimators=10, min_samples_split=2, class_weight='balanced'),
            AdaBoostClassifier(n_estimators=100),
            GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)]

        scores = []
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, Y_train)
            score = clf.score(X_test, Y_test)
            scores.append(score)

        show = pd.DataFrame()
        show['name'] = names
        show['score'] = scores
        return show

    def plot_graphical_model_scores(df):
        """
        Description: returns a saved image of a barplot conveying our accuracy scores
        Parameters: df -> DataFrame
        Returns: Model_scores.png -> file saved in /data/out/
        """
        sns.set(style="whitegrid")
        ax = sns.barplot(y="name", x="score", data=df)
        return plt.savefig("data/out/Accuracy_Score.png")
