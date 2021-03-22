import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def ratio_means(inpath, users):
    '''
    takes in list of csv files, returns tuple
    * first item in tuple is list of means of ratios for scientific group
    * second item in tuple is list of names of politicians
    '''
    means = []
    for user in users:
        filename = inpath + '/' + user + '.csv'
        df = pd.read_csv(filename, index_col=0)
        
        # remove ratio rows with 0 (no engagement for that tweet/retweet)
        df = df[df['ratio'] != 0]
        df_ratio_mean = df[df['ratio'] != float('inf')]['ratio'].mean()
        
        means.append(df_ratio_mean)
        
    return means


def scientific_ratios_graph(inpath, outpath, users):
    '''
    takes in tuple of lists
    * first list contains scientific ratios of politicians
    * second list contains politician names
    '''
    means = ratio_means(inpath, users)
    
    df = pd.DataFrame.from_dict({'ratios': means, 'names': users}).sort_values(['ratios'], axis=0, ascending=True)

    plt.barh(df['names'], df['ratios'], color="blue")
    plt.title('Scientific Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Politician')
    plt.savefig(outpath + '/scientific_ratios.png', bbox_inches='tight')
    

def misinfo_ratios_graph(inpath, outpath, users):
    '''
    takes in tuple of lists
    * first list contains misinfo ratios of politicians
    * second list contains politician names
    '''
    means = ratio_means(inpath, users)
    
    df = pd.DataFrame.from_dict({'ratios': means, 'names': users}).sort_values(['ratios'], axis=0, ascending=True)

    plt.barh(df['names'], df['ratios'], color="orange")
    plt.title('Misinformation Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Politician')
    plt.savefig(outpath + '/misinfo_ratios.png', bbox_inches='tight')