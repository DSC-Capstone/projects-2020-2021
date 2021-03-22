import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.util import *

def import_data(input_paths, type_lst):
    """
    Load input data 
    """
    df_lst = []
    for p, t in zip(input_paths, type_lst):
        df = pd.read_csv(p).iloc[:, 1:]
        df['type'] = t
        df_lst.append(df)

    df = pd.concat(df_lst, ignore_index=True)

    df['api_call'] = cantor_pairing(np.vstack((df.api_id.values, df.package_id.values)))
    df.loc[:,'api_call'] = reset_id(df['api_call'])

    return df

def eda_df(df, agg_col, rename_col, groupby_col='app_id'):
    type_dict = df.groupby(groupby_col).agg({'type':np.unique}).to_dict()['type']
    result = (
        df
        .groupby(groupby_col)
        .agg({agg_col:lambda x: len(np.unique(x))})
        .reset_index()
    )
    result['type'] = result.app_id.map(type_dict)
    result.drop(groupby_col, axis=1, inplace=True)
    result.rename(columns={agg_col:rename_col}, inplace=True)
    return result

def cat_plot(df, x, y, outdir):
    plt.clf()
    sns.catplot(
        data=df,
        x=x,
        y=y
    )
    plt.savefig(os.path.join(outdir, f'{y}_categorical_scatter_plot.png'))

def hist_plot(df, x, y, outdir):
    plt.clf()
    df[x] = np.log(df[x])
    sns.histplot(
        data=df, 
        x=x, 
        hue=y
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{x}_histogram_plot.png'))

def generate(input_paths, type_lst, outdir, eda_dict):
    df = import_data(input_paths, type_lst)
    for k in eda_dict:
        e_df = eda_df(df, k, eda_dict[k], groupby_col='app_id')
        cat_plot(e_df, 'type', eda_dict[k], outdir)
        hist_plot(e_df, eda_dict[k], 'type', outdir)