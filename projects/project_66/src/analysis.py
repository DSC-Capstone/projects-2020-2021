import os
import pandas as pd
import numpy as np
#import plotly.express as px

def run_diff_exp_rscript(in_dir, synthetic_data, tool, tool_rmd, out_dir):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/performDiffExp.R " + in_dir + ' ' + synthetic_data + ' ' + tool + ' ' + tool_rmd + ' ' + out_dir)
    return

def run_comparison_rscript(tool_dir1, tool_dir2, tool_dir3, tool_dir4, tool_dir5, tool_dir6, tool_dir7, out_dir):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/compareTools.R " + tool_dir1 + ' ' + tool_dir2 + ' ' + tool_dir3 + ' ' + tool_dir4 + ' ' + tool_dir5 + ' ' + tool_dir6 + ' ' + tool_dir7 + ' ' + out_dir)
    return

def run_statistics_rscript(tool1, tool2, tool3, tool4, tool5, tool6, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/generateStatistics.R" + tool1 + ' ' + tool2 + ' ' + tool3 + ' ' + tool4 + ' ' + tool5 + ' ' + tool6 + ' ' + data1 + ' ' + data2 + ' ' + data3 + ' ' +
              data4 + ' ' + data5 + ' ' + data6 + ' ' + data7 + ' ' + data8 + ' ' + data9 + ' ' + data10 + ' ' + data11)
    return

def generate_graphs():
    df_all = pd.read_csv('~/RNASeqToolComparison/out/results/statistics.csv', sep = ' ')
    df_all0 = pd.read_csv('~/RNASeqToolComparison/out/results/statistics_0.csv', sep=' ')

    for data in list(df_all['Data'].unique()):
        df = df_all[df_all['Data']==data]
        fig = px.box(df, x='Samples per Condition', y='AUC', color='Tool',
                     title="AUC "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/auctest_"+data+".png")

    for data in list(df_all['Data'].unique()):
        df = df_all[df_all['Data']==data]
        fig = px.box(df, x='Samples per Condition', y='Accuracy', color='Tool',
                     title="Accuracy "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/accuracytest_"+data+".png")

    for data in list(df_all['Data'].unique()):
        df = df_all[df_all['Data']==data]
        fig = px.box(df, x='Samples per Condition', y='FDR', color='Tool',
                     title="FDR "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/fdrtest_"+data+".png")

    for data in list(df_all['Data'].unique()):
        df = df_all[df_all['Data']==data]
        fig = px.box(df, x='Samples per Condition', y='Sensitivty', color='Tool',
                     title="Sensitivty "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/sensitivtytest_"+data+".png")

    for data in list(df_all['Data'].unique()):
        df = df_all[df_all['Data']==data]
        fig = px.box(df, x='Samples per Condition', y='Specificity', color='Tool',
                     title="Specificity "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/specificitytest_"+data+".png")

    for data in list(df_all0['Data'].unique()):
        df = df_all0[df_all0['Data']==data]
        fig = px.box(df, x='Tool', y='False Positive Rate',
                     title="Type I Error Rate "+data)
        fig.update_xaxes(type='category')
        fig.write_image("~/RNASeqToolComparison/results/graphs/fpr_"+data+".png")
    return
