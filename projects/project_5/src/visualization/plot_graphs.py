import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

def polarity_histogram(polarity_path, save_path):
    plt.clf()
    plt.figure(figsize=(15, 14))
    
    df = pd.read_csv(polarity_path)
    categories = ['science (%)', 'myth (%)', 'politics (%)']
    colors = ["lightsteelblue", "rosybrown", "plum"]
    i = 0 
    for c in categories:
        df[c].hist(color = colors[i])
        i += 1
        plt.ylim(0, 350_000)
        plt.xlabel('Percentage of Intersection')
        plt.ylabel('Frequency')
        plt.title('User Polarity of ' + c[:-4].capitalize() + ' Subreddits')
        
        plt.savefig(save_path + '/' + c + '.png', bbox_inches = 'tight')
        plt.clf()
    plt.hist([df[categories[0]], df[categories[1]], df[categories[2]]], label=['Science', 'Myth', 'Politics'], color = colors)
    plt.legend(loc='upper right')
    plt.xlabel('Percentage of Intersection')
    plt.ylabel('Frequency')
    plt.title('User Polarity of All Subreddits')
    plt.savefig(save_path + '/all_polarity.png', bbox_inches = 'tight')
    plt.clf()
def count_chart(count_dict_path, save_path):
    plt.clf()
    
    df = pd.read_csv(count_dict_path, index_col=0, header=0)
    plot = sns.heatmap(df, vmax=0.1).get_figure()
    plot.savefig(save_path, dpi=400)

def polarity_chart(polarity_dict_path, save_paths):
    plt.clf()
    
    df = pd.read_csv(polarity_dict_path, index_col = 0, header=0)
    
    sub_dfs = [df.copy(), df.copy(), df.copy()]
    for i in range(3):
        for column in sub_dfs[i]:
            sub_dfs[i][column] = df[column].map(lambda x: eval(x)[i])
    
    i = 0
    for s in sub_dfs:
        plot = sns.heatmap(s).get_figure()
        plot.savefig(save_paths[i], dpi=400)
        i += 1
        plt.clf()
    #science-politics
    total = sub_dfs[0] + sub_dfs[2]
    new_weight = sub_dfs[2] / total
    plot = sns.heatmap(new_weight, cmap="Reds")
    plot.set_title('Science-Politics Overlap')
    plot.collections[0].colorbar.set_label("Tendency to Political Misinformation")
    plt.savefig(save_paths[3], dpi=400)
    plt.clf()
    #science-myth
    total = sub_dfs[0] + sub_dfs[1]
    new_weight = sub_dfs[1] / total
    plot = sns.heatmap(new_weight, cmap="Reds")
    plot.set_title('Science-Myth Overlap')
    plot.collections[0].colorbar.set_label("Tendency to Myth Misinformation")
    plt.savefig(save_paths[4], dpi=400)