import os
import subprocess
import re
import json
import numpy as np
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats

def get_date(tweet):
    return datetime.datetime.strptime(tweet['created_at'], '%a %b %d %X %z %Y').date()

def get_discussion(tweet):
    return tweet['favorite_count']+ tweet['retweet_count']


if __name__ == '__main__':
    target = sys.argv[1]
    files = os.listdir(target)
    log_disc_lst = []
    
    stats_dic = {}
    
    for file in files:
        if 'jsonl' not in file:
            pass
        else:
            with open(os.path.join(target, file)) as f:
                tweets = f.readlines()
            tweets = [json.loads(tweet) for tweet in tweets]
            twt_dic = {}

            for tweet in tweets:
                date = get_date(tweet)
                if date in twt_dic:
                    twt_dic[date].append(tweet)
                else:
                    twt_dic[date] = [tweet]

            disc = {}
            for date in twt_dic:
                tweet_lst = twt_dic[date]
                discussion_lst = np.array([get_discussion(tweet) for tweet in tweet_lst])
                disc[date] = discussion_lst

            disc_by_day= {}
            for date in disc:
                disc_by_day[date] = disc[date].sum()

            log_disc = {}
            for date in disc_by_day:
                log_disc[date] = np.log(disc_by_day[date] + 1)
                
            disc_vals = np.array(list(log_disc.values()))
            
            mean_disc = disc_vals.mean()
            n_disc = len(disc_vals)
            std_disc = disc_vals.std()
            
            stats_dic[file] = [mean_disc, std_disc, n_disc]
            
            print(file + " Mean Discussion Level:" + str(mean_disc) + '\n')
            log_disc_lst.append(log_disc)
            plt.hist(log_disc.values())
            fig_title = file.replace('.jsonl', '')
            plt.title(fig_title+ 'Daily discusison levels')
            plt.savefig(os.path.join(target, fig_title))
            print("Histogram added to {target} folder".format(target=target))
            plt.clf()
            
    year1, year2 = stats_dic.values()
    mu_1, sigma_1, n_1 = year1
    mu_2, sigma_2, n_2 = year2
    
    
    z_diff = np.abs((mu_1 - mu_2) / np.sqrt(sigma_1**2 / n_1 + sigma_2**2/n_2))
    p_value = stats.norm.pdf(z_diff)
    print("Distributions are different with p-value " + str(p_value))