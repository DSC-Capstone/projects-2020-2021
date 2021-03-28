import pandas as pd
import numpy as np
import sys

def generate_tweetIDS(filename):
    all_tweets = pd.read_csv(filename, delimiter = '\t')
    all_tweets = all_tweets.sample(frac = 1/360)
    np.savetxt(r''+sys.argv[1]+'.txt', all_tweets.tweet_id, fmt='%d')

generate_tweetIDS(sys.argv[1]+'.tsv')
