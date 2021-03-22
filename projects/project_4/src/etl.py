import pandas as pd
import requests
import os
import gzip
import shutil
import json
from flatten_dict import flatten
from twarc import Twarc
import logging
import tweepy
import csv
import pickle
from os import listdir
from os.path import isfile, join

# Credit to: https://gist.github.com/yanofsky/5436496 for the Major tweets
def get_data_major_tweets(logger, consumer_key: str, consumer_secret_key: str, access_token: str, access_token_secret: str, bearer_token: str, output_path: str, exclude_replies: bool, include_rts: bool, max_recent_tweets: float, tweet_ids=[]):
    '''
    Retrieve user tweets and write to csv.
    '''
    # Tweepy Authorization
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Specific tweets
    for tweet_id in tweet_ids:
        status = api.get_status(tweet_id)
        logger.info(type(status))
        fn = os.path.join(output_path, 'tweet_{}.csv'.format(tweet_id))
        tweet_info = {
            'id': status.id,
            'created_at': status.created_at, # TODO: take out?
            'user/id': status.user.id,
            'user/name': status.user.screen_name,
            'text': status.text,
            'entities/hashtags': status.entities['hashtags'] # TODO: take out?
        } 
        user_ids = {}
        retweets_list = api.retweets(tweet_id)
        for retweet in retweets_list:
            user_ids[str(retweet.user.id)] = None
        tweet_info['user_ids'] = user_ids
        f = open(fn, 'wb')
        pickle.dump(tweet_info, f)

        # Get retweeters
        retweets_list = api.retweets(tweet_id)
        for retweet in retweets_list:
            user_id = retweet.user.id
            logger.info('Collecting user {} tweets'.format(user_id))

            #initialize a list to hold all the tweepy Tweets
            alltweets = []  

            #make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = api.user_timeline(user_id, count=200, exclude_replies=exclude_replies, include_rts=include_rts)

            #save most recent tweets
            alltweets.extend(new_tweets)

            if len(alltweets) != 0:
                #save the id of the oldest tweet less one
                oldest = alltweets[-1].id - 1

                # keep grabbing tweets until there are no tweets left to grab
                while len(new_tweets) > 0 and len(alltweets) < max_recent_tweets:
                    #logger.info('getting tweets before {}'.format(oldest))
                    
                    #all subsiquent requests use the max_id param to prevent duplicates
                    new_tweets = api.user_timeline(user_id, count=200, exclude_replies=exclude_replies, include_rts=include_rts, max_id=oldest)
                    
                    #save most recent tweets
                    alltweets.extend(new_tweets)
                    
                    #update the id of the oldest tweet less one
                    oldest = alltweets[-1].id - 1

                    #logger.info('{} tweets downloaded so far'.format(len(all   tweets)))
                    

                alltweets = alltweets[:max_recent_tweets]
        
            #transform the tweepy tweets into a 2D array that will populate the csv 
            outtweets = [[tweet.id_str, tweet.created_at, tweet.text, tweet.entities['hashtags']] for tweet in alltweets]
            # TODO: take out all other categories

            if output_path:
                fn = os.path.join(output_path, 'user_{}_tweets.csv'.format(user_id))
                with open(fn, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "created_at", "text", "entities/hashtags"])
                    writer.writerows(outtweets)
    logger.info("collected major tweet data")

def get_data(logger, preprocessed_data_path: str, training_data_path: str, dims: list, labels: dict, user_data_path: str, exclude_replies: bool, include_rts: bool, max_recent_tweets: int, tweet_ids: list, consumer_key: str, consumer_secret_key: str, access_token: str, access_token_secret: str, bearer_token: str):
    '''
    Retrieves data and writes to directory
    '''

    # TRAINING DATA
    logger.info('getting training data')
    fns = [filename for filename in listdir(preprocessed_data_path) if filename.endswith(".csv") ]
    dir = preprocessed_data_path

    li = []
    for fn in fns:
        logger.info('reading in {}'.format(fn))
        df = pd.read_csv(os.path.join(dir, fn), index_col=None, usecols=['text'], header=0)
        for i in range(len(dims)):
            label = labels[fn.replace('.csv', '')][i]
            df[dims[i]] = pd.Series([label]*df.shape[0])
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    os.makedirs(training_data_path, exist_ok=True)
    df.to_csv(os.path.join(training_data_path, 'data.csv'))

    logger.info('finished getting data and wrote data to {}'.format(os.path.join(training_data_path, 'data.csv')))

    # MAJOR TWEETS DATA
    get_data_major_tweets(logger, consumer_key, consumer_secret_key, access_token, access_token_secret, bearer_token, user_data_path, exclude_replies, include_rts, max_recent_tweets, tweet_ids=tweet_ids)
    return
