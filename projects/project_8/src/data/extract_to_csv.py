import pandas as pd
import numpy as np
import json
import os

def convert_to_json (filepath):
    tweets_json = []
    with open(filepath) as f:
        for jsonObj in f:
            tweetDict = json.loads(jsonObj)
            tweets_json.append(tweetDict)
    return tweets_json

def extract_to_csv(filepath, output_path, filename):
    json_file = convert_to_json(filepath)
    df = pd.DataFrame(columns=['tweet_id', 'text', 'location','retweeted','hashtag','follower_count','created_at', 'language'])

    counter = 0
    for i in json_file:
        text = i['full_text']
        tweet_id = i['id']
        location = i['user']['location']
        retweeted = i['retweeted']
        hashtag = i['entities']['hashtags']

        hashtag_text = []
        for j in hashtag:
            hashtag_text.append(j['text'])

        created_at=i['created_at']
        follower_count = i['user']['followers_count']
        language = i['lang']

        df.loc[counter] = [tweet_id, text, location,retweeted,hashtag_text,follower_count,created_at, language]
        counter+=1

    df.to_csv(output_path+'/'+filename[:-4]+'csv')
    print('saved to '+output_path+'/'+filename[:-4]+'csv')

def convert_all_json(inputpath, outputpath):
    for filename in os.listdir(inputpath):
        if filename.endswith(".json"):
            completeName = os.path.join(inputpath, filename)

            extract_to_csv(completeName, outputpath, filename)
