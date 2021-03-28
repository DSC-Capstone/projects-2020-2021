import os
import pandas as pd
import json
import numpy as np


def clean_data(hydrated_twts_path_2016, hydrated_twts_path_2020, output_dir_2016, output_dir_2020, feats_of_interest, keywords_2016, keywords_2020, users):
    os.system("mkdir " + output_dir_2016)
    # Clean 2016 data
    all_days = {}
    for f in os.listdir(hydrated_twts_path_2016): # Loop through files day by day
        onefile_hashtags = get_feats(hydrated_twts_path_2016 + "/" + f, feats_of_interest)
        all_days = {**all_days, **onefile_hashtags} # Merge this day's dict with all_days
    clean = pd.read_json(json.dumps(all_days), orient='index', convert_axes=False)

    clean['hashtags'] = clean['hashtags'].replace(np.nan, '[]') # Replace nan with a string containing an empty list
    clean['tweet_id'] = clean.index
    clean.to_csv(output_dir_2016 + "/clean_tweets.csv")

    # Clean 2020 data
    all_days = {}
    os.system("mkdir " + output_dir_2020)
    for f in os.listdir(hydrated_twts_path_2020): # Loop through files day by day
        onefile_hashtags = get_feats(hydrated_twts_path_2020 + "/" + f, feats_of_interest)
        if onefile_hashtags == {}:
            continue
        clean = pd.read_json(json.dumps(onefile_hashtags), orient='index', convert_axes=False)
        clean['hashtags'] = clean['hashtags'].replace(np.nan, '[]') # Replace nan with a string containing an empty list
        clean['tweet_id'] = clean.index
        clean.to_csv(output_dir_2020 + "/" + f[:8] + "clean_tweets.csv")

    df = pd.DataFrame()
    for f in os.listdir(output_dir_2020):
        if f[-9:] != ".DS_Store":
            df = df.append(pd.read_csv(output_dir_2020 + "/" + f, index_col=0))
    
    filter_2020(df, output_dir_2020, keywords_2016, keywords_2020, users)

    twenty = pd.read_csv(output_dir_2020 + "/clean_tweets.csv", index_col=0)
    try:
        twenty = twenty.sample(n=500000)
    except:
        pass
    twenty.to_csv(output_dir_2020 + "/clean_tweets.csv")
    return


def filter_2020(df, output_dir_2020, keywords_2016, keywords_2020, users):
    def search_keywords(df, col, keywords):
        "Selects subset of df that contains at least one of the keywords in the specified col"
        pattern = '|'.join(keywords)
        df = df[df[col].str.contains(pattern)]
        return df

    def get_twts_for_users(df, user_list):
        return df[df['screen_name'].isin(user_list)]

    def filter_by_kwords_and_users(df, keywords, users):
        "Get tweets that contain at least one keyword from a list of keywords or are by one of the listed users"
        filtered_kwords = search_keywords(df, 'full_text', keywords)
        filtered_usrs = get_twts_for_users(df, users)
        return filtered_kwords.merge(filtered_usrs, how='outer')

    all_keywords = keywords_2016 + keywords_2020
    df.dropna(subset=['full_text'], inplace=True)
    df.rename(columns={'full_text': 'full_text_original'}, inplace=True) # save original case
    df['full_text'] = df['full_text_original'].apply(lambda x: x.lower()) # convert to lowercase for keyword search
    filtered_2020 = filter_by_kwords_and_users(df, all_keywords, users)
    filtered_2020.drop(columns='full_text', inplace=True) # drop lowercase column
    filtered_2020.rename(columns={'full_text_original': 'full_text'}, inplace=True)
    filtered_2020.to_csv(output_dir_2020 + "/clean_tweets.csv")
    return


def get_tweets(filename):
    print(filename)
    if filename[-9:] != ".DS_Store":
        with open(filename) as fh:
            for tweet in fh:
                yield json.loads(tweet)


def get_feats(filepath, feats_of_interest):
    all_twts_dict = {}
    
    #Calls generator function so that we can read in one tweet at a time
    single_tweets = get_tweets(filepath)
    while(True):
        this_twt_dict = {}
        #Once there are no more tweets in the file, next will return the default ''
        tweet = next(single_tweets, '')
        
        #If there are no more tweets in the file, break out of the while loop
        if tweet == '':
            break
        if tweet['lang'] == "en": # Get only tweets that are in English
            #Gets the hashtags from just the tweet itself NOT from the original tweet if this was a retweet or reply
            for feat in feats_of_interest:
                this_elem = tweet
                feat = feat.split("-") # Get nested keys
                
                if len(feat) > 1: # Must navigate to last nested key to get value
                    for i in range(len(feat)):
                        final_key = feat[i]
                        this_elem = this_elem[final_key]
                        if feat[i] == "hashtags": # Treat "hashtags" list differently
                            this_elem = [dictionary['text'].lower() for dictionary in this_elem if 'text' in dictionary] # convert hashtags to lowercase in process
                        if feat[i] == "user_mentions": # Treat "user_mentions" list differently
                            this_elem = [dictionary['screen_name'] for dictionary in this_elem if 'screen_name' in dictionary]
                        final_val = this_elem
                else: # Key is not nested
                    final_key = feat[0]
                    final_val = this_elem[final_key]

                this_twt_dict[final_key] = final_val # Add this key, value pair to dictionary for this tweet
            all_twts_dict[tweet['id_str']] = this_twt_dict # Add this tweet's dictionary to dict for all tweets on this day
    return all_twts_dict