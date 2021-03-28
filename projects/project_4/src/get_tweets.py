import tweepy
import yaml
import csv
import json
import sys

"""
This script will generate a jsonl file for a given user's most recent tweets on their timeline

run this script by doing this: python get_tweets.py <screen_name> <twitter_credentials.yaml> <out_path>

for example: python get_tweets.py JoeBiden keys.yaml data/JoeBiden_tweets.jsonl
"""

# helper class
class TwitterHarvester(object):
    # https://gist.github.com/MihaiTabara/631ecb98f93046a9a454
    """Create a new TwitterHarvester instance"""
    def __init__(self, consumer_key, consumer_secret,
                 access_token, access_token_secret,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=False):

        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.secure = True
        self.auth.set_access_token(access_token, access_token_secret)
        self.__api = tweepy.API(self.auth,
                                wait_on_rate_limit=wait_on_rate_limit,
                                wait_on_rate_limit_notify=wait_on_rate_limit_notify)
    
    @property
    def api(self):
        return self.__api
    
def limit_handled(cursor):
    # https://github.com/tweepy/tweepy/blob/master/docs/code_snippet.rst
    while True:
        try:
            yield next(cursor)
        except StopIteration:
            break
        except tweepy.RateLimitError:
            print('Rate limit hit')
            time.sleep(15 * 60)
            
            
            
def gather_tweets(user_screen_name, consumer_key, consumer_secret, access_token, access_token_secret, out_path):
    a = TwitterHarvester(consumer_key, consumer_secret, access_token, access_token_secret)
    api = a.api
    
    out_file=out_path
    output_file = open(out_file, 'w', encoding='utf-8')

    for data in limit_handled(tweepy.Cursor(api.user_timeline, 
                                            screen_name=user_screen_name, 
                                            tweet_mode='extended', 
                                            exclude_replies=True, 
                                            include_rts=False).items()):
        
        json.dump(data._json, output_file) 
        output_file.write("\n")
    
if __name__ == "__main__":
    user = sys.argv[1]
    credentials = sys.argv[2]
    out_path = sys.argv[3]
    
    with open(credentials) as file:
        keys = yaml.load(file, Loader=yaml.FullLoader)
    
    consumer_key = keys['consumer_key']
    consumer_secret = keys['consumer_secret']
    access_token = keys['access_token']
    access_token_secret = keys['access_token_secret']
    
    gather_tweets(user, consumer_key, consumer_secret, access_token, access_token_secret, out_path)
