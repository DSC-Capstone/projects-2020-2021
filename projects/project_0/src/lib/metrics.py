import os 
import json

def convert_date(date):
    '''
    takes in a date string from a tweet 
    converts the date into Month/Year format for graphing purposes
    '''
    created_at = date.split()
    month_conversion = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    return str(month_conversion[created_at[1]]) + '.' + created_at[-1]

def make_months():
    '''
    creates a dictionary of all month/year combinations starting from 2008 to 2020
    2008 is the first year a tweet was recorded, 2020 was the last year 
    '''
    d = {}
    for year_num in range(2008, 2021):
        for month_num in range(1,13):
            d[str(month_num) + '.' + str(year_num)] = 0
    return d

def make_years():
    '''
    creates dictionary of all years from 2008 to 2020
    '''
    return {str(year):[] for year in range(2008, 2021)}

def count_likes_over_months(inpath, outpath, category):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {date: metric}}
    metric is the count of likes for the each month
    '''
    likes_over_time = {}

    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)

            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)

                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    date = convert_date(tweet['created_at'])
                    likes = tweet['favorite_count']

                    if name not in likes_over_time:
                        likes_over_time[name] = make_months()

                    likes_over_time[name][date] += likes
    
    filepath = outpath + '/' + category + '_count_likes_over_months.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_over_time, f)

def avg_likes_over_months(inpath, outpath, category, x_months):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {date: metric}}
    metric is the average number of likes over the past x months
    '''
    likes_over_time = {}

    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)

            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)

                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    date = convert_date(tweet['created_at'])
                    likes = tweet['favorite_count']

                    if name not in likes_over_time:
                        likes_over_time[name] = make_months()

                    likes_over_time[name][date] += likes
    
    avg_likes_over_time = {}
    
    for name in likes_over_time:
        if name not in avg_likes_over_time:
            avg_likes_over_time[name] = make_months()
        
        months = list(likes_over_time[name].keys())
        counts = list(likes_over_time[name].values())
        
        for i in range(len(months)): 
            window = counts[max(i+1-x_months, 0): i+1]
            avg_likes_over_time[name][months[i]] = sum(window) // len(window)
    
    filepath = outpath + '/' + category + '_avg_likes_over_months.json'
    with open(filepath, 'w+') as f:
        json.dump(avg_likes_over_time, f)

def max_likes_over_months(inpath, outpath, category, x_months):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {date: metric}}
    metric is the max number of likes over the past x months
    '''
    likes_over_time = {}

    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)

            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)

                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    date = convert_date(tweet['created_at'])
                    likes = tweet['favorite_count']

                    if name not in likes_over_time:
                        likes_over_time[name] = make_months()

                    likes_over_time[name][date] += likes
                    
    max_likes_over_time = {}
    
    for name in likes_over_time:
        if name not in max_likes_over_time:
            max_likes_over_time[name] = make_months()
        
        months = list(likes_over_time[name].keys())
        counts = list(likes_over_time[name].values())
        
        for i in range(len(months)): 
            window = counts[max(i+1-x_months, 0): i+1]
            max_likes_over_time[name][months[i]] = max(window)
    
    filepath = outpath + '/' + category + '_max_likes_over_months.json'
    with open(filepath, 'w+') as f:
        json.dump(max_likes_over_time, f)

def cumu_likes_over_months(inpath, outpath, category):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {date: metric}}
    metric is the cumulative number of likes over all months
    '''
    likes_over_time = {}

    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)

            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)

                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    date = convert_date(tweet['created_at'])
                    likes = tweet['favorite_count']

                    if name not in likes_over_time:
                        likes_over_time[name] = make_months()

                    likes_over_time[name][date] += likes
    
    cumu_likes_over_time = {}
    
    for name in likes_over_time:
        if name not in cumu_likes_over_time:
            cumu_likes_over_time[name] = make_months()
        
        total = 0
        
        for date in likes_over_time[name]:
            total += likes_over_time[name][date]
            cumu_likes_over_time[name][date] = total
    
    filepath = outpath + '/' + category + '_cumu_likes_over_months.json'
    with open(filepath, 'w+') as f:
        json.dump(cumu_likes_over_time, f)

def count_likes_over_tweets(inpath, outpath, category):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {tweet_num: metric}}
    metric is the number of likes for each tweet
    '''
    likes_per_tweet = {}
    
    for subdir, dis, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            curr_num_tweets = 0
            
            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    
                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    likes = tweet['favorite_count'] 
                    curr_num_tweets += 1
                    
                    if name not in likes_per_tweet:
                        likes_per_tweet[name] = {}
                        
                    likes_per_tweet[name][curr_num_tweets] = likes
    
    filepath = outpath + '/' + category + '_count_likes_over_tweets.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_per_tweet, f)

def avg_likes_over_tweets(inpath, outpath, category, x_tweets):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {tweet_num: metric}}
    metric is the average number of likes over the past x tweets
    '''
    likes_per_tweet = {}
    
    for subdir, dis, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            curr_num_tweets = 0
            user_likes = []
            
            if not file.endswith('.jsonl'):
                continue
            
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    
                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue

                    name = tweet['user']['name']
                    likes = tweet['favorite_count'] 
                    curr_num_tweets += 1
                    user_likes.append(likes)
                    
                    if name not in likes_per_tweet:
                        likes_per_tweet[name] = {}
                    
                    user_likes = user_likes[-x_tweets:]
                    likes_per_tweet[name][curr_num_tweets] = sum(user_likes) // len(user_likes)
    
    filepath = outpath + '/' + category + '_avg_likes_over_tweets.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_per_tweet, f)

def max_likes_over_tweets(inpath, outpath, category, x_tweets):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {tweet_num: metric}}
    metric is the max number of likes over the past x tweets
    '''
    likes_per_tweet = {}
    
    for subdir, dis, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            curr_num_tweets = 0
            user_likes = []
            
            if not file.endswith('.jsonl'):
                continue

            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    
                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue
                    
                    name = tweet['user']['name']
                    likes = tweet['favorite_count'] 
                    curr_num_tweets += 1
                    user_likes.append(likes)
                    
                    if name not in likes_per_tweet:
                        likes_per_tweet[name] = {}
                    
                    user_likes = user_likes[-x_tweets:]
                    likes_per_tweet[name][curr_num_tweets] = max(user_likes)
    
    filepath = outpath + '/' + category + '_max_likes_over_tweets.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_per_tweet, f)

def cumu_likes_over_tweets(inpath, outpath, category):
    '''
    takes in a datapath to the folder where politican's tweets are held in jsonl format
    returns a dictionary in the form of {name: {tweet_num: metric}}
    metric is the cumulative number of likes over all tweets
    '''
    likes_per_tweet = {}
    
    for subdir, dis, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            curr_num_tweets = 0
            total = 0
            
            if not file.endswith('.jsonl'):
                continue
                
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    
                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue
                        
                    name = tweet['user']['name']
                    likes = tweet['favorite_count'] 
                    curr_num_tweets += 1
                    total += likes
                    
                    if name not in likes_per_tweet:
                        likes_per_tweet[name] = {}
                    
                    likes_per_tweet[name][curr_num_tweets] = total
    
    filepath = outpath + '/' + category + '_cumu_likes_over_tweets.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_per_tweet, f)

def group_likes_over_years(inpath, outpath, category):
    '''
    creates {year: [likes]} to be used for permutation tests 
    '''
    likes_over_years = make_years()

    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            
            if not file.endswith('.jsonl'):
                continue 

            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)

                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue 
                    
                    year = tweet['created_at'].split()[-1]
                    likes = tweet['favorite_count']
                    
                    if likes > 0:
                        likes_over_years[year].append(likes)
                        
    filepath = outpath + '/' + category + '_group_likes_over_years.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_over_years, f)

def group_likes_over_months(inpath, outpath, category):
    '''
    creates {month: [likes]}  
    '''
    likes_over_months = make_months()
    for i in likes_over_months:
        likes_over_months[i] = []
    
    
    for subdir, dirs, files in os.walk(inpath):
        for file in files:
            filepath = os.path.join(subdir, file)
            
            if not file.endswith('.jsonl'):
                continue
                
            with open(filepath, encoding='utf-8') as f:
                for line in f.readlines():
                    tweet = json.loads(line)
                    
                    if tweet['full_text'][:2] == 'RT':
                        # dont count a tweet if it is a retweet 
                        continue
                        
                    date = convert_date(tweet['created_at'])
                    likes = tweet['favorite_count']

                    likes_over_months[date].append(likes)
                    
    filepath = outpath + '/' + category + '_group_likes_over_months.json'
    with open(filepath, 'w+') as f:
        json.dump(likes_over_months, f)