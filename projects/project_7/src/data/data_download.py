import os
import pandas as pd
import numpy as np
from psaw import PushshiftAPI
import datetime as dt

# Write all post info to disk
def write_data(infotype, infotype_path, before_year, before_day, before_month, after_year, after_day, after_month):
    
    api = PushshiftAPI()
    
    # Keep track of whether to create or append
    first = True
    
    for inf in infotype:
        print(inf)
        
        # For each subreddit in list - get 1000 post ID's from before date and save to disk
        start_epoch=dt.datetime(before_year, before_month, before_day).timestamp()
        end_epoch=dt.datetime(after_year, after_month, after_day).timestamp()
        
        while start_epoch > end_epoch:
            try:
                gen = list(api.search_comments(before=int(start_epoch),
                                            subreddit=inf,
                                            filter=['id', 'author'], limit = 10000))
                df = pd.DataFrame([thing.d_ for thing in gen])
                df['subreddit'] = inf
            except Exception:
                print('exception happened')
                break
            
            if len(df) == 0:
                break
            # Save to either science.csv, myth.csv, or politics.csv
            if first:
                df.to_csv(infotype_path)
                first = False
            # Append to first df
            else:
                df.to_csv(infotype_path, mode = 'a', header = False)

            # Start search from last date
            start_epoch = df['created_utc'].iloc[-1]
            
            # Sanity check
            print('Size: ' + str(os.path.getsize(infotype_path)) + ' Last date: ' + str(dt.datetime.fromtimestamp(start_epoch)))