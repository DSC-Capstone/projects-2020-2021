#to turn raw data into features for modeling
import pandas as pd
import wikipedia
import sys
from datetime import date

sys.path.insert(0, 'src/data')
from make_dataset import *


##### For Wikipedia #####

def summary_stats(data):
    '''
    Create a row of stats for a given article

    :param files: lightdump files
    '''
    summary = pd.DataFrame([], columns = ['Article', 
                                      'Revisions',
                                      'Editors',
                                      'Reverts',
                                      'Bots',
                                      'Pageviews',
                                      'Talk Revisions',
                                      'Talk Editors',
                                      'Talk Reverts'
                                     ])
    for (title, df) in data:
    
        if 'Talk:' in title:
            row.append(df.shape[0])                           # talk revisions

            humans = df[df.user.apply(lambda x: 'bot' not in str(x).lower())]
            row.append(len(humans.user.unique()))             # talk editors

            row.append(df.revert.sum())                       # talk reverts
            d = pd.Series(row, index = summary.columns)
            summary = summary.append(d, ignore_index = True)
        else:
            row = [title]                                     # article
            row.append(df.shape[0])                           # revisions

            humans = df[df.user.apply(lambda x: 'bot' not in str(x).lower())]
            row.append(len(humans.user.unique()))             # editors

            row.append(df.revert.sum())                       # reverts

            bots = df[df.user.apply(lambda x: 'bot' in str(x).lower())]
            row.append(len(bots.user.unique()))               # bots

            try:
                pageviews = query_per_article(title, 
                                              '20150701', 
                                              '20210227', 
                                              interval = 'daily')
                row.append(pageviews.views.sum())                 # pageviews
            except:
                row.append(10000)
            

    return summary

def get_months(data, main_titles):
    months = []
    for (title, df) in data:

        if title in main_titles:
            creation = df.date.min()

            today = date(2021, 2,28)
#             creation = date(int(d[:4]),int(d[5:7]),int(d[8:]))
            num_months = (today.year - creation.year) * 12 + (today.month - creation.month)
            months.append(num_months)
        
    return months

def agg_norm_stats(summary, months):
    '''
    Create aggregated stats

    :param summary: summary df
    '''
    agg = pd.DataFrame([], columns = ['Topic', 
                                          'Revisions',
                                          'Editors',
                                          'Reverts',
                                          'Bots',
                                          'Pageviews',
                                          'Talk Revisions',
                                          'Talk Editors',
                                          'Talk Reverts'
                                         ])
    
    for i in range(0,summary.shape[0], 4):
        temp = summary.iloc[i:i+4]
        topic = summary.iloc[i].Article
        temp = temp.append(temp.sum().rename('Total')).drop(['Article'], axis=1)
        temp = temp.iloc[4]

        row = [topic]
        row += temp.tolist()
        d = pd.Series(row, index = agg.columns)
        agg = agg.append(d, ignore_index = True)
    
    temp = agg.set_index('Topic')
    for idx, d in enumerate(months):
        row = temp.iloc[idx]/ d
        temp = temp.append(pd.Series(row))
        
    num_articles = summary.shape[0]/4
    agg_norm = temp[int(num_articles):]
    agg_norm = agg_norm.astype(float).round(2)
    agg_norm.columns = [x + ' per Month' for x in agg_norm.columns]
    return agg_norm
    
    
def wiki_summary_stats(data, main_titles, outdir):
    '''
    Generate wiki summary stats

    :param wiki_fp: input wiki fp
    :main_titles: titles of main artist pages
    :param outdir: output filepath for csv
    '''
    df = summary_stats(data)
    df = agg_norm_stats(df, get_months(data, main_titles))
    df.to_csv(os.path.join(outdir, 'wiki_summary_stats.csv'))
    
    
##### For Google Trends #####

def trends_summary_stats(trends_fp, outdir):
    '''
    Generate Google Trends summary stats
    
    :param trends_fp: input Google Trends fp
    :param outdir: output filepath for csv
    '''
    
    trend_csvs = os.listdir(trends_fp)
    
    for csv in trend_csvs:
        df = pd.read_csv(os.path.join(trends_fp, csv))
        
        summary = df.groupby('Artist').agg({'Popularity': ['mean', 'median',
                                                 'count', 'max', 'min', 'std',
                                                 'var', 'skew', pd.DataFrame.kurt]})
        
        start = str(df['date'].min())[:10]
        end = str(df['date'].max())[10:]
        
        file_name = 'google_trend_summary_stats_'+ start + '_' + end + '.csv'
        df.to_csv(os.path.join(outdir, file_name))
