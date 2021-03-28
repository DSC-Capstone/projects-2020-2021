import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
from matplotlib.dates import DateFormatter, DayLocator
import logging

def get_top_hashtags(logger, data, outdir, n_hashtags, case_sensitive):
    logger.info('Start generating chart of top hashtags used')
    def get_hashtags(input_str):
        if type(input_str) != str:
            return []
        if not case_sensitive:
            input_str = input_str.lower()
        return [elem[9:-1] for elem in re.findall("'text': '\w+'", input_str)]
    data['list_hashtags'] = data['entities/hashtags'].apply(get_hashtags)
    hashtag_counts = pd.Series(data['list_hashtags'].sum()).value_counts()
    df = pd.DataFrame(hashtag_counts.sort_values(ascending=False)[:n_hashtags]).rename(columns={0: 'n_occ'})
    df.to_csv(os.path.join(outdir, 'top_hashtags.csv'))
    logger.info('Finish generating chart of top hashtags used')
    return df

# Keep in mind this visualization will skip drawing a line to 0 if there are no tweets and may error out if the hashtag has never been used.
def get_histogram_hashtags_by_day(logger, data, outdir, hashtags, case_sensitive):
    for hashtag in hashtags:
        df = data
        hashtag_dates = df[df['entities/hashtags'].str.contains(hashtag, case=case_sensitive).astype(bool)]['created_at']
        hashtag_dates = hashtag_dates.astype('datetime64')
        # New method of making the histograms
        cts = hashtag_dates.dt.floor('d').value_counts() \
                                .rename_axis('date') \
                                .reset_index(name='count') \
                                .sort_values('date') \
                                .reset_index(drop=True) \
                                .rename(columns={'count': 'c'})
        if len(cts) == 0:
            logger.info("#{} was never used".format(hashtag))
            continue
        else:
            logger.info("Creating histogram for #{}".format(hashtag))

        plt.rc('font', size=12)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cts.date, cts.c)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tweets')
        ax.set_ylim(bottom=0, top=cts.c.max())
        ax.set_title('Number of Tweets with #{}'.format(hashtag))
        ax.grid(True)
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d'))

        fig = ax.get_figure()
        fig.savefig(os.path.join(outdir, 'hashtag_{}_histogram.png'.format(hashtag)))
        fig.clf()

def get_histogram_posts_by_user(logger, data, outdir):
    logger.info('Creating histogram for # of posts by user')
    ax = data.groupby(['user/id'])['created_at'].count().plot(kind='hist', title='Posts by User')
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, 'tweets_by_user_histogram.png'))
    fig.clf()

def generate_stats(logger, data, outdir, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    cfg = {}
    for key, value in kwargs.items():
        cfg[key] = value

    get_top_hashtags(logger, data, outdir, cfg['n_hashtags'], cfg['case_sensitive'])
    get_histogram_posts_by_user(logger, data, outdir)
    get_histogram_hashtags_by_day(logger, data, outdir, cfg['hashtag'], cfg['case_sensitive'])
    return