#creating explanatory and results oriented visualizations
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter


sys.path.insert(0, 'src/data')

from make_dataset import *

def plot_albums(title, outdir=None, *album_tups):
    '''
    Plots multiple albums as a single overlayed line plot with
    normalized dates

    :param title: Title of overlaid plot
    :outdir: output filepath for plot
    '''
    assert len(album_tups) > 1, 'passed in only one album'
    assert len(album_tups[0]) == 2, 'need to pass in both album and legend name'
    assert all(['normalized_dates' in album.columns for album, leg in album_tups])
    
    legend = []
    album, leg = album_tups[0]
    legend.append(leg)
    
    fig_dims = (6, 7)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.lineplot(x='normalized_dates', y=0, data=album.groupby('normalized_dates').size().reset_index(), ax=ax)
    for album_tup in album_tups[1:]:
        album, leg = album_tup
        legend.append(leg)
        sns.lineplot(x='normalized_dates', y=0, data=album.groupby('normalized_dates').size().reset_index(), ax=ax)
        
    # release date line
    max_counts = album.normalized_dates.value_counts().max()
    ax.axvline(0, color='purple', alpha=0.5)
    
    # legend
    ax.legend(legend)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Dates With Respect to Album Release', fontsize=13)
    if outdir:
        ax.figure.savefig(os.path.join(outdir, title + '.png'))

##### For Twitter #####

def generate_tweets_twitter_plot(tweets_fp, tweets_release_dates, tweets_legend, outdir):
    '''
    Generate twitter overlaid plot

    :param tweets_fp: file path to directory with data
    :param tweets_release_dates: album release dates for each plot
    :param tweets_legend: legend names for overlaid plot
    :param outdir: output filepath for plot
    '''
    tweet_csvs = os.listdir(tweets_fp)
    tweet_csvs.sort()
    dfs = []

    # normalize dates
    for csv, date in zip(tweet_csvs, tweets_release_dates):
        df = pd.read_csv(os.path.join(tweets_fp, csv))
        dfs.append(normalize_dates(df, date))
        
    # plot overlaid line chart
    tweet_tup = tuple(zip(dfs, tweets_legend))
    plot_albums('Tweet Plots', outdir, *tweet_tup)
    
def percent_col(users, col, a, perc=True):
    '''
    Finds the B% of engagament for a% of users
    :param users: series of aggregate engagement for users
    :param col: engagement column
    :param a: percent of users
    :param perc: if a is a percentage
    '''
    users = users[col].sort_values(ascending=False)
    if perc:
        a = int(len(users) * a)
    prop = users.iloc[:a].sum() / users.sum()
    return round(prop * 100, 2)

def perc_plot(df, suptitle, dfs=None, outdir=None, legend=None):
    '''
    Plots A% of users account for B% engagement plots
    :param df: first (or only) df to plot
    :param suptitle: title of all three plots
    :param dfs: other dfs to be overlaid
    '''
    # if overlaid plot
    if dfs:
        colors = [sns.color_palette()[0]] * (len(dfs) + 1)
        ylim = (35, 100)
    else:
        colors = ['#e02460', '#19cf86', '#1DA1F2']
        ylim = (55, 100)
    
    # prepare lists and plot fig/ax
    fig_dims = (20, 7)
    percs = np.arange(0, 0.051, 0.001)
    perc_cols = ['likes_count', 'retweets_count', 'replies_count']
    perc_titles = ['Likes', 'Retweets', 'Replies']
    fig, ax = plt.subplots(1, 3, figsize=fig_dims)
    fig.suptitle(suptitle, fontsize=18)
    
    for i in range(3):
        # plot curve
        curr = [percent_col(df, perc_cols[i], p) for p in percs]
        sns.lineplot(x=percs*100, y=curr, ax=ax[i], color=colors[i])
        
        # overlay other df curves
        if dfs:
            for j in range(len(dfs)):
                curr_df = [percent_col(dfs[j], perc_cols[i], p) for p in percs]
                sns.lineplot(x=percs*100, y=curr_df, ax=ax[i], color=sns.color_palette()[j + 1])
            if legend:
                ax[i].legend(legend, loc=4)

        # labels
        ax[i].set_title('A% of Users Account for B% of ' + perc_titles[i], fontsize=15)
        ax[i].set_xlabel('A', fontsize=13)
        ax[i].set_ylabel('B', fontsize=13)
        ax[i].xaxis.set_major_formatter(ticker.PercentFormatter())
        ax[i].yaxis.set_major_formatter(ticker.PercentFormatter())
        ax[i].set_ylim(ylim)

    if outdir:
        fig.savefig(os.path.join(outdir, 'percent_plots.png'))
    
def generate_perc_twitter_plots(tweets_fp, outdir):
    '''
    Plots the percent plots
    :param tweets_fp: file path to directory with data
    :param outdir: file path for plot
    '''
    tweet_csvs = os.listdir(tweets_fp)
    tweet_csvs.sort()
    dfs = []

    # read in data
    for csv in tweet_csvs:
        dfs.append(pd.read_csv(os.path.join(tweets_fp, csv)))

    # plot
    perc_plot(dfs[0], 'Percent Plots', dfs=dfs[1:], outdir=outdir)
    
##### For Wikipedia #####

def generate_wiki_plot(wiki_fp, wiki_release_dates, wiki_legend, outdir):
    '''
    Generate wiki overlaid plot

    :param wiki_fp: file path to directory with data
    :param wiki_release_dates: album release dates for each plot
    :param wiki_legend: legend names for overlaid plot
    :param outdir: output filepath for plot
    '''
    wiki_ld = os.listdir(wiki_fp)
    wiki_ld.sort()
    dfs = []

    # normalize dates
    for dump, date in zip(wiki_ld, wiki_release_dates):
        fp = os.path.join(wiki_fp, dump)
        title, df = read_lightdump(fp)
        dfs.append(normalize_dates(df, date))
        
    # plot overlaid line chart
    wiki_tup = tuple(zip(dfs, wiki_legend))
    plot_albums('Wiki Plots', outdir, *wiki_tup)
    
def visualize_revisions(data, main_titles, outdir):
    '''
    Visualize Wikipedia revision data
    
    :param wiki_fp: file path to directory with data
    :param main_titles: list of artist page titles
    :param outdir: output filepath for plot
    '''
    try:
        sns.set_theme()
    except:
        sns.set
    fig_dims = (12, 6)
    fig, ax = plt.subplots(1, 1, figsize = fig_dims)

    for (title, df) in data:

        if title in main_titles:

            df['month'] = pd.to_datetime(df.date).apply(lambda x: str(x.year) + '-' + str(x.month))
            df.month = pd.to_datetime(df.month)

            sns.lineplot(data=df.groupby('month').count().revert, label=title)
            ax.set_title("Wikipedia Page Revisions")
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Number of Revisions', fontsize=12)
            ax.yaxis.set_major_formatter(ticker.EngFormatter())    
            ax.legend()

    fig.savefig(os.path.join(outdir,'revisions.png'))
    
def visualize_revision_length(data, main_titles, outdir):
    '''
    Visualize Wikipedia revision length data
    
    :param wiki_fp: file path to directory with data
    :param main_titles: list of artist page titles
    :param outdir: output filepath for plot
    '''
    try:
        sns.set_theme()
    except:
        sns.set
    fig_dims = (12, 6)
    fig, ax = plt.subplots(1, 1, figsize = fig_dims)

    for (title, df) in data:
        if title in main_titles:

            df.date = pd.to_datetime(df.date)

            sns.lineplot(data=df.groupby('date').size(), label=title)
            ax.set_title("Wikipedia Page Revision Length")
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Revision Length', fontsize=12)
            ax.yaxis.set_major_formatter(ticker.EngFormatter())    
            ax.legend()

    fig.savefig(os.path.join(outdir, 'revision_length.png'))


def visualize_pageviews(views_fp, outdir):
    '''
    Visualize Wikipedia page view data
    
    :param fp: file path to directory with data
    :param outdir: output filepath for plot
    '''
    trend_csvs = os.listdir(views_fp)
    try:
        sns.set_theme()
    except:
        sns.set
    fig_dims = (12, 6)
    fig, ax = plt.subplots(1, 1, figsize = fig_dims)

    for csv in trend_csvs:
        try:
            df = pd.read_csv(os.path.join(views_fp, csv), engine='python')
        except:
            print('Cannot read file: ' + os.path.join(views_fp, csv))
            continue
            
        df['date'] = df.timestamp.apply(lambda x: x[:7])
        df['date'] = pd.to_datetime(df.date)
        
        title = df.iloc[0].article
        
        sns.lineplot(data=df.groupby('date').sum().views, label=title)
        ax.set_title("Wikipedia PageViews per Month")
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Number of Views', fontsize=12)
        ax.yaxis.set_major_formatter(ticker.EngFormatter())    
        ax.legend()

    fig.savefig(os.path.join(outdir, 'pageviews.png'))

##### For Google Trends #####

def visualize_google_trends(trends_fp, outdir):
    '''
    Visualize Google Trends data with search terms between
    given dates
    
    :param trends_fp: file path to directory with data
    :param outdir: output filepath for plot
    '''
    trend_csvs = os.listdir(trends_fp)
    
    for csv in trend_csvs:
        df = pd.read_csv(os.path.join(trends_fp, csv))
        
        start = str(df['date'].min())[:10]
        end = str(df['date'].max())[10:]
        
        title_text = 'Google Search Trends: '+ start + ' to ' +\
                end + ')'
        
        file_name = 'google_trend_plots_'+ start + '_' + end + '.png'
        
        #plotting
        g = sns.lineplot(data = df, x = 'date', y = 'Popularity',
                 hue = 'Artist', dashes = False)
        
        plt.title(title_text, size = 15)
        plt.xlabel("Date", size=13)
        plt.ylabel("Popularity", size=13)
        
        plt.savefig(os.path.join(outdir, file_name))
        

