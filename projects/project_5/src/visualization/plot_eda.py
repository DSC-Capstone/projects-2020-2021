import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re

def num_rows(myth_path, science_path, politics_path, outpath):
    myth = pd.read_csv(myth_path)
    science = pd.read_csv(science_path)
    politics = pd.read_csv(politics_path)

    categories = ["myth", "science", "politics"]
    count_vals = [len(myth), len(science), len(politics)]

    plt.barh(categories, count_vals, color=['#dda0dd', '#bc8f8f','#b0c4de'])
    plt.xlabel("Category")
    plt.ylabel("Number of posts")
    plt.title("Number of posts in each category")

    for index, value in enumerate(count_vals):
        plt.text(value, index, str(value))

    plt.savefig(outpath)
    plt.clf()

def category_rows(path, outpath, colors):
    df = pd.read_csv(path)

    pattern = "\/(.*?)_"
    cat = re.search(pattern, outpath).group(1)

    subcount = df.groupby("subreddit").count()["author"].sort_values(ascending=True)
   
    colors = ast.literal_eval(colors)
    plt.barh(subcount.index, subcount, color=colors)
    plt.xlabel("Number of posts")
    plt.ylabel("Subreddit")
    plt.title("Number of posts per subreddit in " +cat+ " category")

    plt.savefig(outpath)
    plt.clf()

def onetime_posters(path, outpath, colors):
    df = pd.read_csv(path)

    pattern = "\/(.*?)_"
    cat = re.search(pattern, outpath).group(1)

    # number of users who only posted once in each subreddit
    sub_unique = df.groupby(["subreddit","author"])["id"].count()
    sub_unique = sub_unique[sub_unique < 2].sum(level=[0])

    # divided by total number of users in that subreddit 
    # this gives us the percentage of posters who only posted once in each sub
    sub_unique = sub_unique/df.groupby("subreddit")["author"].nunique()

    sub_unique = sub_unique.sort_values(ascending=False)

    colors = ast.literal_eval(colors)
    plt.barh(sub_unique.index, sub_unique, color=colors)
    plt.xlabel("Proportion")
    plt.ylabel("Subreddit")
    plt.title("Proportion of one-time posters in " +cat+ " subreddits")

    plt.savefig(outpath)
    plt.clf()

def average_onetime_posters(science_path, politics_path, myth_path, outpath):
    myth = pd.read_csv(myth_path)
    science = pd.read_csv(science_path)
    politics = pd.read_csv(politics_path)

    politics_sub_unique = politics.groupby(["subreddit","author"])["id"].count()
    politics_sub_unique = politics_sub_unique[politics_sub_unique < 2].sum(level=[0])
    politics_sub_unique = politics_sub_unique/politics.groupby("subreddit")["author"].nunique()
    politics_unique = np.mean(politics_sub_unique)

    science_sub_unique = science.groupby(["subreddit","author"])["id"].count()
    science_sub_unique = science_sub_unique[science_sub_unique < 2].sum(level=[0])
    science_sub_unique = science_sub_unique/science.groupby("subreddit")["author"].nunique()
    science_unique = np.mean(science_sub_unique)

    myth_sub_unique = myth.groupby(["subreddit","author"])["id"].count()
    myth_sub_unique = myth_sub_unique[myth_sub_unique < 2].sum(level=[0])
    myth_sub_unique = myth_sub_unique/myth.groupby("subreddit")["author"].nunique()
    myth_unique = np.mean(myth_sub_unique)

    categories = ["myth", "science", "politics"]

    plt.barh(categories, [myth_unique, science_unique, politics_unique], color=['#dda0dd', '#bc8f8f','#b0c4de'])
    plt.ylabel("Category")
    plt.xlabel("Proportion")
    plt.title("Average proportion of one-time posters across subreddits for each category")

    for index, value in enumerate([myth_unique, science_unique, politics_unique]):
        plt.text(value, index, str("{:.3f}".format(value)))

    plt.savefig(outpath)
    plt.clf()

def average_posts(science_path, politics_path, myth_path, outpath):
    myth = pd.read_csv(myth_path)
    science = pd.read_csv(science_path)
    politics = pd.read_csv(politics_path)

    politics_avg_post = politics.groupby(["subreddit","author"])["id"].count()
    politics_avg_post = politics_avg_post.mean(level=[0]) # average number of posts per sub
    politics_avg_post = np.mean(politics_avg_post) # average number of average posts per sub 

    science_avg_post = science.groupby(["subreddit","author"])["id"].count()
    science_avg_post = science_avg_post.mean(level=[0]) # average number of posts per sub
    science_avg_post = np.mean(science_avg_post) # average number of average posts per sub 

    myth_avg_post = myth.groupby(["subreddit","author"])["id"].count()
    myth_avg_post = myth_avg_post.mean(level=[0]) # average number of posts per sub
    myth_avg_post = np.mean(myth_avg_post) # average number of average posts per sub 

    categories = ["myth", "science", "politics"]

    plt.barh(categories, [myth_avg_post, science_avg_post, politics_avg_post], color=['#dda0dd', '#bc8f8f','#b0c4de'])
    plt.ylabel("Category")
    plt.xlabel("Number of posts")
    plt.title("Average number of posts per user across subreddits for each category")

    for index, value in enumerate([myth_avg_post, science_avg_post, politics_avg_post]):
        plt.text(value, index, str("{:.3f}".format(value)))

    plt.savefig(outpath)
    plt.clf()