"""
"""

import sys
import os

from multilang_analyzer import MultilangAnalyzer
from pages import *
from results import *

from multiprocessing import Pool

from tqdm import tqdm

import glob

import pandas as pd

from random import sample
import re

class WikiPipeline:

    """
    USAGE

    #TODO
    """

    def __init__(self, lang):
        self.analyzer = MultilangAnalyzer(lang)
        self.lang = lang
        self.analyzer.setup()

    def get_target_articles(self, targets, skip_cats = []):
        """
        returns a list of all the articles in the target categories
        """
        #articles = set()
        articles = []
        with tqdm(targets, total=len(targets), desc='grabbing articles from cats...') as cat_iter:
            for cat in cat_iter:
                if cat in skip_cats: continue
                cat_iter.set_postfix({
                    'cat': cat
                })
                try:
                    articles += get_articles(cat, self.lang)
                except:
                    print(f'could not get articles for cat {cat}')
        #return list(articles)
        return articles

    def save_targets(self, target_articles, filename):
        """
        saves all the target articles to a json file
        """
        with open(filename, "w") as f:
            json.dump(target_articles, f)

    def get_all_page_edits(self, name):
        """
        acquires all versions of the given page, in a list of dicts

        name -> [{time, text}, {time, text} ...]    
        """
        return get_history(name, self.lang)

    def get_all_histories(self, targets):
        """
        for each article, gets the article history
        returns a dict with this schema:

        {
            # for each article
            "article_name": [
                # for each edit
                {
                    "time": datetime,
                    "text": string
                }, ...
            ], ...
        }
        """
        histories = {}
        with tqdm(targets, total=len(targets), desc='getting page edit hist') as target_iter:
            for target in target_iter:
                try:
                    target_iter.set_postfix({
                        'target': target
                    })
                    histories[target] = self.get_all_page_edits(target)
                except:
                    print(f'error finding edit history for {target}')
        return histories

    def save_all_histories(self, targets, dir_path):
        """
        for each article, gets the article history
        saves each history to a file in the target directory:
        """
        # find or make our target directory
        if not os.path.idsir(dir_path):
            os.mkdir(dir_path)

        with tqdm(targets, total=len(targets), desc='saving page edit hist') as target_iter:
            for target in target_iter:
                try:
                    # get page edit history
                    target_iter.set_postfix({
                        'target': target, 
                        'now': "querying"
                    })
                    history = self.get_all_page_edits(target)
                    # save edit history
                    target_iter.set_postfix({
                        'target': target, 
                        'now': "saving"
                    })
                    fname = re.sub('[:\s<>\|\?\*]', '_', target)
                    filename = f'{dir_path}/{fname}.json'
                    with open(filename, "w") as f:
                        json.dump(history, f)
                except:
                    print(f'error finding edit history for {target}')

    def process_target(self, target):
        """
        not used above in order to allow for tqdm postfix updates
        """
        try:
            history = self.get_all_page_edits(target)
            fname = re.sub('[:\s<>\|\?\*]', '_', target)
            filename = f'{dir_path}/{fname}.json'
            
            with open(filename, "w") as f:
                json.dump(history, f)
        except:
            print(f'error finding edit history for {target}')

    def save_all_histories_multiprocessing(self, targets, dir_path, processes=8):
        """
        the above, but with multiprocessing
        """
        # find or make our target directory
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        pool = Pool(processes = processes)

        with tqdm(pool.imap_unordered(self.process_target, targets), 
                  total=len(targets), 
                  desc='saving page edit hist') as target_iter:
            for _ in target_iter:
                pass
            
    def pages_full(self, targets, outdir, skip_cats=[]):
        target_articles = self.get_target_articles(targets, skip_cats)
        print('sampling 1000 articles out of {}'.format(len(target_articles)))
        target_articles = sample(target_articles, 1000)
        self.save_all_histories_multiprocessing(target_articles, outdir)
        #self.save_all_histories(target_articles, outdir)

    def sentiment(self, text):
        return self.analyzer.sentiment(text)

    def get_article_sentiment(self, datetextlist):
        """
        given output from get_all_page_edits, get sentiment for each text
        output in the form of tuples (date, sentiment)

        [{time, text}, {time, text} ...] ->  [{time, sentiment}, {time, sentiment} ...]
        """
        return [{'time': item['time'], 'sentiment': self.sentiment(item['text'])} 
                for item in datetextlist]
   
    def get_article_name_from_filename(self, filename):
        return os.path.splitext(filename)[0]
        #return os.path.splitext(os.path.basename)[0]

    def process_sentiment(self, filename):
        article_name = get_article_name_from_filename(filename)
        with open(filename, "r") as f:
            datetextlist = json.load(f)
            try:
                return article_name, self.get_article_sentiment(datetextlist)
            except:
                print(f'error finding sentiment for {article_name}')
        return article_name, none # error return value

    def get_all_sentiments(self, indir):
        target_files = os.listdir(indir)
        sentiment_data = {}
        with tqdm(target_files, total=len(target_files), desc='getting page sentiment') as dir_iter:
            for filename in dir_iter:
                article_name = self.get_article_name_from_filename(filename)
#                 target_iter.set_postfix({
#                     'target': article_name
#                 })
                with open(f'{indir}/{filename}', "r") as f:
                    datetextlist = json.load(f)
                    try:
                        sentiment_data[article_name] = self.get_article_sentiment(datetextlist)
                    except:
                        print(f'error finding sentiment for {article_name}')
        return sentiment_data


    def get_all_sentiments_multiprocessing(self, indir, processes=8):
        target_files = os.listdir(indir)
        sentiment_data = {}

        pool = Pool(processes = processes)

        with tqdm(pool.imap_unordered(self.process_sentiment, target_files), 
                  total=len(target_files), 
                  desc='getting page sentiment') as dir_iter:
            for article_name, sentiment in dir_iter:
                sentiment_data[article_name] = sentiment


    def save_sentiment(self, sentimentlist, filename):
        """
        save dict of list of dict of sentiment data to json
        """
        with open(filename, "w") as f:
            json.dump(sentimentlist, f)

    def sentiment_full(self, indir, outfile):
        sentiment_data = self.get_all_sentiments_multiprocessing(indir)
        print('saving...')
        self.save_sentiment(sentiment_data, outfile) 

    def load_sentiment(self, filename):
        """
        load dict of list of dict of sentiment as saved in save_sentiment
        dict { // overarching data structure
         "page_name": list( // each page has a list of edits
           dict{ // each edit dict has its date as a string and its sentiment as a float
            "time": datetime string,
            "sentiment": float
           }, ...
          ), ...
        }
        """
        with open(filename, "r") as f:
            return json.load(f)

    def fe_regression(self, sentiment_data, language):
        # do the fixed effects regression
        # save plot to a file
        res1, res2 = results(sentiment_data, language)

    def results_full(self, infile, language):
        sentiment_data = self.load_sentiment(infile)
        self.fe_regression(sentiment_data, language)
    
    def combine_lists():
        '''
        Combine all sentiment json files into single json file
        '''
        result = []
        for f in glob.glob("*.json"):
            with open(f, "rb") as infile:
                result.append(json.load(infile)) #combine into a Python list
        with open("merged_file.json", "wb") as outfile:
            json.dump(result, outfile) #write list to file as a json
