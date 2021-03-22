#!/usr/bin/env python

import sys
import json
import pandas as pd
import os
import logging
import pickle

sys.path.insert(0, 'src')

from etl import get_data
from eda import generate_stats
from train import train_model
from analysis import compute_user_stats
from utils import convert_notebook
from logging.handlers import RotatingFileHandler
from statistics import compute_results

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'train', 'analysis', 'results'. 
    
    `main` runs the targets in order of data=>train=>analysis=>results.
    '''
    # Setup Logger
    logger = logging.getLogger('project_log')
    logger.setLevel(logging.DEBUG)
    fh = RotatingFileHandler('example.log', maxBytes=1000000, backupCount=0)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('STARTING PROGRAM')

    # Data Target
    if 'data' in targets or 'all' in targets:
        logger.info('Starting data target')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/twitter-api-keys.json') as fh:
            twitter_cfg = json.load(fh)
        get_data(logger, **data_cfg, **twitter_cfg)
        logger.info('Finishing data target')

    # Train Model target
    if 'train' in targets or 'all' in targets:
        logger.info('Starting train target')
        with open('config/train-params.json') as fh:
            train_cfg = json.load(fh)
        df = pd.read_csv(os.path.join(train_cfg['training_data_path'], 'data.csv')).drop(columns=['Unnamed: 0'])
        train_model(logger, df, **train_cfg)

        convert_notebook('train', **train_cfg)
        logger.info('finished train target: wrote html file to {}'.format(os.path.join(train_cfg['outdir'], 'train.html')))

    # Analysis target: calculate user polarities
    if 'analysis' in targets or 'all' in targets:
        logger.info('Starting analysis target')
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        # do user stats
        tweets = {}
        for tweet_id in analysis_cfg['tweet_ids']:
            path = os.path.join(analysis_cfg['user_data_path'], 'tweet_{}.csv'.format(tweet_id))
            tweet = pickle.load(open(path, 'rb'))
            tweets[tweet_id] = tweet
        for key, value in tweets.items():
            for user_id in list(value['user_ids'].keys()):
                value['user_ids'][user_id] = pd.read_csv(os.path.join(analysis_cfg['user_data_path'], 'user_{}_tweets.csv'.format(user_id)))
        mdls = []
        dims = analysis_cfg['dims']
        for dim in dims:
            path = os.path.join(analysis_cfg['model_path'], '{}.mdl'.format(dim))
            mdl = pickle.load(open(path, 'rb'))
            mdls.append(mdl)
        compute_user_stats(logger, tweets, mdls, dims, analysis_cfg['user_data_path'], analysis_cfg['flagged'])

        convert_notebook('analysis', **analysis_cfg)
        logger.info('finished analysis target: wrote html file to {}'.format(os.path.join(analysis_cfg['outdir'], 'analysis.html')))

    # Results target: calculate results
    if 'results' in targets or 'all' in targets:
        logger.info('Starting results target')
        with open('config/results-params.json') as fh:
            results_cfg = json.load(fh)
        fp = os.path.join(results_cfg['user_data_path'], 'polarities.csv')
        polarities = pd.read_csv(fp, usecols=results_cfg['dims'] + ['flagged']).dropna()
        compute_results(logger, polarities, results_cfg['dims'], results_cfg['outdir'])

        convert_notebook('results', **results_cfg)
        logger.info('finished results target: wrote html file to {}'.format(os.path.join(results_cfg['outdir'], 'results.html')))


    # Test target
    if 'test' in targets or 'all' in targets:
        logger.info('Starting TEST target')

        # Train target
        logger.info('Starting TEST train target')
        with open('config/train-params.json') as fh:
            train_cfg = json.load(fh)
        df = pd.read_csv(os.path.join(train_cfg['training_data_path'], 'data.csv')).drop(columns=['Unnamed: 0'])
        train_model(logger, df, **train_cfg)
        convert_notebook('train', **train_cfg)
        logger.info('finished TEST train target: wrote html file to {}'.format(os.path.join(train_cfg['outdir'], 'train.html')))

        # Analysis target
        logger.info('Starting TEST analysis target')
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        # do user stats
        tweets = {}
        for tweet_id in analysis_cfg['tweet_ids']:
            path = os.path.join(analysis_cfg['user_data_path'], 'tweet_{}.csv'.format(tweet_id))
            tweet = pickle.load(open(path, 'rb'))
            tweets[tweet_id] = tweet
        for key, value in tweets.items():
            for user_id in list(value['user_ids'].keys()):
                value['user_ids'][user_id] = pd.read_csv(os.path.join(analysis_cfg['user_data_path'], 'user_{}_tweets.csv'.format(user_id)))
        mdls = []
        dims = analysis_cfg['dims']
        for dim in dims:
            path = os.path.join(analysis_cfg['model_path'], '{}.mdl'.format(dim))
            mdl = pickle.load(open(path, 'rb'))
            mdls.append(mdl)
        compute_user_stats(logger, tweets, mdls, dims, analysis_cfg['user_data_path'], analysis_cfg['flagged'])

        convert_notebook('analysis', **analysis_cfg)
        logger.info('finished TEST analysis target: wrote html file to {}'.format(os.path.join(analysis_cfg['outdir'], 'analysis.html')))

        # Results target: calculate results
        logger.info('Starting TEST results target')
        with open('config/results-params.json') as fh:
            results_cfg = json.load(fh)
        fp = os.path.join(results_cfg['user_data_path'], 'polarities.csv')
        polarities = pd.read_csv(fp, usecols=results_cfg['dims'] + ['flagged']).dropna()
        compute_results(logger, polarities, results_cfg['dims'], results_cfg['outdir'])

        convert_notebook('results', **results_cfg)
        logger.info('finished TEST results target: wrote html file to {}'.format(os.path.join(results_cfg['outdir'], 'results.html')))


        logger.info('finished TEST target')
        
    logger.info('ENDING PROGRAM')
   
    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
