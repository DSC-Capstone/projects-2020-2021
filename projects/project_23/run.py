#!/usr/bin/env/python

import yaml
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, 'src')

from pipeline import WikiPipeline

PAGES_ARGS_PATH = 'config/pages-param.yml'
SENT_ARGS_PATH = 'config/sentiment-param.yml'
RESULTS_ARGS_PATH = 'config/results-param.yml'
TEST_PARGS_PATH = 'config/test-pparam.yml'
TEST_SARGS_PATH = 'config/test-sparam.yml'
TEST_RARGS_PATH = 'config/test-rparam.yml'

def read_yml(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.SafeLoader)

def run_pages(argpath):
    args = read_yml(argpath)
    for entry in args:
        lang = entry['lang']
        param = entry['param']
        print(f'running pages for lang {lang}...')
        pl = WikiPipeline(lang)
        pl.pages_full(**param)

def run_sentiment(argpath):
    args = read_yml(argpath)
    for entry in args:
        lang = entry['lang']
        param = entry['param']
        print(f'running sentiment for lang {lang}...')
        pl = WikiPipeline(lang)
        pl.sentiment_full(**param)

def run_results(argpath):
    args = read_yml(argpath)
    for entry in args:
        lang = entry['lang']
        param = entry['param']
        print(f'running results for lang {lang}...')
        pl = WikiPipeline(lang)
        pl.results_full(**param)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', 
        '--pages', 
        help='obtain list of pages to analyze from wikipedia', 
        action='store_true'
    )
    parser.add_argument(
        '-s', 
        '--sentiment', 
        help='run sentiment analysis on pages from list', 
        action='store_true'
    )
    parser.add_argument(
        '-r', 
        '--results', 
        help='stats and visuals from sentiment analysis', 
        action='store_true'
    )
    parser.add_argument(
        '-t', 
        '--test', 
        help='run test suite', 
        action='store_true'
    )

    args = parser.parse_args()
    if args.test:
        print('running test suite...')
        run_pages(TEST_PARGS_PATH)
        run_sentiment(TEST_SARGS_PATH)
        run_results(TEST_RARGS_PATH)
        print('tests are done')
    if args.pages:
        run_pages(PAGES_ARGS_PATH)
    if args.sentiment:
        run_sentiment(SENT_ARGS_PATH)
    if args.results:
        run_results(RESULTS_ARGS_PATH)
