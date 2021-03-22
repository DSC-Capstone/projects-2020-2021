#!/usr/bin/env python
import pandas as pd
import sys
import os
import json
import shutil
import warnings
import pickle




sys.path.insert(0, 'src')
from data.etl import scrape_item_data, scrape_review_data
from features.collab_filter_feats import build_features
from models.collab_filter_model import collab_model_build
from evaluation.collab_filter_evaluation import auc_eval

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'features', 'model'. 
    
    `main` runs the targets in order of data=>features=>model.
    '''
    
    
    with open('config/data-params.json') as fh:
        data_cfg = json.load(fh)
        outdir = data_cfg['outputdir']
        chrome_path = data_cfg['chromeexecdir']
        
    # run all targets    
    if 'all' in targets:
        targets = ['data', 'features', 'model', 'accuracy']
        
    if 'test' in targets:
        targets += ['features', 'model', 'accuracy']
        with open('config/test-params.json') as fh:
            data_cfg = json.load(fh)['inputdir']



    
    
    # Scrape data from Sephora.com. 
    if 'item_data' in targets:
        scrape_item_data(outdir, chrome_path, "./data/cosmetic_url.csv")

    if 'review_data' in targets:
        scrape_review_data(outdir, chrome_path, "./data/cosmetic_url.csv")
        
        

   
        
    # create and save the features
    if 'features' in targets:
        (interactions, weights) = build_features(data_cfg)
                    
        
    
    # load the model using the saved features.
    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            num_threads = model_cfg['NUM_THREADS']
            num_components = model_cfg['NUM_COMPONENTS']
            num_epochs = model_cfg['NUM_EPOCHS']
            item_alpha = model_cfg['ITEM_ALPHA']
        collab_filter_model = collab_model_build(interactions, num_threads,
            num_components, num_epochs, item_alpha)
    
    # Get accuracy of the model
    if 'accuracy' in targets:
        auc_eval(collab_filter_model, interactions, num_threads)
        
        
    


    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)