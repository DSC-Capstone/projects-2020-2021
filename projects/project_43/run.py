#!/usr/bin/env python

from os import listdir, path, makedirs
import sys
import json
from src.data import etl
from src.baselines import mostPop, randomFor, conBased


def main(targets):
    """ Runs data pipeline to parse all the data into these folders and turn movie title data into a csv"""

    if targets == 'test':
        filepath = 'config/test_params.json'
        with open(filepath) as file:
            configs = json.load(file)

        etl.main(configs)
        #rmse.main(configs)
        
        print("####################")
        mostPop.main(configs)
        randomFor.main(configs)
        conBased.main(configs)
        print("####################")

    if targets == 'data' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)
            
        etl.main(configs)
        
    #if targets == 'train' or targets == 'all':
    #    filepath = 'config/train_eval_params.json'
    #    with open(filepath) as file:
    #        configs = json.load(file)
        
    #    train.main(configs)
        
    #if targets == 'rmse' or targets == 'all':
    #    filepath = 'config/rmse_params.json'
    #    with open(filepath) as file:
    #        configs = json.load(file)        
        
    #    rmse.main(configs)

    return None


if __name__ == '__main__':
    targets = sys.argv[1]
    main(targets)
