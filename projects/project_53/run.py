#!/usr/bin/env python

from os import listdir, path, makedirs
import sys
import json
from src.data import etl, api_etl
from src.models import model, new_user, create_model
from src.baselines import create_baseline


def main(targets):
    """ Runs data pipeline to parse the data archived data and api data into the LightFM models and return analysis against
    baselines. Can run on normal or test data."""

    if targets == 'test':
        filepath = 'config/test_params.json'
        with open(filepath) as file:
            configs = json.load(file)
        
        etl.main(configs)
        print('')
        api_etl.main(filepath)
        print('')
        
        print('########### Created Model and Model Inputs ###########')
        create_model.main(configs)
        print(' ')
        print('########### Generate Recommendations ###########')
        model.main(configs)
        print('')
        print('########### Add User to Model ###########')
        new_user.main(configs)
        
        print('####################')
        print('########### Baseline ###########')
        create_baseline.main(configs)
        model.main(configs, True)
        print('####################')

    if targets == 'data' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)
        
        etl.main(configs)
        print('')
        
    if targets == 'api' or targets == 'all':               
        filepath = 'config/api_params.json'
        
        api_etl.main(filepath)
        print('')
        
    if targets == 'models' or targets == 'all':
        filepath = 'config/models_params.json'
        with open(filepath) as file:
            configs = json.load(file)
            
        print('########### Created Model and Model Inputs ###########')
        create_model.main(configs)
        print(' ')
        print('########### Generate Recommendations ###########')
        model.main(configs)
        print('')
        print('########### Add User to Model ###########')
        new_user.main(configs)
        
        
    if targets == 'baselines' or targets == 'all':
        filepath = 'config/models_params.json'
        with open(filepath) as file:
            configs = json.load(file)        
        
        print('####################')
        print('########### Baseline ###########')
        create_baseline.main(configs)
        models.main(configs, True)
        print('####################')

    return None


if __name__ == '__main__':
    targets = sys.argv[1]
    main(targets)