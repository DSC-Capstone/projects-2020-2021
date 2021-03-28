#!/usr/bin/env python

import sys
import json
import os
import pathlib
from pathlib import Path

import src
from src.data import collect_data
from src.data import preprocess_data
from src.features import create_features
from src.models import train_model
from src.utils import ensure_path_exists

import logging

# Handle NumExpr environment variable
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

def main(targets):

    # Will change to test config path if test target is seen
    config_dir = 'config'
    run_all = False

    # Set up logging
    with open(Path(config_dir, 'logging.json')) as f:
        logging_params = json.load(f)

    if logging_params['produce_logs']:
        log_file = logging_params['log_file']
        ensure_path_exists(log_file)
        logging.basicConfig(
            filename=log_file, filemode='a',
            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.DEBUG
        )
        logging.info(f"{'*'*80}\nBEGIN RUN\n{'*'*80}")

    # Regardless of if a logfile is being collected, we should also get the logs
    # to show up in standard out.
    logging.getLogger().addHandler(logging.StreamHandler())

    if 'all' in targets or len(targets) == 0:
        run_all = True

    if 'clean' in targets:
        # Would probably just delete the data folder... but should truly look at
        # the configuration to decide what to delete.
        raise NotImplementedError
    
    if 'test' in targets:
        # If `test` is the only target seen, then run all targets with the 
        # configs and data found in the test directory.
        #
        # Otherwise, if additional targets are specified then only run those
        # targets but still use test config (and therefore test data).
        # print('Test target recognized. Will use test configuration files.')
        logging.info('Test target recognized. Will use test configuration files.')
        config_dir = 'test/config'

        if len(targets) == 1:
            # print('Testing all targets: `data`, `features`, `train`.')
            run_all = True
    
    if 'data' in targets or run_all:
        # Load, clean, and preprocess data. Then store preprocessed data to
        # configured intermediate directory.
        # print('Data target recognized.')
        logging.info('Data target recognized.')

        with open(Path(config_dir, 'data-params.json'), 'r') as f:
            data_params = json.load(f)

        print('Running ETL pipeline.')
        logging.info('Running ETL pipeline.')
        preprocess_data(**data_params)
        print('ETL pipeline complete.')
        logging.info('ETL pipeline complete.')

    if 'features' in targets or run_all:
        # Creates features for preprocessed data and stores feature-engineered
        # data to a configured csv and directory.
        # print('Features target recognized.')
        logging.info('Features target recognized.')

        with open(Path(config_dir, 'features-params.json'), 'r') as f:
            features_params = json.load(f)

        # print('Engineering features.')
        logging.info('Engineering features.')
        create_features(**features_params)
        # print('Feature engineering complete.')
        logging.info('Feature engineering complete.')
         
    if 'train' in targets or run_all:
        # Trains model based on feature-engineeered data, report some of its
        # scores, and save the model.
        # print('Train target recognized.')
        logging.info('Train target recognized.')

        with open(Path(config_dir, 'train-params.json'), 'r') as f:
            train_params = json.load(f)

        # print('Training model.')
        logging.info('Training model.')
        train_model(**train_params)
        # print('Model training complete.')
        logging.info('Model training complete.')

    if 'generate' in targets:
        # Generates data from network-stats
        #
        # NOTE: This target should *not* be included in `all`.
        # print('Generate target recognized.')
        logging.info('Generate target recognized.')

        with open(Path(config_dir, 'generate-params.json'), 'r') as f:
            generate_params = json.load(f)

        # print('Collecting data with network-stats.')
        logging.info('Collecting data with network-stats.')
        collect_data(**generate_params)
        # print('Data collection complete.')
        logging.info('Data collection complete.')

    return

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
