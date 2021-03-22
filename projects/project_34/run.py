#!/usr/bin/env python

import re
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score

import src.features.feature
import src.models.model
import src.analysis.analysis

def main(target):
    '''
    Runs the main project pipeline logic.
    'feature' should be included in targets since it is the base for each other targets.
    'model' should be included in targets when 'test' is one of target.

    '''
    if 'feature' in targets:
        with open('config/feature.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        features_mal = parse_all(data_cfg['path_mal'], 1)
        features_benign = parse_all(data_cfg['path_benign'], 0)
        chains = generate_chains(features_mal, features_benign)
        X, y = generate_Xy(chains)


    if 'model' in targets:
        with open('config/model.json') as fh:
            data_cfg_model = json.load(fh)

        # make the data target
        X_train, X_test, y_train, y_test = ttsplit(X,y)
        reg_knn = build_KN(X_train, y_train, int(data_cfg_model['n_neighbors']))
        pred_train = reg_knn.predict(X_train)
        pred_test = reg_knn.predict(X_test)
        reg_advanced = build_KN(X_train, y_train, int(data_cfg_model['n_neighbors_adv']))
        pred_train_adv = X_train.apply(lambda x:advanced_predict(reg_advanced, x), axis=1)
        pred_test_adv = X_test.apply(lambda x:advanced_predict(reg_advanced, x), axis=1)


    if 'analysis' in targets:
        with open('config/eda.json') as fh:
            data_cfg_eda = json.load(fh)

        # make the data target
        TFNP_analysis(chains, reg_advanced)
        eda(chains, data_cfg_eda['plot'])


    if 'test' in targets:
        # make the data target
        test_mal = parse_all('test/testdata/mal', 1)
        test_benign = parse_all('test/testdata/ben', 0)
        test_chains = generate_chains(test_mal, test_benign)
        X_t, y_t = generate_Xy(test_chains)
        # output
        pred_t = X_t.apply(lambda x:advanced_predict(reg, x), axis=1)
        # performance
        accuracy_score(y_t, pred_t)


    return 'Finished'


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
