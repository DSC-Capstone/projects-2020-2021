import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from util import *
from data.make_dataset import clean_df
from data.reduce_api import run_reduce_api
from features.build_features import build_mat
from features.word2vec import build_w2v
from features.node2vec import build_n2v
from features.metapath2vec import build_m2v
from features.build_new_features import build_mat as new_build_mat
from models.run_model import run_model 
from models.clf import run_clf
from visualization.eda import generate

def load_params(fp):
    """
    Load params from json file 
    """
    with open(fp) as fh:
        param = json.load(fh)

    return param

def main(targets):
    """
    Runs the main project pipeline logic, given the target 
    targets must contain: 'baseline_df' ...
    """
    if 'baseline_df' in targets:
        params = load_params('config/data-params.json')
        clean_df(**params)
    
    if 'reduce_api' in targets:
        params = load_params('config/reduce_api.json')
        run_reduce_api(**params)

    if 'eda' in targets:
        params = load_params('config/eda-params.json')
        generate(**params)

    if 'feature_build' in targets:
        params = load_params('config/feature-params.json')
        build_mat(**params)

    if 'new_feature_build' in targets:
        params = load_params('config/new_feature-params.json')
        new_build_mat(**params)
    
    if 'run_model' in targets:
        params = load_params('config/test-params.json')
        run_model(**params)

    if 'word2vec' in targets:
        params = load_params('config/word2vec.json')
        build_w2v(**params)

    if 'node2vec' in targets:
        params = load_params('config/node2vec.json')
        build_n2v(**params)

    if 'metapath2vec' in targets:
        params = load_params('config/metapath2vec.json')
        build_m2v(**params)
        
    if 'clf' in targets:
        params = load_params('config/clf.json')
        run_clf(**params)


    if 'test' in targets:
        params = load_params('config/test/data-params.json')
        clean_df(**params)
        params = load_params('config/test/feature-params.json')
        build_mat(**params)
        params = load_params('config/test/test-params.json')
        run_model(**params)
        params = load_params('config/test/word2vec.json')
        build_w2v(**params)
        params = load_params('config/test/node2vec.json')
        build_n2v(**params)
        params = load_params('config/test/metapath2vec.json')
        build_m2v(**params)
        params = load_params('config/test/clf.json')
        run_clf(**params)
   

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)