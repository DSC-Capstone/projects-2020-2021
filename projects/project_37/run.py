#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')
sys.path.insert(0, 'src')

from etl import run_etl
from analysis import generate_analysis
# from model import train
from utils import convert_notebook
from tester import run_tests

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'test' in targets:
        run_tests()

    if 'data' in targets:
        with open('config/etl-params/etl-params.json') as fh:
            etl_cfg = json.load(fh)
        
        run_etl(**etl_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        generate_analysis(**analysis_cfg)

    if 'report' in targets:
        with open('config/report-params.json') as fh:
            analysis_cfg = json.load(fh)
        convert_notebook(**analysis_cfg)
            
    if 'model' in targets:
        with open('config/model-params/model-params.json') as fh:
            model_cfg = json.load(fh)
        # make the data target
        train(**model_cfg)
    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
