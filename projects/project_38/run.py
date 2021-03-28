import sys
import os
import json
import requests
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import numpy as np
sys.path.insert(0, 'src')
from etl import get_data
from eda import do_eda
from auto import autophrase
from visual import visual
from example import example

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    `main` runs the targets in order of data=>analysis=>model.
    '''
    if 'all' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        get_data(**data_cfg)

        with open('config/eda-params.json') as fh:
            eda_cfg = json.load(fh)
        do_eda(**eda_cfg)

        with open('config/auto-params.json') as fh:
            auto_cfg = json.load(fh)
        autophrase(**auto_cfg)

        with open('config/visual-params.json') as fh:
            visual_cfg = json.load(fh)
        visual(**visual_cfg)

        with open('config/example-params.json') as fh:
            example_cfg = json.load(fh)
        example(**example_cfg)
    
    if 'test' in targets:
        with open('config/data-params-test.json') as fh:
            data_cfg = json.load(fh)
        get_data(**data_cfg)
        
        with open('config/eda-params-test.json') as fh:
            eda_cfg = json.load(fh)
        do_eda(**eda_cfg)

        with open('config/auto-params-test.json') as fh:
            auto_cfg = json.load(fh)
        autophrase(**auto_cfg)

        with open('config/visual-params-test.json') as fh:
            visual_cfg = json.load(fh)
        visual(**visual_cfg)


    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        get_data(**data_cfg)
    

    if 'eda' in targets:
        with open('config/eda-params.json') as fh:
            eda_cfg = json.load(fh)
        do_eda(**eda_cfg)


    if 'auto' in targets:
        with open('config/auto-params.json') as fh:
            auto_cfg = json.load(fh)
        autophrase(**auto_cfg)

    if 'visual' in targets:
        with open('config/visual-params.json') as fh:
            visual_cfg = json.load(fh)
        visual(**visual_cfg)
    
    if 'example' in targets:
        with open('config/example-params.json') as fh:
            visual_cfg = json.load(fh)
        visual(**visual_cfg)

    return
if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
