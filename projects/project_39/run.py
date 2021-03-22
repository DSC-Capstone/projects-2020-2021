#!/usr/bin/env python

import sys
import json
import subprocess
import time

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')

from etl import get_data
from eda import analyze_data
from model import train


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'.

    `main` runs the targets in order of data=>analysis=>model.
    '''

    data = None
    if 'data' in targets:
        with open('config/data-params.json', 'r', encoding = "utf-8") as fh:
            data_cfg = json.load(fh)
        # make the data target
        data = get_data(**data_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        # make the data target
        analyze_data(data, **analysis_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        # make the data target
        train(**model_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    print("The arguments are:", str(sys.argv))
    print("The name of the script:", sys.argv[0])
    print("Number of arguments:", len(sys.argv))
    print("targets:", targets)
    main(targets)
