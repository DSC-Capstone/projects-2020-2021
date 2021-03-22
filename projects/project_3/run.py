import sys
import os
import json

sys.path.insert(0, 'src')

from etl import get_data
from clean import clean_data
from compute import analyze

def main(targets):
    if 'data' in targets:
        with open('config/etl-params.json') as fh:
            data_cfg = json.load(fh)
        get_data(**data_cfg)

    if 'clean' in targets:
        with open('config/clean-params.json') as fh:
            clean_cfg = json.load(fh)
        clean_data(**clean_cfg)

    if 'compute' in targets:
        with open('config/compute-params.json') as fh:
            compute_cfg = json.load(fh)
        analyze(**compute_cfg)

    if 'test' in targets:
        #gets clean data and does the eda target
        with open('config/test_clean-params.json') as fh:
            clean_cfg = json.load(fh)
        clean_data(**clean_cfg)

        with open('config/test_compute-params.json') as fh:
            compute_cfg = json.load(fh)
        analyze(**compute_cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)