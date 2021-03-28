#!/usr/bin/env python
import sys
import json

from src.test import test
from src.conversion import convert
from src.viz_analysis import plot_all

def main(targets):
    data_cfg = json.load(open('config/conversion.json'))
    viz_cfg = json.load(open('config/viz-params.json'))
    test_cfg = json.load(open('config/test.json'))
    
    if 'test' in targets:
        test(**test_cfg)
        print('Successfully Completed. Plots are saved.')
    if 'conversion' in targets:
        convert(**data_cfg)
        print('Raw Data Bags Extracted and Converted to csv')
    if 'viz_analysis' in targets:
        plot_all(**viz_cfg)
        print('Data plotted')
    return





if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)


