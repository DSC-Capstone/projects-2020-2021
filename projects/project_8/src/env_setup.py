import os
import json

basedir = os.path.dirname(__file__)

def make_datadir():
    data_loc = os.path.join(basedir,'..','data')
    analysis_loc = os.path.join(data_loc, 'analysis')
    int_loc = os.path.join(data_loc, 'interim')
    if not os.path.exists(data_loc):
        os.mkdir(data_loc)
        os.mkdir(int_loc)
        os.mkdir(analysis_loc)
    return