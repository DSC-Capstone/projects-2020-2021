# dependencies & imports

import sys, json, os
import warnings 
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')

from data import make_dataset
from features import build_features, build_images, build_labels

# define main
def main(targets):

    if ('test' in targets):
        
        # get test params
        with open('config/test_params.json') as fh:
            data_cfg = json.load(fh)
        
        raw_fp = data_cfg['raw_path']
        test_fn = data_cfg['file_name']
        market_fn = data_cfg['market_name']
        time_wdw = data_cfg['time_wdw']
        img_fp = data_cfg['output_img_path']
        label_fp = data_cfg["output_lable_path"]
        
        data_file = os.path.join(raw_fp, test_fn)
        data = pd.read_csv(data_file, parse_dates = ['time'])
        
        market_file = os.path.join(raw_fp, market_fn)
        market_data = pd.read_csv(market_file, parse_dates = ['time'])

    # all case
    else:
        
        # get all params
        with open('config/data_params.json') as fh:
            data_cfg = json.load(fh)
        
        raw_fp = data_cfg['raw_path']
        time_wdw = data_cfg['time_wdw']
        img_fp = data_cfg['output_img_path']
        
        # merged 8:30-9:30 data
        data = make_dataset.merge_data(raw_fp)
        
    # data with volatility
    data = build_features.feature_engineer(data, time_wdw)
    # data for gramian angular field
    data_gaf = make_dataset.gaf_df(data)
    # creates images from polar coordinates, saves them to img_fp
    build_images.gramian_img(img_fp, data_gaf)
    # creates a table of image id and its corresponding label, saves it to label_fp
    build_labels.label (data, market_data, label_fp)
    
    return 

if __name__ == '__main__':
    
    targets = sys.argv[1:]
    main(targets)
