import sys
import json
import pandas as pd
from tqdm import tqdm
from os import listdir

import nltk
nltk.download('wordnet')

sys.path.insert(1, './src/')
from data_preprocessing import *
from data_downloads import *
from feature_encoding import *
from reports import *
from train import *

os.system('mkdir -p data')

data_prep_config = json.load(open('config/data_prep.json', 'r'))
feature_encoding_config = json.load(open('config/feature_encoding.json', 'r'))
# test_config = json.load(open('config/test.json', 'r'))
eda_config = json.load(open('config/eda.json', 'r'))
train_config = json.load(open('config/train.json', 'r'))
final_report_config = json.load(open('config/final_report.json', 'r'))

# By default, not testing
testing = False

# By default, notebook should not be in testing mode
notebook_config = {'testing': False}
with open('./config/notebook.json', 'w') as outfile:
    json.dump(notebook_config, outfile)

def data_prep(data_prep_config):
    # "raw_8k_fp": "8K-gz/",
    # "raw_eps_fp": "EPS/",

    global testing
    if testing:
        data_prep_config['testing'] = True
        data_prep_config['data_dir'] = './test/'

    data_dir = data_prep_config['data_dir']
    raw_dir = data_dir + data_prep_config['raw_dir']

    # Download RAW data if needed (and if not in testing mode)
    if not data_prep_config['testing']:
        if 'raw' not in listdir(data_dir):
            os.system('mkdir ' + raw_dir)
        if '8K-gz' not in listdir(raw_dir):
            download_8k(raw_dir)
        if 'EPS' not in listdir(raw_dir):
            download_eps(raw_dir)
        if 'price_history' not in listdir(raw_dir):
            download_price_history(raw_dir)
        if 'sp500.csv' not in listdir(raw_dir):
            os.system('cp ./test/raw/sp500.csv ' + raw_dir)
        print(' => All raw data ready!')

    # Process 8K, EPS and Price History as needed
    processed_dir = data_dir + data_prep_config['processed_dir']
    os.system('mkdir -p ' + processed_dir)

    # handler_clean_8k(data_prep_config['data_dir'])

    if not testing: # only process eps when it's not testing
        handler_process_eps(data_dir)
    # Run part 3, 4
    updated_merged_df = handle_merge_eps8k_pricehist(data_dir)
    updated_merged_df.to_csv(processed_dir + 'merged_all_data.csv', index = False)
    print()
    print(' => Done Data Prep!')
    print()

def feature_encoding(feature_encoding_config):
    global testing
    if testing:
        feature_encoding_config['data_dir'] = './test/'

    data_dir = feature_encoding_config['data_dir']
    data_file = data_dir + feature_encoding_config['data_file']
    phrase_file = data_dir + feature_encoding_config['phrase_file']
    out_dir = data_dir + feature_encoding_config['out_dir']
    n_unigrams = feature_encoding_config['n_unigrams']
    threshhold = feature_encoding_config['threshhold']

    merged_data, unigram_features = text_encode(data_file, phrase_file, n_unigrams, threshhold, out_dir = out_dir)
    print(' => Exporting...')
    merged_data.to_pickle(out_dir + 'feature_encoded_merged_data.pkl')
    unigram_features.to_csv(out_dir + 'model_unigrams.csv', index = False)

def handle_train(train_config):
    global testing
    if testing == True:
        train_config['data_dir'] = './test/'
        train_config['testing'] = True
    train(train_config)

# def handle_final_report(report_config):
#     # global testing
#     # if testing == True:
#     #     report_config['data_dir'] = './test/'
#     generate_report_from_notebook(report_config)
#
# def handle_eda(eda_config):
#     # global testing
#     # if testing == True:
#     #     eda_config['data_dir'] = './test/'
#     generate_report_from_notebook(eda_config)

def main():
    if len(sys.argv) == 1:
        target = 'all'
    else:
        target = sys.argv[1]

    # testing = False
    if target == 'data_prep':
        data_prep(data_prep_config)
    elif target == 'feature_encoding':
        feature_encoding(feature_encoding_config)
    elif target == 'eda':
        generate_report_from_notebook(eda_config)
    elif target == 'train':
        handle_train(train_config)
    elif target == 'report':
        generate_report_from_notebook(final_report_config)
    elif target == 'test':
        global testing
        testing = True
        notebook_config['testing'] = True
        eda_config['data_dir'] = './test/'
        final_report_config['data_dir'] = './test/'
        with open('./config/notebook.json', 'w') as outfile:
            json.dump(notebook_config, outfile)

        data_prep(data_prep_config)
        feature_encoding(feature_encoding_config)
        generate_report_from_notebook(eda_config)
        handle_train(train_config)
        generate_report_from_notebook(final_report_config)

main()
