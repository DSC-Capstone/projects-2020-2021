#!/usr/bin/env python

import os
import sys
import json

sys.path.insert(0, 'src')
from etl import convert_txt
from model import autophrase
from weight_phrases import change_weight
from webscrape import webscrape
from website import activate_website
from utils import convert_report

#def main(targets):
def main():
    data_config = json.load(open('config/data-params.json'))
    model_config = json.load(open('config/model-params.json'))
    weight_config = json.load(open('config/weight-params.json'))
    webscrape_config = json.load(open('config/webscrape-params.json'))
    website_config = json.load(open('config/website-params.json'))
    report_config = json.load(open('config/report-params.json'))
    test_config = json.load(open('config/test-params.json'))

    os.system('git submodule update --init')
    
    # Getting the target
    # If no target is given, then run 'website'
    if len(sys.argv) == 1:
        targets = 'website'
    else:
        targets = sys.argv[1]
        
    if 'data' in targets:
        convert_txt(**data_config)
    if 'autophrase' in targets:
        autophrase(data_config['outdir'], data_config['pdfname'], model_config['outdir'], model_config['filename'])
    if 'weight' in targets:
        try:
            unique_key = '_' + sys.argv[2]
            change_weight(unique_key, **weight_config)
        except:
            change_weight(unique_key='', **weight_config)
    if 'webscrape' in targets:
        try:
            unique_key = '_' +  sys.argv[2]
            webscrape(unique_key, **webscrape_config)
        except:
            webscrape(unique_key='', **webscrape_config)
    if 'report' in targets:
        convert_report(report_config['experiment_in_path'], report_config['experiment_out_path'])
        convert_report(report_config['analysis_in_path'], report_config['analysis_out_path'])
    if 'website' in targets:
        activate_website(**website_config)
    if 'test' in targets:
        convert_txt(test_config['indir'], data_config['outdir'], test_config['pdfname'],)
        autophrase(data_config['outdir'], test_config['pdfname'], model_config['outdir'], model_config['filename'])
        change_weight(unique_key='', **weight_config)
        webscrape(unique_key='', **webscrape_config)
        convert_report(report_config['experiment_in_path'], report_config['experiment_out_path'])
        convert_report(report_config['analysis_in_path'], report_config['analysis_out_path'])
    return

#if __name__ == '__main__':
#    # run via:
#    # python main.py data features model
#    targets = sys.argv
#    main(targets)
main()
