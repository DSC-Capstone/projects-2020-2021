import sys
import os
import json
import subprocess

sys.path.insert(0, 'src')

import env_setup
from analysis.data_analysis import *
from analysis.daily_sentiment import *
from data.clean_text import *
from data.extract_to_csv import *
from data.case_download import *
from data.train_dataset import *
from features.svc import *
from features.logreg import *
from models.timeseries import *



def main(targets):

    test = False
    env_setup.make_datadir()
    test_targets = ['test-data','analysis','feature', 'model']

    if 'test' in targets:
        targets = test_targets
        test = True
    

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_param = json.load(fh)

        all_dates = get_date_list(data_param["start_date"],data_param["end_date"])
        for i in all_dates:
            subprocess.call([data_param["script_type"], data_param["script_path"], i])
        convert_all_json(data_param['data_path'], data_param['data_path'])
        clean_all_csv(data_param['data_path'], data_param['out_path'])
        total_case(data_param['case_csv'], data_param['out_path'], data_param['start_date'], data_param['end_date'], data_param['url'])
        get_training_dataset(**data_param)


    if 'test-data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        test = True
        data_cfg['test'] = test
        clean_all_csv(data_cfg['path_test'], data_cfg['out_path'])
        get_training_dataset(**data_cfg)


    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            ana_cfg = json.load(fh)
        ana_cfg['test'] = test
        plot_daily_sentiment(**ana_cfg)
        analyze_data(**ana_cfg)


    if 'feature' in targets:
        with open('config/feature-params.json') as fh:
            fea_cfg = json.load(fh)
        build_logreg(**fea_cfg)
        build_svc(**fea_cfg)


    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        model_cfg['test'] = test
        make_prediction(**model_cfg)


    if 'all' in targets:
        main(['data', 'analysis', 'feature', 'model'])


    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
