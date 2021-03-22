import json
import os
import shutil

from etl import run_etl
from analysis import generate_analysis
import tensorflow as tf
# from model import model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run_tests():
    print("-----RUNNING TESTS-----")
    
    print("Testing ETL workflow...", end='')
    with open(os.path.join('config', 'test', 'data-params.json')) as fh:
        data_cfg = json.load(fh)

    run_etl(**data_cfg)
    
    # retry without data_source
    print("from app_list...")
    data_cfg['parse_params']['data_source'] = None
    os.remove(os.path.join('test', 'output', 'graph.pkl'))
    run_etl(**data_cfg)

    print("Testing Analysis workflow...")
    with open(os.path.join('config', 'test', 'analysis-params.json')) as fh:
        analysis_cfg = json.load(fh)
    
    generate_analysis(**analysis_cfg)
    
#     TODO
#     print("Testing Model workflow...")
#     with open(os.path.join('config', 'test', 'model-params.json')) as fh:
#         model_cfg = json.load(fh)
    
    shutil.rmtree(os.path.join('test', 'output'))

    print("ALL TESTS PASSED.")
