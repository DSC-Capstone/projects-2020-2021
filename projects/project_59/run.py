# import custom modules
import sys
sys.path.insert(0, "src/util")
sys.path.insert(0, "src/model")
sys.path.insert(0, "src/data_util")

# imports for model
import torch
import torchvision
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

from baseline import *

from nbdt.model import SoftNBDT
from nbdt.model import HardNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.loss import HardTreeSupLoss

from wn_utils import *
from graph import *
from dir_grab import *
from hierarchy import *
from debug_data import *
from write_to_json import *
from loss import *

from datetime import datetime

def main(targets):
    '''
    runs project code based on targets
    
    configure filepaths based on data-params.json
    
    targets:
    data - builds data from build.sh, will run if no model folder is found.
    test - checks if data target has been run, runs model to train
    hierarchy - creates induced hierarchy and visualizes it
    '''
    
    if 'data' in targets:
        print('---> Running data target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
        
        # check for directory's existence and rename, raise if no directory exists
        DATA_DIR = data_cfg['dataDir']
        if os.path.isdir(os.path.join(DATA_DIR, 'train')): # default name after extraction
            os.rename(
                os.path.join(DATA_DIR, 'train'),
                os.path.join(DATA_DIR, 'train_snakes_r1')
            )
        elif not os.path.isdir(os.path.join(DATA_DIR, 'train')) | os.path.isdir(os.path.join(DATA_DIR, 'train_snakes_r1')):
            raise Exception('Please run build.sh before running run.py')
        
        # important name variables
        TRAIN_DIR = os.path.join(DATA_DIR, 'train_snakes_r1')
        VALID_DIR = os.path.join(DATA_DIR, 'valid_snakes_r1') # new dir to be made
        train_pct = 0.8
        
        # delete corrupted data from download
        delete_corrupted(TRAIN_DIR)

        # create validation set
        create_validation_set(DATA_DIR, TRAIN_DIR, VALID_DIR, train_pct)
        
        print("---> Finished running data target.")
        
    if 'train' in targets:
        print('---> Running train target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
            
        # check that data target has been ran
        VALID_DIR = os.path.join(data_cfg['dataDir'], 'valid_snakes_r1')
        if not os.path.isdir(VALID_DIR):
            raise Exception('Please run data target before running test')
        
        if 'SoftTreeSupLoss' in targets:
            criterion = SoftTreeLoss_wrapper(data_cfg)
        elif 'HardTreeSupLoss' in targets:
            criterion = HardTreeLoss_wrapper(data_cfg)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # create and train model
        model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = run_model(data_cfg, model_cfg, criterion)
        
        # write performance to data/model_logs
        write_model_to_json(
            loss_train,
            acc_train,
            fs_train,
            loss_val,
            acc_val,
            fs_val,
            n_epochs = model_cfg['nEpochs'],
            model_name = model_cfg['modelName'],
            fp = model_cfg['performancePath']
        )
        
        print("---> Finished running train target.")
        
    if 'test' in targets:
        print('---> Running test target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
        
        # check that data target has been ran
        VALID_DIR = os.path.join(data_cfg['dataDir'], 'valid_snakes_r1')
        if not os.path.isdir(VALID_DIR):
            raise Exception('Please run data target before running test')
        
        # create and train model
        print("!!! Please enter either 1 for (SoftTreeSupLoss, SoftNBDT) or 2 for (HardTreeSupLoss, HardNBDT) !!!")
        loss_type = input()
        
        assert (
            loss_type in ['1','2']
        ), "Please input either 1 or 2."
        
        if loss_type == '1':
            loss_type = 'SoftTreeSupLoss'
        elif loss_type == '2':
            loss_type = 'HardTreeSupLoss'
        
        run_nbdt(data_cfg, model_cfg, loss_type)
        
        print("---> Finished running test target.")
        
    if "hierarchy" in targets:
        print('---> Runnning hierarchy target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
            
        # use pretrained densenet
        model = models.densenet121(pretrained=True)
        # set features from classes, in this case 45, input_size always 224
        model.classifier = nn.Linear(model.classifier.in_features, model_cfg['nClasses'])
        input_size = model_cfg['inputSize']
        
        ## load state dict from previous
        if not os.path.exists(data_cfg['hierarchyModelPath']):
            raise Exception('Please run train target before hierarchy target, or change hierarchyModelPath in data-params if model has been trained.')
        model_weights = torch.load(data_cfg['hierarchyModelPath'])
        model.load_state_dict(model_weights)
        
        # generate hierarchy
        print("---> Generating hierarchy...")
        generate_hierarchy(
            dataset='snakes',
            arch = data_cfg['hierarchyModel'],
            model = model,
            method = 'induced'
        )
        print("---> Finished generating hierarchy.")
        
        # test hierarchy
        print("---> Testing hierarchy...")
        test_hierarchy(
            'snakes',
            os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON'])
        )
        
        generate_hierarchy_vis(
            os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
            'snakes'
        )
        
    if "nbdt_loss" in targets:
        print('---> Runnning nbdt_loss target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)        
        
        
    if "inference" in targets:
        print('---> Runnning baseline_cnn target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
        
        print("---> Finished running baseline_cnn target.")
        
        
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)