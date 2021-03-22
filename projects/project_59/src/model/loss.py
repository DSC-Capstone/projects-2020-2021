import torch.nn as nn
from nbdt.loss import SoftTreeSupLoss
from nbdt.loss import HardTreeSupLoss
import os

def SoftTreeLoss_wrapper(data_cfg):
    '''
    Creates SoftTreeSupLoss wrapper for our dataset from the nbdt package
    
    returns:
    criterion - SoftTreeSupLoss
    '''
    criterion = nn.CrossEntropyLoss()
    criterion = SoftTreeSupLoss(
        dataset = data_cfg['dataset'],
        hierarchy='induced-densenet121',
        path_graph = os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
        path_wnids = data_cfg['wnidPath'],
        criterion = criterion
    )
    
    return criterion

def HardTreeLoss_wrapper(data_cfg):
    '''
    Creates HardTreeSupLoss wrapper for our dataset from the nbdt package
    
    returns:
    criterion - HardTreeSupLoss
    '''
    criterion = nn.CrossEntropyLoss()
    criterion = HardTreeSupLoss(
        dataset = data_cfg['dataset'],
        hierarchy='induced-densenet121',
        path_graph = os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
        path_wnids = data_cfg['wnidPath'],
        criterion = criterion
    )
    
    return criterion
