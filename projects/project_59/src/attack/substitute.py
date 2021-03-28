# Adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
import os, argparse
os.chdir('/home/rcgonzal/DSC180Malware/m2v-adversarial-hindroid/')

from __future__ import print_function
from scipy import sparse
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.imbalanced_dataset_sampler.imbalanced import ImbalancedDatasetSampler

class HindroidSubstitute(nn.Module):
    def __init__(self, n_features):
        super(HindroidSubstitute, self).__init__()
        self.layer_1 = nn.Linear(n_features, 64, bias=False)
        # Linear - how to freeze layer ^
        # biases = false
        self.layer_2 = nn.Linear(64, 64, bias=False)
        self.layer_3 = nn.Linear(64, 64, bias=False)
        self.layer_4 = nn.Linear(64, 2, bias=False)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x # logits

class HindroidDataset(Dataset):
    def __init__(self, features_path, labels_path, label_col='m2vDroid', transform=None):
        '''
        Creates a  dataset from the A matrix representation of apps and their associated labels.
        
        Parameters:
        -------------------
        features_path: Path to A matrix in sparse format.
        labels_path: Path to labels in csv format.
        label_col: Default 'm2vDroid'. Useful for specifying which kernel to use for HinDroid.
        '''
        self.features = sparse.load_npz(os.path.join(features_path))
        self.feature_width = self.features.shape[1]
        features_folder = os.path.split(features_path)[0]
        self.features_idx = list(pd.read_csv(
            os.path.join(features_folder, 'predictions.csv'),
            usecols=['app'], 
            squeeze=True
        ))
        self.transform = transform
        
        try:
            self.labels = pd.read_csv(
                labels_path, 
                usecols=['app', label_col],
                index_col = 'app',
                squeeze=True
            )
            self.labels = self.labels[self.features_idx].values # align labels with features index
        except (KeyError, ValueError) as e:
            print(e)
            print('Seems like you may be trying to use a different model. This class is setup for m2vDroid by default.')
            print('For HinDroid you must specify `label_col` as either AAT, ABAT, APAT, ABPBTAT, or APBPTAT.')
            
        assert (self.features.shape[0] == self.labels.size), 'Length mismatch between features and labels.'
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = self.features[idx]
        features = features.todense().astype('float').A
        labels = self.labels[idx]
        
#         if self.transform:
#             features = self.transform(features)
#             labels = self.transform(labels)
        
#         sample = {'features': features, 'labels': labels}
        
        return features, labels
    
    def get_labels(self, idx):
        return self.labels[idx]
    
def hindroid_custom_get_label(dataset, idx):
    return dataset.get_labels(idx)