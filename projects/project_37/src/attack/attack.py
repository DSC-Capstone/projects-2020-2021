from __future__ import print_function
from scipy import sparse
import pandas as pd
import numpy as np
import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.imbalanced_dataset_sampler.imbalanced import ImbalancedDatasetSampler

def attack():
    dataset = HindroidDataset(**dataset_args)
    fit_substitute()
    pass

def fit_substitute(train_dataset_args, test_dataset_args, no_cuda=False):
    pass
#     use_cuda = not no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)

#     device = torch.device("cuda" if use_cuda else "cpu")

#     train_kwargs = {'batch_size': batch_size}
#     test_kwargs = {'batch_size': batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     train_dataset = HindroidDataset(**train_dataset_args)
#     test_dataset = HindroidDataset(**test_dataset_args)
#     train_dataset = HindroidDataset(**train_datset_args)
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         sampler = ImbalancedDatasetSampler(
#             train_dataset, 
#             callback_get_label = hindroid_custom_get_label),
#         **train_kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         sampler = ImbalancedDatasetSampler(
#             test_dataset, 
#             callback_get_label = hindroid_custom_get_label),
#         **train_kwargs)

#     model = HindroidSubstitute().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         scheduler.step()

#     torch.save(model.state_dict(), "mnist_cnn.pt")

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
    
def train(model, device, train_loader, optimizer, epoch, weight=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target, weight=weight) # do we use different loss?
        loss.backward()
        optimizer.step()
        
        # logging
        log_interval = 10
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


def test(model, device, test_loader, weight=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target, weight=weight, reduction='sum').item()  # sum up batch loss
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))