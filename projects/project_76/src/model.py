#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""
Title: Graph Convolutional Networks in Pytorch
Date: February 25, 2019
Availability: https://github.com/tkipf/pygcn
"""

'''
This file contains a Pytorch implementation of our main model for stock movement prediction using graph convolutional networks.
It contains the model and the code that trains and tests the model.
'''

import torch
import torch.optim as optim
import math
import json

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from src.featureSpaceDay import *


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# In[15]:


import torch.nn as nn
import torch.nn.functional as F

class VanillaGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):#, dropout):
        super(VanillaGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x#F.log_softmax(x, dim=1)


# In[2]:


import numpy as np
import pandas as pd


# In[ ]:

def modelRun():
    #load in adjacency matrix
    # In[17]:
    #read in the numbers from the config folder
    with open('./config/model-params.json') as f:
        p = json.loads(f.read())
        num_days = p['num_days']
        NUM_EPOCHS = p['NUM_EPOCHS']
        LEARNING_RATE = p['LEARNING_RATE']
        NUM_HIDDEN = p['NUM_HIDDEN'];
        nfeats = p['nfeat']
        nclasses = p['nclass']
        dataset = p['dataset']

    adj = pd.read_csv(dataset)
    adj = adj.iloc[:,1:]
    colsTickers = adj.columns;
    print(adj);

    # In[3]:


    sample_adj = adj


    # In[12]:


    #normalize adjacency matrix
    #sample_adj = np.array([[x for x in range(3)] for y in range(3)])
    D = np.diag(sample_adj.sum(1))
    adj = np.linalg.inv(D**(1/2)).dot(sample_adj).dot(D**(1/2))




    model = VanillaGCN(nfeat= nfeats,
                nhid=NUM_HIDDEN,
                nclass=nclasses)


    # In[18]:


    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE)
    crit = nn.BCEWithLogitsLoss()


    # In[ ]:


    def accuracy(preds, labels):
        
        preds = torch.round(torch.sigmoid(preds))

        acc = torch.round((preds == labels).sum().float() / labels.shape[0] * 100)
        
        return acc


    # In[ ]:


    TRAINING_SIZE = int(124 * .7)
    adj = torch.FloatTensor(adj)
    #training loop
    for e in range(NUM_EPOCHS):
        #use 70% of days for training
        epoch_loss = 0
        for i in range(int(TRAINING_SIZE)):
            features, labels = featureDaySpace(i,num_days)
            features = features.ffill(axis =0);
            if features.isnull().sum().sum() > 0:

                print(features.isnull().sum().sum())

                
                print('day' + str(i));
                print(features);
                return;

            labels = torch.FloatTensor(np.array(labels))
            features = torch.FloatTensor(np.array(features))
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            #turn the labels from [30] to [30,1]
            labels = labels.unsqueeze(1).float()


            
            loss_train = crit(output, labels)
            acc_train = accuracy(output, labels)
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.item()
        epoch_loss /= TRAINING_SIZE
        print(f'Loss for epoch {e}: {epoch_loss}')


    # In[ ]:

    DATA_LENGTH = 124
    #test loop
    test_losses = []
    test_accs = []
    test_predictions = []
    recall= []
    precision = []
    for i in range(TRAINING_SIZE + 1, DATA_LENGTH - num_days - 1):
        features, labels = featureDaySpace(i,num_days)
        features = features.ffill(axis =0);
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(np.array(labels))
        model.eval()
        output = model(features, adj)
        labels = labels.unsqueeze(1).float()

        #change back to numpy array so sklearn methods can take it
        tempOutput = torch.round(torch.sigmoid(output)).detach().numpy() 
        tempLabel = labels.detach().numpy() 
        recall.append(recall_score(tempOutput,tempLabel))
        precision.append(precision_score(tempOutput,tempLabel))

        loss_test = crit(output, labels)
        acc_test = accuracy(output, labels)

        test_losses.append(loss_test.item())
        test_accs.append(acc_test.item())
        test_predictions.append(labels);
    total_test_loss = np.mean(test_losses)
    total_test_acc = np.mean(test_accs)

    test_predictions =['Bullish' if x == 1 else 'Bearish' for x in output]
    for x in range(len(test_accs)):
            print("Day " + str(x) + ":" + str(test_accs[x]));


    print();

    print('This is our total test loss: ' + str(total_test_loss))
    print('This is our test accuracy: ' + str(total_test_acc))
    print('This is our precision: ' + str(np.mean(precision)))
    print('This is our recall: ' + str(np.mean(recall)))

    for x in range(len(colsTickers)):
        print(colsTickers[x] +':' + test_predictions[x]);



