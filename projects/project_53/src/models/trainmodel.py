import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils, datasets
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

def cross_accuracy(y_pred,y): 
    ypt = torch.log_softmax(y_pred, dim=1)
    temp, ypt = torch.max(ypt, dim=1)
    acc = ((ypt == y).sum() / len(y)) * 100
    return acc


def train_model():
    print('Begin Training') 

    for e in tqdm(range(0,2)):
        e_loss, e_acc = 0, 0
        model.train()
        for X_train, y_train in train_loader: 
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            ytp = model(X_train).squeeze()
            t_loss = criterion(ytp, y_train)
            t_acc = cross_accuracy(ytp, y_train)
            t_loss.backward()
            optimizer.step()

            e_loss += t_loss.item()
            e_acc += t_acc.item()
        loss_stats['train'].append(e_loss/len(train_loader))
        accuracy_stats['train'].append(e_acc/len(train_loader))
        print("Done with Epoch {}".format(e))

    print('Training Finished')