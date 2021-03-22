import torch
import torch_geometric
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm.notebook import tqdm
import numpy as np
import yaml

from GraphDataset import GraphDataset

from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split


import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

class EdgeBlock(torch.nn.Module):
    def __init__(self):
        super(EdgeBlock, self).__init__()
        #A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        #Alternatively, an ordered dict of modules can also be passed in.
        
        #Applies a linear transformation to the incoming data: y = xA^T + by=xA^T+b
        self.edge_mlp = Seq(Lin(48*2, 128), # changed 2 to 6
                            BatchNorm1d(128),
                            ReLU(),
                            Lin(128, 128))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], 1)
        return self.edge_mlp(out)

class NodeBlock(torch.nn.Module):
    def __init__(self):
        super(NodeBlock, self).__init__()
        self.node_mlp_1 = Seq(Lin(48+128, 128), 
                              BatchNorm1d(128),
                              ReLU(), 
                              Lin(128, 128))
        self.node_mlp_2 = Seq(Lin(48+128, 128), 
                              BatchNorm1d(128),
                              ReLU(), 
                              Lin(128, 128))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

    
class GlobalBlock(torch.nn.Module):
    def __init__(self):
        super(GlobalBlock, self).__init__()
        self.global_mlp = Seq(Lin(128, 128),                               
                              BatchNorm1d(128),
                              ReLU(), 
                              Lin(128, 6)) #changed to 6 from 2

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


class InteractionNetwork(torch.nn.Module):
    def __init__(self):
        super(InteractionNetwork, self).__init__()
        self.interactionnetwork = MetaLayer(EdgeBlock(), NodeBlock(), GlobalBlock())
        self.bn = BatchNorm1d(48)
        
    def forward(self, x, edge_index, batch):
        
        x = self.bn(x)
        x, edge_attr, u = self.interactionnetwork(x, edge_index, None, None, batch)
        return u
    
@torch.no_grad()
def test(model,loader,total,batch_size,leave=False):
    model.eval()
    
    # double check tensor shape
    # try smallest being 1 instead - nathan
    # make the model bigger, add more neurons
#     training_weights = [3.479, 4.002, 3.246, 2.173, 0.253, 1.360]    
#     training_weights = [3.479, 4.002, 3.246, 2.173, 0.253, 1.360]    
    xentropy = nn.CrossEntropyLoss(reduction='mean')#, weight = torch.Tensor(training_weights))

    sum_loss = 0.
    t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
    for i,data in t:
        data = data.to(device)
        y = torch.argmax(data.y,dim=1)
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_loss_item = xentropy(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size,leave=False, weights = None):
    model.train()
    
    training_weights = [3.479, 4.002, 3.246, 2.173, 0.253, 1.360]    
    training_weights2 = np.array([3.479, 4.002, 3.246, 2.173, 0.253, 1.360]) *(1/.253)  
    
    if weights:
        xentropy = nn.CrossEntropyLoss(reduction='mean', weight = torch.Tensor(weights))
    else:
        xentropy = nn.CrossEntropyLoss(reduction='mean')

    sum_loss = 0.
    t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
    for i,data in t:
        data = data.to(device)
        y = torch.argmax(data.y,dim=1)
        optimizer.zero_grad()
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_loss = xentropy(batch_output, y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)