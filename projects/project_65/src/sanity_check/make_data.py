import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import itertools
from torch_geometric.data import Dataset, Data, DataLoader

def make_data(n_features:str=48,
            n_tracks:int=10,
            n_samples:int=2000+500,
            x_idx:int=3,
            y_idx:int=3,
            save_to:str="../data/{}_sythesized.pt"):
    # generate irrelevant input
    X=torch.normal(mean=torch.zeros(n_samples*n_tracks,n_features),std=torch.ones(n_samples*n_tracks,n_features),)
    X_batch=torch.arange(n_samples).repeat_interleave(n_tracks)

    # generate relevant input
    x=torch.randint(0,2,(n_samples*n_tracks,))*10
    y=torch.randint(-2,2,(n_samples*n_tracks,))*10

    X[:,x_idx]+=x
    X[:,y_idx]+=y

    # graph label is sign(mean(x+2y))
    label=torch.sign(scatter_mean(x+2*y,X_batch))
    label[label<0]=0
    temp=torch.zeros(n_samples,2)
    temp[:,1]=label
    temp[label==0,0]=1
    label=temp

    # make graphs
    pairs=np.stack([[m, n] for (m, n) in itertools.product(range(n_tracks),range(n_tracks)) if m!=n])
    edge_index=torch.tensor(pairs, dtype=torch.long)
    edge_index=edge_index.t().contiguous()

    datas=[]
    for i in range(n_samples):
        start_idx,end_idx=i*10,i*10+10
        data=Data(x=X[start_idx:end_idx,:], edge_index=edge_index, y=label[i].view(1,2))
        data.u=None

        datas.append(data)
        
    # split into test and train
    train_data=datas[:2000]
    test_data=datas[2000:]

    # save to file
    torch.save(train_data, save_to.format("train"))
    torch.save(test_data, save_to.format("test"))