from .InteractionNetwork import InteractionNetwork
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def test(model,loader,total,batch_size,leave=False):
    model.eval()
    
    xentropy = nn.CrossEntropyLoss(reduction='mean')

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


def train(model, optimizer, loader, total, batch_size,leave=False):
    model.train()
    
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


def main(train_data='../data/train_sythesized.pt',
        test_data='../data/test_sythesized.pt',
        model_save_to='../data/model/IN_sythesized.pth',
        fig_save_to='../data/model/IN_sythesized_roc.png'):
    """
    training and eval pipeline
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model=InteractionNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(train_data)
    valid_num = int(valid_frac*full_length)
    batch_size = 32

    train_dataset, valid_dataset = random_split(train_data, [full_length-valid_num,valid_num])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, shuffle=False)


    test_samples = len(test_data)
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)

    n_epochs = 10
    stale_epochs = 0
    best_valid_loss = 99999
    patience = 5
    t = tqdm(range(0, n_epochs))

    for epoch in t:
        loss = train(model, optimizer, train_loader, train_samples, batch_size,leave=bool(epoch==n_epochs-1))
        valid_loss = test(model, valid_loader, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('           Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model saved to:',model_save_to)
            torch.save(model.state_dict(),model_save_to)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break


    model.eval()
    t = tqdm(enumerate(test_loader),total=500/batch_size)
    y_test = []
    y_predict = []
    for i,data in t:
        data = data.to(device)    
        batch_output = model(data.x, data.edge_index, data.batch)    
        y_predict.append(batch_output.detach().cpu().numpy())
        y_test.append(data.y.cpu().numpy())
    y_test = np.concatenate(y_test)
    y_predict = np.concatenate(y_predict)


    # create ROC curves
    fpr_gnn, tpr_gnn, threshold_gnn = roc_curve(y_test[:,1], y_predict[:,1])
        
    # plot ROC curves
    plt.figure()
    plt.plot(tpr_gnn, fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.semilogy()
    plt.ylim(0.001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.savefig(fig_save_to)


if __name__=="__main__":
    main()