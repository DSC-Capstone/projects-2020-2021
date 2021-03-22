import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy.random import choice
from torch.nn.parameter import Parameter
import math
from sklearn import preprocessing
import numpy as np
from torch import optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
#The whole functions are from DSC180A Group02 which is contributed by Xinrui Zhan, Yimei Zhao and Shang Li
def aggregate(A, X, len_walk, num_neigh, agg_func):
    norm = torch.div(A, torch.sum(A, axis=1))
    norm = torch.matrix_power(norm, len_walk)
    result = torch.zeros(X.shape)
    for i in range(A.shape[0]):
        x = A[i].cpu().detach().numpy()
        ind = np.random.choice(range(x.shape[0]), num_neigh, replace=False)
        if agg_func == "MEAN":
            result[i] = torch.mean(X[ind], axis=0)
        else:
            result[i] = torch.mean(X[ind], axis=0).values
    return result

class SageLayer(nn.Module):
    def __init__(self, F, O, len_walk = 2, num_neigh = 10, agg_func="MEAN", bias=True): 
        super(SageLayer, self).__init__()
        self.F = F
        self.O = O
        self.weight = Parameter(torch.FloatTensor(2 * F, O))
        self.len_walk = len_walk
        self.num_neigh = num_neigh
        self.agg_func = agg_func
        if bias:
            self.bias = Parameter(torch.FloatTensor(O))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, X, A):
        aggregated = aggregate(A, X, self.len_walk, self.num_neigh, self.agg_func)
        aggregated = aggregated.to(X.device)
        combined = torch.cat([X, aggregated], dim=1)
        combined = torch.mm(combined, self.weight)
        if self.bias is not None:
            return combined + self.bias
        else:
            return combined
        return combined
    
class GraphSage_model(nn.Module):
    def __init__(self, X, A, n=0, F=79, class_num=7, 
                 agg_func='MEAN', hidden_neuron=200, len_walk=2, num_neigh=10, bias=True):
        super(GraphSage_model, self).__init__()

        self.F = F
        self.class_num = class_num
        self.n = n
        self.agg_func = agg_func
        self.X = X
        self.A = A
        
        self.gs1 = SageLayer(F, hidden_neuron, len_walk=len_walk, num_neigh=num_neigh, agg_func=agg_func, bias=bias)
        self.gsh = SageLayer(hidden_neuron, hidden_neuron, len_walk=len_walk, num_neigh=num_neigh, agg_func=agg_func, bias=bias)
        self.gs2 = SageLayer(hidden_neuron, self.class_num, len_walk=len_walk, num_neigh=num_neigh, agg_func=agg_func, bias=bias)
        
    def forward(self, X):
        X = self.gs1(X, self.A)
        X = F.relu(X)
        for i in range(self.n):
            X = self.gsh(X, self.A)
            X = F.relu(X)
        X = self.gs2(X, self.A)
        return F.log_softmax(X, dim=1)

    
class GraphSage():
    def __init__(self, A, X, y, device='cuda', n=0, F=79, class_num=7, agg_func="MEAN", hidden_neuron=200,
                len_walk=2, num_neigh=10, bias=True, val_size=0.3):
        if device == 'cuda':
            self.device= torch.device('cuda')
        else:
            assert('only support cuda')
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        
        X = torch.tensor(X)
        X = X.type(torch.float)
        y = torch.tensor(y)
        y = y.type(torch.long)
        y = y.to(self.device)
        A = torch.from_numpy(A).float()
        
        self.X = X.to(self.device)
        self.A = A.to(self.device)
        self.y = y.to(self.device)
        
        train_idx = np.random.choice(self.X.shape[0], round(self.X.shape[0]*(1-val_size)), replace=False)
        val_idx = np.array([x for x in range(X.shape[0]) if x not in train_idx])
        print("Train length :{a}, Validation length :{b}".format(a=len(train_idx), b=len(val_idx)))
        
        self.idx_train = torch.LongTensor(train_idx)
        self.idx_val = torch.LongTensor(val_idx)
        self.graphsage = GraphSage_model(self.X, self.A, n=n, F=F, agg_func=agg_func, hidden_neuron=hidden_neuron,
                                         class_num = class_num, len_walk=len_walk, bias=bias, num_neigh=num_neigh)
        self.graphsage.to(self.device)
        
    def train(self, optimizer, epoch):
        self.graphsage.train()
        optimizer.zero_grad()
        output = self.graphsage(self.X)
        loss = F.cross_entropy(output[self.idx_train], self.y[self.idx_train])
        loss.backward(retain_graph=True)
        optimizer.step()
        print('Epoch: {x}'.format(x=epoch))
        print('training loss {:.4f}'.format(loss.item()))
            
    def test(self):
        self.graphsage.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = self.graphsage(self.X)
            test_loss = F.cross_entropy(output[self.idx_val], self.y[self.idx_val], reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)[self.idx_val]
            correct += pred.eq(self.y[self.idx_val].view_as(pred)).sum().item()

        test_loss /= len(self.idx_val)
        print('Validtion: Average loss: {:.4f}, Accuracy: {:.4f}%'.format(test_loss, 100. * correct / len(self.idx_val)))
        return 100. * correct / len(self.idx_val)
    
    def train_epoch(self, epochs=50, lr=1e-3):
        acc = []
        optimizer = optim.Adam(self.graphsage.parameters(), lr=lr)#, weight_decay=1e-1)

        for epoch in range(epochs):
            self.train(optimizer, epoch)
            accs = self.test()
            acc.append(accs)
        accs = {'acc': acc}
        return accs
    
    def visualization(self):
        self.model.eval()
        output = self.model(self.X)
        pred = output.argmax(dim=1, keepdim=True)
        A = self.A.cpu().detach().numpy()
        Y = pred.cpu().detach().numpy()
        G = nx.from_numpy_matrix(A, nx.DiGraph())
        pos = nx.spring_layout(G, seed=675)
        nx.draw(G, pos=pos, node_size=10,node_color=Y, width = 0.1)
        plt.show()
