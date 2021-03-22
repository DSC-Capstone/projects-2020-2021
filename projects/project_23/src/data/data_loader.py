import numpy as np
import networkx as nx
import pandas as pd
import torch

class data_loader():
    def __init__(self, feature_address, edges_address, directed = False):
        features = pd.read_csv(feature_address)
        edges = pd.read_csv(edges_address)

        #adjacency matrix
        adj = self.get_adj(edges, directed)    
        self.labels = np.array(features['79']) 
        self.features = np.array(features.iloc[:, :features.shape[1]-1])
        # self.features = torch.FloatTensor(self.features)

        self.A = adj
    def get_adj(self, edges, directed):
        rows, cols = edges["followed"], edges["following"]
    
        nodes = list(set(rows).union(set(cols)))
        n_nodes = len(nodes)
    
        node_index = {}
        for i in np.arange(len(nodes)):
            node_index[nodes[i]] = i
            i += 1
    
        adj = np.zeros((n_nodes, n_nodes), dtype='int64')

        for i in range(len(edges)):
            adj[node_index[rows[i]], node_index[cols[i]]]  = 1.0
            if not directed: 
                adj[node_index[cols[i]], node_index[rows[i]]]  = 1.0 
            
        return adj
    
    def get_data(self):
        return self.features, self.labels, self.A
