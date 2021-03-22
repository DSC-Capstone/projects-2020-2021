import networkx as nx
import numpy as np
import torch

def build_features(data, edges):
    nodes = list(data['id'])

    edge_list = edges.values.tolist()

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    A = nx.adjacency_matrix(G)

    labels = np.array(list(data['Rk']))

    X = data.drop(['Tm', 'Rk', 'id'], axis=1).to_numpy()

    features = torch.Tensor(X)

    adj = torch.Tensor(A.toarray())

    labels = torch.Tensor(labels)

    return features, adj, labels
