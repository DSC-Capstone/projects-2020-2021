import numpy as np
import networkx as nx
import pandas as pd
import torch

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os
from collections import defaultdict 
from nltk.corpus import stopwords

class data_loader_sen():
    def __init__(self, features, tweets, edges_address, directed = False):
        features = pd.read_csv(features)
        edges = pd.read_csv(edges_address)
        tweets = pd.read_csv(tweets)

        #adjacency matrix
        adj = self.get_adj(edges, directed)    
        
        self.labels = np.array(features['79'])
        voting_feature = list(np.array(features.iloc[:, :features.shape[1]-1]))
        sen_feature = self.get_feature(tweets)

        for i in range(len(voting_feature)):
            sen_feature[i] += list(voting_feature[i])
        self.features = sen_feature
            
        
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
    def get_feature(self, tweets):
        #generate top 1000 popular words
        tweets = tweets.groupby("Senator Name").agg({"Tweets": sum})
        total_words = defaultdict(int)
        stopWords = set(stopwords.words('english'))
        total_tweet = tweets["Tweets"].values
        for i in total_tweet:
            row_words = i.split()
            for j in row_words:
                j = j.lower()
                if j not in stopWords:
                    total_words[j] += 1
        total_words = {key: val for key, val in sorted(total_words.items(), key = lambda ele: ele[1], reverse = True)}
        total_keys = total_words.keys()
        top_1000 = []
        count = 0
        for z in total_keys:
            top_1000.append(z)
            count += 1
            if count == 1000:
                break;
        id_1000 = dict(zip(top_1000, range(1000)))
    
        #create features
        feature = []
        for i in total_tweet:
            feat = [0]*1000
            row_words = i.split()
            for j in row_words:
                if j in top_1000:
                    feat[id_1000[j]] += 1
            feature.append(feat)
        return feature
    
    def get_data(self):
        return self.features, self.labels, self.A
