import stellargraph
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from scipy import sparse
import os
from build_features import create_struct
from eda import run_model
from feature_extraction import matrix_a
from feature_extraction import matrix_b
from feature_extraction import matrix_p
from tqdm import tqdm
import numpy as np
from scipy import sparse
import os
import gensim.models
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
#Derived From Gensim Source Code
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, corpus_path):
        self.lines = open(corpus_path).readlines()

    def __iter__(self):
        for line in tqdm(self.lines):
            yield line.strip().split(' ')

def api_nn_A(app):
        app_id = int(app.split('_')[1])
        neighbor_ids = np.nonzero(A_csr[app_id])[1]
        return np.array([f'api_{s}' for s in neighbor_ids])

def app_nn_A(api):
    assert api.startswith('api_')
    api_id = int(api.split('_')[1])
    neighbor_ids = np.nonzero(A_csc[:, api_id])[0]
    return np.array([f'app_{s}' for s in neighbor_ids])


######


 
with open('config/data-params.json') as fh:
        data_cfg = json.load(fh) 
        print('creating data structure...')
        data_dict = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[0]
        malware_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[1]
        benign_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[2]
        print('creating matrices...')
        a = matrix_a(data_dict)
        b = matrix_b(data_dict)
        p_m = matrix_p(data_dict) 
#Contants 

n=2
p=2
q=1
walk_length = 2
path = 'walk1'
A = a.tocsr()
P = p_m.tocsr()
A_csr = A
A_csc = A.tocsc(copy=True)


#Define Source and Target nodes
app_nodes = pd.DataFrame([], index=[f'app_{i}' for i in range(A.shape[0])])
api_nodes = pd.DataFrame([], index=[f'api_{i}' for i in range(A.shape[1])])

#Random Walk Functions
def random_walk(p=1, q=1, walk_length=50, app=None):
        path = []
        if app is None:
            app = 'app_' + str(np.random.choice(np.arange(A_csr.shape[0])))
        prev_nbrs = api_nn_A(app)
        curr_node = np.random.choice(prev_nbrs)
        prev_node = app
        path.append(app)
        path.append(curr_node)

        for i in range(walk_length - 2):
            if curr_node.startswith('api_'):
                neighbor_apis, neighbor_apps = all_nn_api(curr_node)
                curr_neighbors = np.concatenate([neighbor_apis, neighbor_apps])
            elif curr_node.startswith('app_'):
                curr_neighbors = api_nn_A(curr_node)

            alpha_1 = np.intersect1d(prev_nbrs, curr_neighbors, assume_unique=True)
            alpha_p = prev_node
            alpha_q = np.setdiff1d(
                np.setdiff1d(curr_neighbors, alpha_1, assume_unique=True),
                [alpha_p], assume_unique=True
            )
            alphas = [*alpha_1, alpha_p, *alpha_q]
            probs_q = np.full(len(alpha_q), 1/q/len(alpha_q)) if len(alpha_q) else []
            probs_1 = np.full(len(alpha_1), 1/len(alpha_1)) if len(alpha_1) else []
            probs = [*probs_1, 1/p, *probs_q]
            probs = np.array(probs) / sum(probs)

            new_node = np.random.choice(alphas, p=probs)
            path.append(new_node)
            prev_node = curr_node
            prev_nbrs = curr_neighbors
            curr_node = new_node

        return path

def walk(n, p, q, walk_length):
        print('in perform walks')

        num_apps = A_csr.shape[0]

        walks = []
        for app_i in tqdm(range(num_apps)):
            app = 'app_' + str(app_i)

            for j in range(n):
                try:
                    path = random_walk(p, q, walk_length, app=app)
                    walks.append(path)
                except:
                    continue
        return walks

def all_nn_api(api):
        api_id = int(api.split('_')[1])
        nbr_apis = np.concatenate([
            api_nn_P(api)
        ])
        neighbor_apis = np.unique(nbr_apis)
        neighbor_apps = app_nn_A(api)
        return neighbor_apis, neighbor_apps

def api_nn_P(api):
        api_id = int(api.split('_')[1])
        neighbor_ids = np.nonzero(P[:, api_id])[0]
        ls = [f'api_{s}' for s in neighbor_ids]
        ls.remove(api)
        return np.array(ls)
#Store Walks 
def runall(path, walk, n, p, q):
    walks = walk(n=n, p=p, q=q, walk_length=2)
    print('performed walks')
    outfile = open(path, 'w')
    for walk in tqdm(walks):
        outfile.write(' '.join(walk) + '\n')
    outfile.close()

    print('added walks to local dir')

    # Gensim Model + Walks
    sentences = MyCorpus(path)
    model = gensim.models.Word2Vec(sentences=sentences, size=64,
                               sg=1, negative=5, window=3, iter=5, min_count=1)
    node_ids = model.wv.index2word
    node_embeddings = (
        model.wv.vectors
    ) 
    print('Successfully Created Node Embeddings Using Node 2 Vec')
    #print(node_embeddings)
    #Evaluate - SVC 
    svm = SVC(kernel='rbf', C=5)
    #svm.fit(node_embeddings, data['EMBEDDING_TRAIN_Y'])
    #print(svm.score(node_embeddings, data['EMBEDDINGS_TEST_Y']))

def node2vec():
    runall(path, walk, n, p, q)


    
