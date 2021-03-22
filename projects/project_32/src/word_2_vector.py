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
import json
from sklearn.ensemble import RandomForestRegressor

#Derived From Gensim Source Code
class MyCorpus(object):
    def __init__(self, corpus_path):
        self.lines = open(corpus_path).readlines()

    def __iter__(self):
        for line in tqdm(self.lines):
            yield line.strip().split(' ')



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


# Use APA order of traversal to find corresponding embeddings
def APA(length):
    while True:
        default = np.random.choice(np.arange(A.shape[0]))

        walk = f'app_{default}'

        for i in range(length):

            try:
                # Traverse through the left 
                i = np.random.choice(np.nonzero(A[default])[1])
                
                # Traverse through the left
                p = np.random.choice(np.nonzero(P[:, i])[0])
                
                #Find the corresponding right
                default = np.random.choice(np.nonzero(A_csc[:, p])[0])

                walk += f' api_{i} api_{p} app_{default}'
            except:
                continue

        yield walk

def word2vec():
    print('Traversing Graph with Random Walks')
    f = open(path, 'w')
    traverse = APA(length=walk_length)
    for _ in tqdm(range(100)):
        f.write(next(traverse) + '\n')
    f.close()
    
    sentences = MyCorpus(path)
    model = gensim.models.Word2Vec(sentences=sentences, size=256, sg=1, negative=5, window=7, iter=3)
    print('Creating Vector Embeddings Using Word 2 Vec')
    embeddings = model.wv.vectors
    print('Complete!')

    #test embeddings train test split - control check 
    #test embeddings overall 
    X = [1,1]
    Y = [1,1]

    regressor = DecisionTreeRegressor(max_depth=None).fit(X, Y)
