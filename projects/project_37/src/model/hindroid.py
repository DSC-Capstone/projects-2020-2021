from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report

import os, json, pickle, csv
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from p_tqdm import p_map, p_imap
from scipy import sparse
from itertools import combinations, product
from functools import partial
from sparse_dot_mkl import dot_product_mkl
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from cupyx.scipy.sparse import vstack as gpu_vstack
import time

class Hindroid():
    def __init__(self, source_folder, name=None):
        # load matrices
        self.source_folder = source_folder
        self.name = name if name is not None else os.path.basename(source_folder.rstrip(os.path.sep))
        self.A = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'A_mat.npz')).astype('int32')
        self.B = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'B_mat.npz')).astype('i1').tocsr()
        self.P = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'P_mat.npz')).astype('i1').tocsr()
        
#         self.make_prefit_matrices()
        
        # load models
        with open(os.path.join(source_folder, 'hindroid', 'AAT.mdl'), 'rb') as file:
            self.AAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'ABAT.mdl'), 'rb') as file:
            self.ABAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'APAT.mdl'), 'rb') as file:
            self.APAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'ABPBTAT.mdl'), 'rb') as file:
            self.ABPBTAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'APBPTAT.mdl'), 'rb') as file:
            self.APBPTAT = pickle.load(file)
            
        self.app_map = pd.read_csv(os.path.join(source_folder, 'app_map.csv'), index_col='app', squeeze=True)
        self.api_map = pd.read_csv(os.path.join(source_folder, 'api_map.csv'), index_col='api', squeeze=True)
        self.api_map = self.api_map.str.replace('api', '').astype(int)
    
    def make_prefit_matrices(self):
        formula_map = {
            'AAT': [self.A.T],
            'ABAT': [self.B, self.A.T],
            'APAT': [self.P, self.A.T],
            'ABPBTAT': [self.B, self.P, self.B, self.A.T],
            'APBPTAT': [self.P, self.B, self.P, self.A.T],
        }
        
        for mp, mat_list in formula_map.items():
            
            outpath = os.path.join(self.source_folder, 'hindroid', f'{mp}_prefit.npz')
            if os.path.exists(outpath):  # skip if already present
                print(f'{mp}_prefit.npz already exists.')
                continue
            mat = [None]*mat_list[-1].shape[0]
            pbar = trange(mat_list[-1].shape[1])
            pbar.set_description(f'Computing {mp} prefix')
            for i in pbar:
                formula_idx = -1
                mat[i] = mat_list[formula_idx][:,i]
                while formula_idx > -len(mat_list):
                    formula_idx -= 1
                    mat_list[formula_idx]
                    mat[i] = mat_list[formula_idx].dot(mat[i])
            
            sparse.save_npz(outpath, sparse.hstack(mat))
    
    def load_prefit_matrix(self, metapath):
        return sparse.load_npz(os.path.join(self.source_folder, f'{metapath}_prefit.npz'))
    
    def fit_predict(self, path):
        '''
        Predicts all apps listed in the folder defined by `path` in `app_list.csv`.
        
        Outputs predictions to a csv in 
        '''
        outpath = os.path.join(path, f'hindroid-{self.name}')
        os.makedirs(outpath, exist_ok=True)
        # get app data, compute unique apis
        apps = pd.read_csv(os.path.join(path, 'app_list.csv'), usecols=['app'], squeeze=True, dtype=str)
        app_data_list = (
            os.path.join('data', 'out', 'all-apps', 'app-data/') +
            apps
            + '.csv'
        )
        data = dd.read_csv(list(app_data_list), dtype=str, usecols=['app', 'api'])
        data['api'] = data['api'].map(self.api_map)
        data = data.dropna()
        data['api'] = data.api.astype(int)
        print('Computing unique APIs per app')
        apis_by_app = data.groupby('app').api.unique().compute()
        
        def make_feature(api_list):
            app_features = np.zeros(self.api_map.size)
            app_features[list(api_list)] = 1
            return app_features
        
        features = sparse.lil_matrix((len(apps), len(self.api_map)), dtype='float32')
        row_idx = 0
        pbar = tqdm(apis_by_app)
        pbar.set_description('Building A-test matrix')
        for api_list in pbar:
            features[row_idx, api_list] = 1
            row_idx += 1
        features = features.tocsr()
        sparse.save_npz(os.path.join(outpath, 'A_test.npz'), features)
        
        print("Making predictions")
        time.sleep(1)
        results = self.batch_predict(features)
        results.index = apis_by_app.index
        
        true_labels = pd.read_csv('data/out/all-apps/app_list.csv', usecols=['app', 'malware'], index_col='app', squeeze=True)
        true_labels = true_labels[apps]
        
        for col, pred in results.iteritems():
            print(f'{col}:')
            print(classification_report(true_labels, pred))
        results['true'] = true_labels
        results.to_csv(os.path.join(outpath, f'predictions.csv'))
        
        
        return results
        
    
    def predict(self, x):
        '''
        Predict feature vector(s) of apps with all available kernels.
        
        Parameters:
        -----------------
        x: np.array, feature vectors with width same as the A matrix width, number of unique apis, or self.A.shape[0].
        
        Returns:
        A series with all predictions indexed by their metapath. 
        '''
        metapath_map = {
            'AAT': 'dot_product_mkl(x, self.A.T)',
            'ABAT': 'dot_product_mkl(dot_product_mkl(x, self.B), self.A.T)',
            'APAT': 'dot_product_mkl(dot_product_mkl(x, self.P), self.A.T)',
            'ABPBTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.B), self.P), self.B.T), self.A.T)',
            'APBPTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.P), self.B), self.P.T), self.A.T)',
        }
        
        predictions = {}
        for metapath, formula in metapath_map.items():
            features = eval(formula)
            pred = eval(f'self.{metapath}.predict(features)')
            predictions[metapath]= pred
        
        return pd.Series(predictions)
    
    def predict_with_kernel(self, x, kernel):
        '''
        Predict a feature vector(s) of apps with a specified kernel.
        
        Parameters:
        -----------------
        x: np.array, vector with size the same as the A matrix width, number of unique apis, or self.A.shape[0].
        kernel: str, A member of {'AAT', 'ABAT', 'APAT', 'ABPBTAT', 'APBPTAT'}
        '''
        formula_map = {
            'AAT': 'x * self.A.T',
            'ABAT': 'x * self.B * self.A.T',
            'APAT': 'x * self.P * self.A.T',
            'ABPBTAT': 'x * self.B * self.P * self.B * self.A.T',
            'APBPTAT': 'x * self.P * self.B * self.P * self.A.T',
        }
        
        x = x.reshape(-1, x.size)
        features = eval(formula_map[kernel])
        prediction = eval(f'self.{kernel}.predict(features)')
        
        return prediction[0]
    
    def batch_predict(self, X, outfolder=None):
        '''
        Predict a batch of feature vectors of apps with all available kernels.
        
        Parameters:
        -----------------
        X: np.array, vector with size the same as the A matrix width, number of unique apis, or self.A.shape[0].
        
        Returns: DataFrame with predictions using Apps as rows and kernels as columns with the true labels appended.  
        '''
        formula_map = {
            'AAT': 'X_ * self.A.T',
            'ABAT': 'X_ * self.B * self.A.T',
            'APAT': 'X_ * self.P * self.A.T',
            'ABPBTAT': 'X_ * self.B * self.P * self.B * self.A.T',
            'APBPTAT': 'X_ * self.P * self.B * self.P * self.A.T',
        }
        
        results = pd.DataFrame(
            columns=['AAT', 'ABAT', 'APAT', 'ABPBTAT', 'APBPTAT'],
        )
                
        # predict by model
        for col in results.columns:
            if outfolder is not None and os.path.exists(os.path.join(outfolder, f'{col}.npz')):
                print(f'{col} already present.')
                continue
            print(f'')
            batch_size = 100
            fit_features = []
            pbar = tqdm(range(0, X.shape[0], batch_size))
            pbar.set_description(f"Predicting {col}, batch")
            for i in pbar:
                X_ = X[i:i+batch_size]
                fit_features.append(eval(formula_map[col]))
            fit_features = sparse.vstack(fit_features)
            if outfolder is not None:
                sparse.save_npz(os.path.join(outfolder, f'{col}.npz'), fit_features)
            
            preds = eval(f'self.{col}.predict(fit_features.todense())')
            results[col] = preds
        
        return results
        
def gpu_dot(A, B):
    '''
    Performs dot product of A and B on gpu row-by-row to avoid OOM errors.
    '''
    result = []
    for i in tqdm(range(A.shape[0])):
        result.append(A[i].dot(B))
    return gpu_vstack(result)
        
        
        
        
        