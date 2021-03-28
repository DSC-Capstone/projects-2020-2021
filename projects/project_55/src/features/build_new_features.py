import os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csc_matrix # compressed sparse column matrix 
from scipy.sparse import csr_matrix # compressed sparse row matrix
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm

from src.util import *

# disable a warning
pd.options.mode.chained_assignment = None


np.random.seed(10)

class preprocess_csv:
    """
    Preprocess csv file for construction of new matrix 
    """
    def __init__(self, paths, sample_size, type_lst, output_path, train_test_split=0.8,):
        self.output_path = output_path
        self.df = pd.DataFrame()
        self._load_csv(paths, sample_size, type_lst)
        self._reconstruct_ids()
        self._train_test_split(train_test_split)
        self._train_test_y()
        self._save_x()
        self._save_y()
        #self.train
        #self.test 
        #self.y_train
        #self.y_test 
        
    
    def _load_csv(self, paths, sample_size, type_lst):
        
        # load_csv files and save as pd.DataFrame 
        # also save types for each app for later use 
        df_lst = []
        for ind, p in enumerate(paths):
            
            # load # of apps according to the sample_size 
            app_size = sample_size[ind]
            
            # read .csv
            p_df = pd.read_csv(p).iloc[:, 1:]
            
            # random select apps 
            app_lst = np.random.choice(p_df.app_id.unique(), app_size, replace=False)
            
            # only load random selected apps 
            p_df = p_df[p_df.app_id.isin(app_lst)]
            
            # set type
            p_df['type'] = type_lst[ind]
            df_lst.append(p_df)

        self.df = pd.concat(df_lst, ignore_index=True)
        
    def _reconstruct_ids(self):
        
        # api_id: uniquely combine the api_id and package_id 
        self.df['api_package_id'] = cantor_pairing(
            np.vstack((self.df.api_id.values, self.df.package_id.values))
        )
        
        # block_id: combine block_id, class_id, and app_id 
        self.df['block_id'] = cantor_pairing(
            np.vstack((self.df.block_id.values, self.df.class_id.values, self.df.app_id.values))
        )

        self.df['api_package_id'] = reset_id(self.df['api_package_id'])
        self.df['block_id'] = reset_id(self.df['block_id'])
        
    def _train_test_split(self, train_test_split):
        
        # calculate train_size 
        train_size = int(self.df.app_id.nunique() * train_test_split)
        
        # select app list randomly 
        app_lst = np.random.choice(self.df.app_id.unique(), size=train_size, replace=False)
        
        # design train and test dataframe 
        self.train = self.df[self.df.app_id.isin(app_lst)]
        self.test = self.df[~self.df.app_id.isin(app_lst)]
        
    def _train_test_y(self):
        
        # generate y_train and y_test 
        self.y_train = list(self.train.groupby('app_id').agg({'type':np.unique}).to_dict()['type'].values())
        self.y_test = list(self.test.groupby('app_id').agg({'type':np.unique}).to_dict()['type'].values())
    
    def _save_x(self):
        self.train.to_csv(f'{self.output_path}/preprocess_train.csv', index=False)
        self.test.to_csv(f'{self.output_path}/preprocess_test.csv', index=False)
        
    def _save_y(self):
        json.dump({'y_train': self.y_train, 'y_test':self.y_test}, open(f"{self.output_path}/label.json", 'w' ))

class matA():
    """
    Generate matrix A 
    """
    def __init__(self, output_path, pack=True, api=True):
        self.output = output_path
        self.pack=pack
        self.api=api
        
        # check which reduce method is used 
        if self.pack and self.api:
            self.apply_name = 'api_package_id'
        elif self.pack: 
            self.apply_name = 'package_id'
        else:
            self.apply_name = 'api_id'
            
        # Load training and testing df 
        usecols = ['app_id', 'block_id', self.apply_name]
        self.train = pd.read_csv(f'{output_path}/preprocess_train.csv', usecols=usecols)
        self.test = pd.read_csv(f'{output_path}/preprocess_test.csv', usecols=usecols)
        
        # load train/test y labels 
        self.y_train = json.load(open(f'{output_path}label.json'))['y_train']
        self.y_test = json.load(open(f'{output_path}label.json'))['y_train']
        self._reset_id()
        self._construct_csr_train_mat()
        self._construct_csr_test_mat()
        self._save_train_test()
        #self.df 
        #self.X_train
        #self.X_test
        
    def _reset_id(self):
        
        # reset app id
        self.train.loc[:,'app_id'] = reset_id(self.train['app_id'])
        self.train.loc[:, 'block_id'] = reset_id(self.train['block_id'])

            
        # delete test api row if test api not in train
        self.test = self.test[self.test[self.apply_name].isin(self.train[self.apply_name])]
        self.test.loc[:,'app_id'] = reset_id(self.test['app_id'])

        # reset id in train api_id 
        unique_arr = np.unique(self.train[self.apply_name])
        api_id_map = {val:ind for ind, val in enumerate(unique_arr)}

        # map both train and test api_id
        self.train.loc[:, 'apply_id'] = self.train[self.apply_name].map(api_id_map)
        self.test.loc[:, 'apply_id'] = self.test[self.apply_name].map(api_id_map)
            
        
    def _construct_csr_train_mat(self):
        row = self.train.app_id.values
        col = self.train.apply_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(col)
        self.X_train = csr_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique)))
        self.X_train = self.X_train.astype(bool).astype(int)

        
    def _construct_csr_test_mat(self):
        row = self.test.app_id.values
        col = self.test.apply_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(self.train.apply_id.values)
        self.X_test = csr_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique)))
        self.X_test = self.X_test.astype(bool).astype(int)
        
        
    def _save_train_test(self): 
        if self.pack and self.api:
            save_npz(f'{self.output}/A_train.npz', self.X_train)
            save_npz(f'{self.output}/A_test.npz', self.X_test)
        elif self.pack:
            save_npz(f'{self.output}/A_train_reduced_pack.npz', self.X_train)
            save_npz(f'{self.output}/A_test_reduced_pack.npz', self.X_test)
        else:
            save_npz(f'{self.output}/A_train_reduced_api_name.npz', self.X_train)
            save_npz(f'{self.output}/A_test_reduced_api_name.npz', self.X_test)

class matR():
    """
    Generate matrix R 
    """
    def __init__(self, output_path):
        self.output = output_path
        
        # Load training and testing df 
        usecols = ['app_id', 'block_id', 'return_id']
        self.train = pd.read_csv(f'{output_path}/preprocess_train.csv', usecols=usecols)
        self.test = pd.read_csv(f'{output_path}/preprocess_test.csv', usecols=usecols)
        
        # load train/test y labels 
        self.y_train = json.load(open(f'{output_path}label.json'))['y_train']
        self.y_test = json.load(open(f'{output_path}label.json'))['y_train']
        self._reset_id()
        self._construct_csr_train_mat()
        self._construct_csr_test_mat()
        self._save_train_test()
        #self.df 
        #self.X_train
        #self.X_test
        
    def _reset_id(self):
        
        # reset app id
        self.train.loc[:,'app_id'] = reset_id(self.train['app_id'])
        self.train.loc[:, 'block_id'] = reset_id(self.train['block_id'])

            
        # delete test api row if test api not in train
        self.test = self.test[self.test['return_id'].isin(self.train['return_id'])]
        self.test.loc[:,'app_id'] = reset_id(self.test['app_id'])

        # reset id in train api_id 
        unique_arr = np.unique(self.train['return_id'])
        api_id_map = {val:ind for ind, val in enumerate(unique_arr)}

        # map both train and test api_id
        self.train.loc[:, 'return_id'] = self.train['return_id'].map(api_id_map)
        self.test.loc[:, 'return_id'] = self.test['return_id'].map(api_id_map)
            
        
    def _construct_csr_train_mat(self):
        row = self.train.app_id.values
        col = self.train.return_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(col)
        self.X_train = csr_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique)))
        self.X_train = self.X_train.astype(bool).astype(int)

        
    def _construct_csr_test_mat(self):
        row = self.test.app_id.values
        col = self.test.return_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(self.train.return_id.values)
        self.X_test = csr_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique)))
        self.X_test = self.X_test.astype(bool).astype(int)
        
        
    def _save_train_test(self): 
        save_npz(f'{self.output}/R_train.npz', self.X_train)
        save_npz(f'{self.output}/R_test.npz', self.X_test)
        
class matB():
    """
    Generate matrix B 
    """
    def __init__(self, A):
        self.train = A.train.copy()
        self.output = A.output
        self.pack = A.pack
        self.api = A.api
        #self.X_train
        self._construct_csr_train_mat()
        self._save_mat()
        
        
    def _construct_csr_train_mat(self):
        row = self.train.apply_id.values
        col = self.train.block_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(col)
        preB = csc_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique))).astype(bool).astype(int)
        self.X_train = preB.dot(preB.T)
        self.X_train = self.X_train.astype(bool).astype(int)
        
    def _save_mat(self): 
        if self.pack and self.api:
            save_npz(f'{self.output}/B_train.npz', self.X_train)
        elif self.pack:
            save_npz(f'{self.output}/B_train_reduced_pack.npz', self.X_train)
        else:
            save_npz(f'{self.output}/B_train_reduced_api_name.npz', self.X_train)

class matRB():
    """
    Generate matrix B 
    """
    def __init__(self, R):
        self.train = R.train.copy()
        self.output = R.output
        #self.X_train
        self._construct_csr_train_mat()
        self._save_mat()
        
        
    def _construct_csr_train_mat(self):
        row = self.train.return_id.values
        col = self.train.block_id.values
        data = np.ones((len(row), ))
        row_unique = np.unique(row)
        col_unique = np.unique(col)
        preRB = csc_matrix((data, (row, col)), shape=(len(row_unique), len(col_unique))).astype(bool).astype(int)
        self.X_train = preRB.dot(preRB.T)
        self.X_train = self.X_train.astype(bool).astype(int)
        
    def _save_mat(self): 
        save_npz(f'{self.output}/B_train_R.npz', self.X_train)

def build_mat(paths, sample_size, type_lst, output_path, matlst):
    """
    Build matrixes 
    """
    _ = preprocess_csv(
        paths, sample_size, type_lst, output_path
    )
    print('Preprocess Finished')
    
    if 'A' in matlst:
        A = matA(output_path, pack=True, api=True)
        A_pack = matA(output_path, pack=True, api=False)
        A_api = matA(output_path, pack=False, api=True)      
        print('A Finished')

    if 'R' in matlst:
        R = matR(output_path)
        print('R Finished')

    if 'RB' in matlst:
        matRB(R)
        print('RB Finished')

    if 'B' in matlst:
        matB(A)
        matB(A_pack)
        matB(A_api)
        print('B Finsihed')
