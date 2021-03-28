import pandas as pd
import numpy as np
import json 
from tqdm import tqdm
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from src.util import *

class model():
    def __init__(self,X_paths,type_list,clf_lst,type_A,y_path,metapaths,output_path):
        self.df = pd.DataFrame()
        self.metapaths = metapaths
        self.output_path = output_path
        self.clf_lst = clf_lst
        self.type_A = type_A

        self._load_matrix(X_paths,type_list)
        self._load_y(y_path)
        self._construct_kernel(metapaths)
        self._save_data(self.metapaths,self.kernels)
        self._evaluate(self.metapaths,self.kernels)
        

    def _load_matrix(self,X_paths,type_list):
        # load_matrix and save them with their corresponding type of matrix
        for p,t in zip(X_paths,type_list):
            matrix = sparse.load_npz(p)
            if "A_train" in t:
                self.A_tr_mat = matrix
            elif "A_test" in t:
                self.A_test_mat = matrix 
            elif "B_train" in t:
                self.B_tr_mat = matrix 
            elif "P_train" in t:
                self.P_tr_mat = matrix
            elif "R_train" in t:
                self.R_tr_mat = matrix 
            elif "R_test" in t: 
                self.R_test_mat = matrix
            elif "I_train" in t:
                self.I_tr_mat = matrix
            else :
                raise NotImplementedError 
    
    def _load_y(self, y_path):
        # open the json file
        y_value = json.load(open(y_path))
        self.y_train = y_value['y_train']
        self.y_test = y_value['y_test']


    def _kernel_func(self, metapath):
        # store the matrix 
        if "A" in metapath:
            A_tr_mat_trans = self.A_tr_mat.T
        if "R" in metapath:
            R_tr_mat_trans = self.R_tr_mat.T
        if "B" in metapath: 
            B_tr_mat = self.B_tr_mat
        if "P" in metapath:
            P_tr_mat = self.P_tr_mat
        if "I" in metapath:
            I_tr_mat = self.I_tr_mat
        

        # return functions for metapath 
        if metapath == "AA":
            func = lambda X: X.dot(A_tr_mat_trans)
        elif metapath == "ABA":
            func = lambda X: (X.dot(B_tr_mat)).dot(A_tr_mat_trans)
        elif metapath == "APA":
            func = lambda X: (X.dot(P_tr_mat)).dot(A_tr_mat_trans)
        elif metapath == "ABPBA":
            func = lambda X: (((X.dot(B_tr_mat)).dot(P_tr_mat)).dot(B_tr_mat.T)).dot(A_tr_mat_trans)
        elif metapath == "APBPA":
            func = lambda X: (((X.dot(P_tr_mat)).dot(B_tr_mat)).dot(P_tr_mat.T)).dot(A_tr_mat_trans)
        elif metapath == "AIA":
            func = lambda X: (X.dot(I_tr_mat)).dot(A_tr_mat_trans)
        elif metapath == "ABPIPBA":
            func = lambda X: ((((((X.dot(B_tr_mat)).dot(P_tr_mat))).dot(I_tr_mat)).dot(P_tr_mat.T)).dot(B_tr_mat.T)).dot(A_tr_mat_trans)
        elif metapath == "ABIPIBA":
            func = lambda X: ((((((X.dot(B_tr_mat)).dot(I_tr_mat))).dot(P_tr_mat)).dot(I_tr_mat.T)).dot(B_tr_mat.T)).dot(A_tr_mat_trans)
        elif metapath == "RR":
            func = lambda X: X.dot(R_tr_mat_trans)
        elif metapath == "RBR":
            func = lambda X: (X.dot(B_tr_mat)).dot(R_tr_mat_trans)
        else:
            raise NotImplementedError
        return func 
    
    def _construct_kernel(self,metapaths):
        kernel_funcs = []
        for metapath in metapaths:
            kernel_funcs.append(self._kernel_func(metapath))
        self.kernels = kernel_funcs 
    
        
    def _save_data(self,metapaths,kernels):
        y_train = self.y_train
        y_test = self.y_test 
        if "A" in metapaths[0]:
            X_train = self.A_tr_mat
            X_test = self.A_test_mat
        else:
            X_train = self.R_tr_mat
            X_test = self.R_test_mat
        for mp, kernel in zip(metapaths, kernels):
            print(mp)
            gram_train = kernel(X_train).toarray()
            gram_test = kernel(X_test).toarray()
            pd.DataFrame(gram_train).to_csv(f'{self.output_path}/{mp}_{self.type_A}_train.csv', index=False)
            pd.DataFrame(gram_test).to_csv(f'{self.output_path}/{mp}_{self.type_A}_test.csv', index=False)
            print(f'{mp} data saved')

    def choose_model(self, model_name):
        if model_name == 'svm':
            # svm_pipe = Pipeline([
            # ('ct', StandardScaler()),
            # ('pca', PCA(svd_solver='full')),
            # ('svm', SVC())
            # ])
            # # Using cv to find the best hyperparameter
            # param_grid = {
            # 'svm__C': [0.1,1, 5, 10, 100],
            # 'svm__gamma': [1,0.1,0.01,0.05, 0.001],
            # 'svm__kernel': ['rbf', 'sigmoid'],
            # 'pca__n_components':[1, 0.99, 0.95, 0.9]
            # }
            # model = GridSearchCV(svm_pipe, param_grid, n_jobs=-1)
            model = SVC(kernel='precomputed')
        elif model_name == 'rf':
            # rf_pipe = Pipeline([
            # ('ct', StandardScaler()),
            # ('pca', PCA(svd_solver='full')),
            # ('rf', RandomForestClassifier())
            # ])
            # # Using cv to find the best hyperparameter
            # param_grid = {
            # 'rf__max_depth': [2, 4, 6, 8, None],
            # 'rf__n_estimators': [5, 10, 15, 20, 50, 100],
            # 'rf__min_samples_split': [3, 5, 7, 9],
            # 'pca__n_components':[1, 0.99, 0.95, 0.9]
            # }
            # model = GridSearchCV(rf_pipe, param_grid, n_jobs=-1)
            model = RandomForestClassifier()
        else:
            # gb_pipe = Pipeline([
            # ('ct', StandardScaler()),
            # ('pca', PCA(svd_solver='full')),
            # ('gb', GradientBoostingClassifier())
            # ])
            # # Using cv to find the best hyperparameter
            # param_grid = {
            #     'gb__loss': ['deviance', 'exponential'],
            #     'gb__n_estimators': [5, 10, 15, 20, 50, 100],
            #     'gb__max_depth': [2, 4, 6, 8],
            #     'gb__min_samples_split': [3, 5, 7, 9],
            #     'pca__n_components':[1, 0.99, 0.95, 0.9],
            # }
            # model = GridSearchCV(gb_pipe, param_grid, n_jobs=-1)
            model = GradientBoostingClassifier()
        return model 

    def _clf_acc(self,train,test):
        clf_dict = {}
        # train clf
        for m in self.clf_lst:
            model = self.choose_model(m)
            clf_dict[m] = model.fit(train, self.y_train) 

        # get test score 
        for m in clf_dict:
            model = clf_dict[m]
            test_acc = model.score(test, self.y_test)
            train_acc = model.score(train, self.y_train)
            test_f1 = f1_score(self.y_test, model.predict(test))
            tn, fp, fn, tp = confusion_matrix(self.y_test, model.predict(test)).ravel()
            print(f'Model: {m}    Train-Acc:{train_acc}    Test-Acc: {test_acc}     F1: {test_f1}     TP: {tp}      TN: {tn}     FN: {fn}     FP:  {fp}')

    def _evaluate(self,metapaths,kernels):
        print(self.type_A)
        if "A" in metapaths[0]:
            X_train = self.A_tr_mat
            X_test = self.A_test_mat
        else:
            X_train = self.R_tr_mat
            X_test = self.R_test_mat
        for mp, kernel in zip(metapaths, kernels):
            print(mp)
            gram_train = pd.read_csv(f'{self.output_path}/{mp}_{self.type_A}_train.csv').values
            gram_test = pd.read_csv(f'{self.output_path}/{mp}_{self.type_A}_test.csv').values
            self._clf_acc(gram_train,gram_test)
        print("DONE")

    

def run_model(X_paths,type_list,clf_lst,type_A,y_path,metapaths,output_path):
    """
    Run model 
    """
    model(X_paths,type_list,clf_lst,type_A,y_path,metapaths,output_path)
    print("Result saved")