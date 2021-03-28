import json # load json labels 
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec # load word2vec model
from scipy import sparse
from scipy.sparse import load_npz # load sparse matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.manifold import TSNE


class clf:
    """
    classifier of embedding features 
    """
    def __init__(self, model_path, train_path, test_path, label_path, clf_lst, plot_path, vec_size):
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
        self.label_path = label_path
        self.clf_lst = clf_lst
        self.plot_path = plot_path
        self.vec_size = vec_size
        #self.model
        #self.train_mat
        #self.test_mat
        #self.api_features
        #self.X_train
        #self.X_test
        #self.y_train
        #self.y_test

        self._load()
        self._load_api_features()
        self._load_train_test()
        self._plot_tsne()
        self._clf_acc()


    def _load(self):
        self.model = Word2Vec.load(self.model_path)
        self.train_mat = load_npz(self.train_path) #loading A_train
        self.test_mat = load_npz(self.test_path) #loading A_test
        self.labels = json.load(open(self.label_path))

    def _load_api_features(self):
        api_count = self.train_mat.shape[1] # get the number of total apis 
        api_lst = []

        # load api vec from word2vec model 
        for i in range(api_count):
            if f'api{i}' in self.model.wv.vocab:
                api_lst.append(self.model.wv[f'api{i}'])
            # if api# doesn't appear in the model corpus, then append 0 
            else:
                api_lst.append(np.zeros(self.vec_size))
        self.api_features = np.array(api_lst)

    def _load_train_test(self):
    
        # creating X_train with shape(num_training_app, vec_size)
        self.api_features = sparse.csc_matrix(self.api_features)
        self.X_train = (self.train_mat.dot(self.api_features)) / self.train_mat.sum(axis=1).reshape(-1,1)
        self.X_train = self.X_train.tolist()
        #self.X_train = [X_train[i].tolist() for i in range(len(X_train))]

        # create X_test with shape(num_tesing_app, vec_size)
        self.X_test = (self.test_mat.dot(self.api_features)) / self.test_mat.sum(axis=1).reshape(-1,1)
        self.X_test = self.X_test.tolist()
        #self.X_test = [X_test[i].tolist() for i in range(len(X_test))]

        self.y_train = self.labels['y_train']
        self.y_test = self.labels['y_test']

    def _plot_tsne(self):
        n_row = len(self.X_train) + len(self.X_test)
        df_dict = {}
        df_dict['id'] = [f'app{i}' for i in range(n_row)]
        df_dict['vec'] = np.append(self.X_train, self.X_test, axis=0).tolist()

        # map the label types 
        labels_lst = np.append(np.where(np.array(self.y_train) == 1, 'train_malware', 'train_benign'),
                      np.where(np.array(self.y_test) == 1, 'test_malware', 'test_benign'))
        labels_dict = {k:v for k,v in zip(df_dict['id'], labels_lst)}

        df = pd.DataFrame.from_dict(df_dict)
        df = df[df.vec.apply(np.sum) != 0]
        df_dict['id'] = df.id.values.tolist()
        df_dict['vec'] = df.vec.values.tolist()

        # use tsne change the embedding to 2-dim 
        tsne = TSNE()
        X = tsne.fit_transform(df_dict['vec'])

        viz_df = pd.DataFrame()
        viz_df['id'] = df_dict['id']
        viz_df['vec1'] = X[:,0]
        viz_df['vec2'] = X[:,1]
        viz_df['y'] = [labels_dict[i] for i in viz_df['id']]

        # get the scatter plot 
        color_dict = {
            'train_malware': 'pink',
            'train_benign': 'lightblue', 
            'test_malware': 'crimson',
            'test_benign': 'steelblue'
        }
        sns.scatterplot(
            data=viz_df,
            x='vec1',
            y='vec2',
            hue='y',
            palette= color_dict,
        )

        plt.savefig(self.plot_path)

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
            model = SVC()
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

    def _clf_acc(self):
        clf_dict = {}
        # train clf
        for m in tqdm(self.clf_lst):
            model = self.choose_model(m)
            X_train = [self.X_train[i] for i in range(len(self.X_train))]
            X_test = [self.X_test[i] for i in range(len(self.X_test))]
            clf_dict[m] = model.fit(X_train, self.y_train, ) 

        # get test score 
        for m in tqdm(clf_dict):
            model = clf_dict[m]
            test_acc = model.score(X_test, self.y_test)
            train_acc = model.score(X_train, self.y_train)
            test_f1 = f1_score(self.y_test, model.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(self.y_test, model.predict(X_test)).ravel()

            print(f'Model: {m}    Train-Acc:{train_acc}    Test-Acc: {test_acc}     F1: {test_f1}     TP: {tp}      TN: {tn}     FN: {fn}     FP: {fp}')

def run_clf(model_path, train_path, test_path, label_path, clf_lst, plot_path, vec_size):
    """
    Run Classifier 
    """
    clf(model_path, train_path, test_path, label_path, clf_lst, plot_path, vec_size)




        

