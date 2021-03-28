from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score

import os
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
from p_tqdm import p_umap, p_imap
from scipy import sparse
from itertools import combinations, product
from functools import partial
import csv
from sparse_dot_mkl import dot_product_mkl

def build_from_folder(outfolder, redo=False):
    if not os.path.exists(os.path.join(outfolder, 'hindroid', 'P_mat.npz')) or redo:
        build_matrices(outfolder, redo=redo)
    if not os.path.exists(os.path.join(outfolder, 'hindroid', 'APBPTAT.mdl')) or redo:
        make_models(outfolder, redo=redo)
    return

def edge_prep(edges_path):
#     print(f"Dask Cluster: {client.cluster}")
#     print(f"Dashboard port: {client.scheduler_info()['services']['dashboard']}")

    edges = dd.read_csv(edges_path, dtype=str).compute()
    edges['target'] = edges.target.str.replace('api', '').astype(int)


    # A matrix prep
    app_api_edges = edges[edges.source.str.startswith('app')]
    app_api_edges['source'] = app_api_edges.source.str.replace('app', '').astype(int)
    app_api_edges.groupby('source').target.unique().sort_index().to_pickle('data/temp/app_api_sets.pkl')
    del app_api_edges

    # B matrix prep
    api_method_edges = edges[edges.source.str.startswith('method')]
    api_method_edges = api_method_edges.groupby('source').target.unique()
    api_method_edges[api_method_edges.apply(len)>1].to_pickle('data/temp/method_api_sets.pkl')
    del api_method_edges

    # P matrix prep
    api_package_edges = edges[edges.source.str.startswith('package')]
    api_package_edges = api_package_edges.groupby('source').target.unique()
    api_package_edges[api_package_edges.apply(len)>1].to_pickle('data/temp/package_api_sets.pkl')
    del api_package_edges
    del edges

def build_matrices(outfolder, redo):
    os.makedirs(os.path.join(outfolder, 'hindroid'), exist_ok=True)

    edges_path = os.path.join(outfolder, 'edges.csv')
    app_map_path = os.path.join(outfolder, 'app_map.csv')
    api_map_path = os.path.join(outfolder, 'api_map.csv')

    apis = pd.read_csv(api_map_path, index_col='api').uid.str.replace('api', '').astype(int).values
    apps = pd.read_csv(app_map_path, index_col='app').uid.str.replace('app', '').astype(int).values
    
    num_apps = apps.size
    num_apis = apis.size
    
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit([(api,) for api in apis])
        
#     with Client() as client:
    edge_prep(edges_path)

    # A matrix
    print("Constructing A matrix...")

    if not os.path.exists(os.path.join(outfolder, 'hindroid', 'A_mat.npz')) or redo:
        A_mat = mlb.transform(pd.read_pickle('data/temp/app_api_sets.pkl'))
        f'Constructed: {repr(A_mat)}'
        sparse.save_npz(os.path.join(outfolder, 'hindroid', 'A_mat.npz'), A_mat)
    else:
        print("Already present.")

    # B Matrix
    print("Constructing B matrix...")
    if not os.path.exists(os.path.join(outfolder, 'hindroid', 'B_mat.npz')) or redo:
        B_mat = build_BP_mat(pd.read_pickle('data/temp/method_api_sets.pkl'), num_apis)
        print(f'Constructed: {repr(B_mat)}')
        sparse.save_npz(os.path.join(outfolder, 'hindroid', 'B_mat.npz'), B_mat.astype('int'))
    else:
        print("Already present.")
    
    # P Matrix
    print("Constructing P matrix...")
    if not os.path.exists(os.path.join(outfolder, 'hindroid', 'P_mat.npz')) or redo:
        P_mat = build_BP_mat(pd.read_pickle('data/temp/package_api_sets.pkl'), num_apis)
        print(f'Constructed: {repr(P_mat)}')
        sparse.save_npz(os.path.join(outfolder, 'hindroid', 'P_mat.npz'), P_mat.astype('int'))
    else:
        print("Already present.")

def build_BP_mat(api_sets, num_apis):
    comb_func = lambda api_list: np.array(list(combinations(api_list, r=2)))
    row = []
    col = []
    for combos in p_imap(comb_func, api_sets):
        row.extend(combos[:,0])
        col.extend(combos[:,1])
    mat = sparse.csr_matrix(([True]*len(row), (row, col)), shape=(num_apis, num_apis), dtype=bool)
    del row, col
    mat.setdiag(True)
    mat += mat.T
    return mat

def make_models(source_folder, redo):
    print('Fitting models:')
    apps = load_apps(source_folder)
    
    source_folder = os.path.join(source_folder, 'hindroid')
    metapath_map = {
        'AAT': 'A_ * A.T',
        'ABAT': 'A_ * B * A.T',
        'APAT': 'A_ * P * A.T',
        'ABPBTAT': 'A_ * B * P * B * A.T',
        'APBPTAT': 'A_ * P * B * P * A.T',
    }
    
    A = sparse.load_npz(os.path.join(source_folder, 'A_mat.npz')).astype('float32')
    B = sparse.load_npz(os.path.join(source_folder, 'B_mat.npz')).astype('float32').tocsr()
    P = sparse.load_npz(os.path.join(source_folder, 'P_mat.npz')).astype('float32').tocsr()
    
    metrics = pd.DataFrame(columns = ['kernel', 'acc', 'recall', 'f1'])
    
    for metapath, formula in metapath_map.items():
        print(f'\tFitting {metapath} model...')
        commuting_matrix = []
        batch_size = 100
        for i in tqdm(range(0, A.shape[0], batch_size)):
            A_ = A[i:i+batch_size]
            commuting_matrix.append(eval(formula))
        commuting_matrix = sparse.vstack(commuting_matrix, format='csr')
        
        sparse.save_npz(os.path.join(source_folder, f'{metapath}.npz'), commuting_matrix)
        
        mdl = SVC(kernel='precomputed')
        mdl.fit(commuting_matrix.todense(), apps.label)
        
        # collect metrics
        accuracy = accuracy_score(apps.label, mdl.predict(commuting_matrix.todense()))
        recall = recall_score(apps.label, mdl.predict(commuting_matrix.todense()))
        f1 = f1_score(apps.label, mdl.predict(commuting_matrix.todense()))
        metrics = metrics.append(pd.Series({
            'kernel': metapath, 
            'acc': accuracy, 
            'recall': recall, 
            'f1': f1
        }), ignore_index=True)
        
        with open(os.path.join(source_folder, f'{metapath}.mdl'), 'wb') as file:
            pickle.dump(mdl, file)
    print(metrics.set_index('kernel'))

    
def load_apps(source_folder):
    apps = (
        pd.read_csv(
            os.path.join(source_folder, 'app_map.csv'),
            dtype=str,
            index_col='app'
        ).join(
            pd.read_csv(
                os.path.join(source_folder, 'app_list.csv'),
                dtype=str,
                index_col='app'
            )
        )
    )
    category = pd.read_csv(
        os.path.join('data', 'out', 'all-apps', 'app_list.csv'),
        index_col='app', 
        usecols=['app', 'category'], 
        squeeze=True)
    apps['category'] = category
    # ONLY FOR TESTING fill missing vals
    apps['category'] = apps['category'].fillna({'testapp1': 'test', 'testapp2': 'malware'})
    
    apps = apps.reset_index().set_index(apps.uid.str.replace('app', '').astype(int)).sort_index()
    apps['label'] = (apps.category=='malware').astype(int)
    return apps
    