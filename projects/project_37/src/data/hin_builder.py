import os, json
import pandas as pd
import numpy as np
import pickle
from p_tqdm import p_umap
from shutil import copyfile

from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from stellargraph import StellarGraph, IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk

from gensim.models import Word2Vec


def get_features(outfolder, walk_args=None, w2v_args=None, redo=False):
    '''
    Implements metapath2vec by:
    1. Building a graph
    2. Performing a random metapath walk then 
    3. Applying word2vec on the walks generated.
    ---------
    Parameters:
    outfolder:      Path to directory where output will be saved, should contain app_list.csv
    walk_args:      Arguments for stellargraph.data.UniformRandomMetaPathWalk
    w2v_args:       Arguments for gensim.models.Word2Vec
    '''
    # save parameters to outfolder
    params = {
        "outfolder": outfolder,
        "walk_args": walk_args,
        "w2v_args": w2v_args 
    }
    with open(os.path.join(outfolder, 'params.json'), 'w') as param_file:
        json.dump(params, param_file)
    
    # define paths
    app_list_path = os.path.join(outfolder, 'app_list.csv')
    nodes_path = os.path.join(outfolder, 'nodes.json')
    edge_path = os.path.join(outfolder, 'edges.csv')
    graph_path = os.path.join(outfolder, 'graph.pkl')
    feature_path = os.path.join(outfolder, 'features.csv')
    app_heap_path = os.path.join('data', 'out', 'all-apps', 'app-data/')
    metapath_walk_outpath = os.path.join(outfolder, 'metapath_walk.json')
    
    # generate app list
    apps_df = pd.read_csv(app_list_path)
    app_data_list = app_heap_path + apps_df.app + '.csv'
    
    if os.path.exists(graph_path) and not redo:  # load graph from file if present
        with open(graph_path, 'rb') as file:
            g = pickle.load(file)
    else:                                        # otherwise compute from data
        g = build_graph(outfolder, app_data_list, nodes_path, edge_path)

    # save graph to file
    with open(graph_path, 'wb') as file:
        pickle.dump(g, file)

    if os.path.exists(metapath_walk_outpath) and not redo:  # load graph from file if present
        with open(metapath_walk_outpath, 'r') as file:
            metapath_walks = json.load(file)
    else:                                        # otherwise compute from data
        # random walk on all apps, save to metapath_walk.json
        print('Performing random walks')
        rw = UniformRandomMetaPathWalk(g)
        app_nodes = list(
            apps_df.app.map(
                pd.read_csv(os.path.join(outfolder, 'app_map.csv'), index_col='app').uid
            )
        )
        metapath_walks = rw.run(app_nodes, n=walk_args['n'], length=walk_args['length'], metapaths=walk_args['metapaths'])
        
        with open(metapath_walk_outpath, 'w') as file:
            json.dump(metapath_walks, file)
    
    print('Running Word2vec')
    w2v = Word2Vec(metapath_walks, **w2v_args)
    
    features = pd.DataFrame(w2v.wv.vectors)
    features['uid'] = w2v.wv.index2word
    features['app'] = features['uid'].map(
        pd.read_csv(os.path.join(outfolder, 'app_map.csv'), index_col='uid').app
    )
    features = features[features.uid.str.contains('app')].set_index('uid')
    features.to_csv(feature_path)

def build_graph(outfolder, app_data_list, nodes_path, edge_path):
#     with Client() as client, performance_report(os.path.join(outfolder, "performance_report.html")):
#     print(f"Dask Cluster: {client.cluster}")
#     print(f"Dashboard port: {client.scheduler_info()['services']['dashboard']}")

    data = dd.read_csv(list(app_data_list), dtype=str).compute()

    nodes = {}
    api_map = None

    # setup edges.csv
    pd.DataFrame(columns=['source', 'target']).to_csv(edge_path, index=False)

    for label in ['api', 'app', 'method', 'package']:
        print(f'Indexing {label}s')
#         uid_map = data[label].unique()
        uid_map = pd.DataFrame()
        uid_map[label] = data[label].unique()

#             if base_data is not None: # load base items
#                 base_items = pd.read_csv(
#                     os.path.join(base_data, label+'_map.csv'),
#                     usecols=[label]
#                 )
#                 uid_map = pd.concat([base_items, uid_map], ignore_index=True).drop_duplicates().reset_index(drop=True)

        uid_map['uid'] = label + pd.Series(uid_map.index).astype(str)
        uid_map = uid_map.set_index(label)
        uid_map.to_csv(os.path.join(outfolder, label+'_map.csv'))
        nodes[label] = IndexedArray(index=uid_map.uid.values)

        # get edges if not api
        if label == 'api':
            api_map = uid_map.uid  # create api map
        else:
            print(f'Finding {label}-api edges')
            edges = data[[label, 'api']].drop_duplicates()
            edges[label] = edges[label].map(uid_map.uid)
            edges['api'] = edges['api'].map(api_map)
            edges.to_csv(edge_path, mode='a', index=False, header=False)

    del data
    
    # save nodes to file
    with open(nodes_path, 'wb') as file:
        pickle.dump(nodes, file)

    return StellarGraph(nodes = nodes, edges = pd.read_csv(edge_path))