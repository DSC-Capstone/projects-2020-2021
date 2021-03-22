import pandas as pd
import numpy as np
import json, os, re, random, sys
import networkx as nx
from stellargraph import StellarGraph as sg
from stellargraph import IndexedArray

def get_jsons(src, filename):
    '''
    Helper function to open json files
    
    Parameters
    ----------
    src
        The source directory of the json file to return
        
    filename
        The name of the file to return
        
    Returns
    -------
    Returns a dictionary containing the data in filename
    '''
    with open(os.path.join(src,filename)) as dic:
        return json.load(dic)
    
def make_stellargraph(src):
    '''
    Function to create a StellarGraph network of apps, api calls, blocks, packages, and invoke types.
        
    Returns
    -------
    Returns an instance of the StellarGraph network representation of the files found in directory in "key_directory" of config/dict_build.json" file 
    '''
    #get dictionaries of relationships
    A=get_jsons(src, "dict_A.json")
    B=get_jsons(src, "dict_B.json")
    P=get_jsons(src, "dict_P.json")
    C=get_jsons(src, "api_calls.json")
    
    #get all nodes
    a_nodes=IndexedArray(index=list(set(A.keys())))
    b_nodes=IndexedArray(index=list(set(B.keys())))
    c_nodes=IndexedArray(index=list(set(C.keys())))
    p_nodes=IndexedArray(index=list(set(P.keys())))
    print("Nodes Created")
    
    graph_nodes={
        "app_nodes":a_nodes,
        "block_nodes":b_nodes,
        "api_call_nodes":c_nodes,
        #"invoke_type_nodes":i_nodes,
        "package_nodes":p_nodes
    }

    #get all edges
    a_edges=np.array(list(nx.Graph(A).edges))
    b_edges=np.array(list(nx.Graph(B).edges))
    p_edges=np.array(list(nx.Graph(P).edges))
    print("Edges computed")
    
    #np.concatenate contributes to majority of runtime for make_stellargraph(src)
    edges=pd.DataFrame(np.concatenate((a_edges,b_edges,p_edges))).rename(columns={0:"source",1:"target"})
    print("Concatted")
    
    length0=edges.shape[0]
    
    removed=list(edges.loc[edges.source==edges.target].target)
    
    edges=edges.loc[edges.source!=edges.target].copy()
    
    length1=edges.shape[0]
    if length0-length1!=0:
        print("Removed %i repeated keys"%(length0-length1))
        for r in removed:
            print(r)
    return sg(graph_nodes,edges)