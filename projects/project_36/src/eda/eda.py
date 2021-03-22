
import os
import json
import random
import time
import sys
import pprint
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib.pyplot import figure
from functools import partial
from IPython.display import display, Markdown, Latex
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# data_creation_path=os.path.join(os.path.dirname(os.getcwd()))
# os.chdir(os.path.dirname(os.getcwd()))
# sys.path.insert(1, data_creation_path)
# print(os.getcwd())
import src.data_creation.explore as explore
import src.data_creation.json_functions as jf
import src.graph_learning.word2vec as word2vec
from run import update_paths

def get_w2v_embeddings(src):
    '''
    Function to get word2vec embeddings from <src>
    
    Parameters
    ----------
    src: str, required
        Path of embedding file to read
    Returns
    -------
    None
    '''
    embeddings=[]
    with open(src, "r") as file:
        next_line=file.readline()
        next_line=file.readline()
        while next_line:
            embeddings.append(next_line.split(" ")[1:-1])
            next_line=file.readline()
    embeddings=[list(map(float, layer)) for layer in embeddings]
    return embeddings

def get_m2v_embeddings(src):
    '''
    Function to get word2vec embeddings from <src>
    
    Parameters
    ----------
    src: str, required
        Path of embedding file to read
    Returns
    -------
    None
    '''
    embeddings=[]
    with open(src, "r") as file:
        next_line=file.readline()
        next_line=file.readline()
        while next_line:
            embeddings.append(next_line.split(" ")[:-1])
            next_line=file.readline()
    # embeddings=[list(map(float, layer)) for layer in embeddings]
    return embeddings

def w2v_on_m2v(embeddings):
    model = Word2Vec(
        corpus,
        min_count=1,
        size= corp_size,
        workers= work_size,
        window = window_size,
        sg = 1
    )
    save_w2v_embedding(model, corp_size, unique_apis, **params)
    print("Word2Vec done in " +str(time.time() -s ) + " seconds")
    return

def get_corpus(embeddings, node):
    '''
    Function to get corpus of nodes for w2v embeddings
    Parameters
    ----------
    embeddings: 2dListOfStrings, required
        2d list of string nodes 
    node: char or str, requried
        character of the nodetype to match
    Returns
    -------
    A list of node embeddings
    '''
    corpus=[[int(emb[1:]) for emb in layer if node in emb] for layer in embeddings]
    return corpus#[int(c[1:]) for c in corpus]

def get_walk_start(embeddings):
    '''
    Function to get order of apps walked in <embeddings>
    Parameters
    ----------
    embeddings: 2dListofStrings, required
        2d list of metapath2vec embeddings
    Returns
    -------
    List of first app node in each layer of <embeddings>
    '''
    return [layer[0] for layer in embeddings]

def merge_dictionaries(dicts):
    '''
    Function to merge an iterable of dictionaries

    Parameters
    ----------
        dict: iterableOfDictionaries, required
            An iterable, such as a list or numpy array, of dictionaries

    Returns
    -------
    A single dictionary of key-value pairs from dictionaries in all elements of <dicts>
    '''
    result={}
    for d in dicts:
        result.update(d)
    return result

def save_dataframe(**kwargs):
    '''
    Function to save dataframe <df> to path <dst>

    Parameters
    ----------
    df: Pandas DataFrame, required
        Dataframe to save to csv
    dst: str, required
        Path to save csv file to
    fn: str, required
        Filename to save <df> to
    verbose: logical, required
        If true print <dst> and <fn> to standard output
    update_nodes: logical, requried
        If true update the nodes in each column from integers to strings with a character denoting
        node type. If true 'a' will be appended to the front of app nodes, 'b' to will be appended
         to the front of block nodes, 'p' will be appended to the front of package nodes, and 'c' 
         will be appended to the front of api-call nodes
    Returns
    -------
    None
    '''

    def update_nodes(col):
        update_table={
            "app":"a",
            "block":"b",
            "api_call":"c",
            "package":"p"
        }
        tag=pd.Series([update_table[col.name]]*col.size)
        col=tag.str.cat(col.astype(str).values, sep='')
        return col

    fp=os.path.join(kwargs["dst"], kwargs["fn"])
    if kwargs["verbose"]:
        print("Saving malware app data to %s"%fp)
    if kwargs["update_nodes"]:
        df=df.apply(update_nodes)
    df.to_csv(fp)

def average_unique_by_app(df, col):
    '''
    Returns the average value of each column grouped by <df[col]>
    '''
    return df.groupby(col).nunique().mean()

def get_neighbors_by_node(data, display_out=True):
    '''
    Function to get average numbers of neighbors per type of node in <data>

    Parameters
    ----------
    data: Pandas Dataframe, required
        Pandas dataframe of rowwise data for every `block`, `api_call`, `package`, 
        and `app` node combination
    Returns
    -------
    Pandas dataframe of average neigbors per `block`, `api_call`, `package`, and `app` node
    '''
    avg_neighbors=[]
    col_names=[]
    for col in data.columns:
        mean_neighbors=average_unique_by_app(data, col)
        col_names.append("avg_%s_neighbors"%col)
        avg_neighbors.append(mean_neighbors)
    df=pd.concat(avg_neighbors, axis=1)
    df.rename(columns=dict(zip(df.columns, col_names)), inplace=True)
    if display_out:
        display(df)
    return df

def get_apps(src, lim=None, verbose=False):
    app_files = [x for x in os.listdir(src) if x[-4:]== "json"]
    mal_files = [x for x in app_files if '_M_' in x]
    # num_mal = len(mal_files)
    ben_files = [x for x in app_files if '_B_' in x]
    # num_ben = len(ben_files)
    if verbose:
        print('Malicious Apps: ', num_mal)
        print('Benign Apps: ', num_ben)

    if lim:
        random.shuffle(mal_files)
        random.shuffle(ben_files)
        mal_files=mal_files[:lim]
        ben_files=ben_files[:lim]
    return ben_files, mal_files

def get_dataframes(path, apps, feature, malicious):
    if malicious:
        fname="Malware"
    else:
        fname="Benign"
    df = explore.create_feature_frame(path, apps, feature).sum(axis=1).to_frame(name=fname).apply(lambda x: x/len(apps))
    df = df[df[fname] != 1/len(apps)]
    return df

def get_df_stats(mal_df, ben_df):
    
    idx=list(mal_df.describe().index)
    mal_values=mal_df.describe().T.values[0]
    ben_values=ben_df.describe().T.values[0]
    
    both=zip(("stats",'malicious','benign'),(idx, mal_values, ben_values))
    return pd.DataFrame(dict(both))

def plt_unique(mal_df, ben_df, groupby):
    '''
    Funcion to plot the number of unique nodes per app

    Parameters
    ----------
    mal_df: Pandas Dataframe, required
        DataFrame of malware data to plot
    ben_df: Pandas Dataframe, required
        DataFrame of benign data to plot
    Returns
    -------
    None
    '''
    mal_nunique=average_unique_by_app(mal_df, groupby)
    ben_nunique=average_unique_by_app(ben_df, groupby)


    malx=[col.capitalize().replace("_"," ")+"s/%s Node"%groupby.capitalize() for col in mal_df.columns if col!=groupby]
    benx=[col.capitalize().replace("_"," ")+"s/%s Node"%groupby.capitalize() for col in ben_df.columns if col!=groupby]
    malx.extend(benx)
    # x=[malx[i]+"/"+types[i]+ " Node" for i in range(len(types))]
    x=malx
    y = list(mal_nunique.append(ben_nunique).values)
    x_pos = [i for i, _ in enumerate(x)]

    figure(figsize=(15,10)) 
    plt.bar(x_pos, y, color='black')
    plt.xlabel("Feature Nodes")
    plt.ylabel("Average Number of Neighbors Per %s Node"%groupby.capitalize())
    plt.title("Average Number of Neighbors Per %s Node"%groupby.capitalize())

    plt.xticks(x_pos, x)

    plt.show()
    
def plt_most_freq(src, value_type, proportion=False, ymin=None, ymax=None):\
    
    lookup_features={
        'api':'API Calls',
        'invoke':'Invoke Methods',
        'package': 'Packages'
    }
    
    benign_apps, malicious_apps=get_apps(src)
    benign_data=get_dataframes(src, benign_apps, value_type, False)
    malicious_data=get_dataframes(src, malicious_apps, value_type, True)
    
    malware_col = malicious_data["Malware"]#[:N]
    benign_col = benign_data["Benign"]#[:N]
    
    if proportion:
        malware_col=round(malware_col/sum(malware_col)*100,2)
        benign_col=round(benign_col/sum(benign_col)*100,2)
    
    #invoke_merge = ben_data.join(mal_data, how='outer').fillna(0).round(decimals=0)
    
    figure(figsize=(15,10)) 
    ind = np.arange(len(malicious_data.index)) 
    width = 0.4    
    plt.bar(ind, benign_col, width, label='Benign', color='white', edgecolor='black', hatch="\\")
    plt.bar(ind + width, malware_col, width, label='Malware', color='black')

    plt.ylabel('Log Scaled Proportion of %s to Number of Apps'%lookup_features[value_type])
    plt.title('Frequency of %s'%value_type)
    plt.yscale('log')

    plt.xticks(ind + width / 2, malicious_data.index, rotation='vertical')
    plt.legend(loc='best')
    plt.show()
    
def plt_m2v_embeddings(dst, fn, embeddings, display=True):
    '''
    Fucntion to plot histogram of weights in <embeddings>

    Parameters
    ----------
    dst: str, required
        Path to save plots to
    fn: str, required
        filename to save plot as
    embeddings: 2dListOfFloats, required
        List of embeddings to make histogram of
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None. 
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    embeddings=np.sum(embeddings, axis=1)
    plt.hist(embeddings,100)
    plt.title("Distribution of Unstructured API Embedding Weights (n=804,296)")
    plt.xlabel("Aggregate of Embedding Row")
    
    if display:
        plt.show()
        
    fp=os.path.join(dst,fn)
    os.makedirs(dst, exist_ok=True)
    print("saving to: %s"%fp)
    plt.savefig(fp)

def print_unstructured_embedding_weights(src, dst, fn, display=True):
    '''
    Function to plot histograms of <model> embedding weights
    
    Parameters
    ----------
    model: Python Object, required
        Trained SHNE model
    dst: str, required
        Path to save plots to
    fn: str, required
        filename to save plot as
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None. 
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    model=torch.load(src)
    figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    embeddings = np.sum(model["word_embedding.weight"].numpy(),axis=1)
    plt.hist(embeddings,100)
    plt.title("Distribution of Unstructured API Embedding Weights (n=804,296)")
    plt.xlabel("Aggregate of Embedding Row")
    
    if display:
        plt.show()
        
    fp=os.path.join(dst,fn)
    os.makedirs(dst, exist_ok=True)
    print("saving to: %s"%fp)
    plt.savefig(fp)
    
def print_w2v_embedding_weights(embeddings, dst, fn, display=True):
    '''
    Function to plot histograms of <model> embedding weights
    
    Parameters
    ----------
    model: Python Object, required
        Trained SHNE model
    dst: str, required
        Path to save plots to
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None. 
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    embeddings = np.sum(embeddings,axis=1)
    plt.hist(embeddings,100)
    plt.title("Distribution of Unstructured API Embedding Weights (n=804,296)")
    plt.xlabel("Aggregate of Embedding Row")
    
    if display:
        plt.show()
        
    fp=os.path.join(dst,fn)
    os.makedirs(dst, exist_ok=True)
    print("saving to: %s"%fp)
    plt.savefig(fp)

def print_structured_embedding_weights(src, dst, fn, display=True):
    '''
    Function to plot histograms of <model> embedding weights
    
    Parameters
    ----------
    model: Python Object, required
        Trained SHNE model
    dst: str, required
        Path to save plots to
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None. 
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    model=torch.load(src)
    fig, ax = plt.subplots(2, 2,sharey='all', sharex='all')
    
    a = np.sum(model["a_latent"].numpy(),axis=1)
    p = np.sum(model["p_latent"].numpy(),axis=1)
    c = np.sum(model["v_latent"].numpy(),axis=1)
    b = np.sum(model["b_latent"].numpy(),axis=1)
    
    ax[0,0].hist(a, 100, color="red")
    ax[0,0].set_title("APP Embedding Weights")
    ax[0,0].set_xlabel("Aggregate of Embedding Row")
    
    ax[0,1].hist(a, 100, color="blue")
    ax[0,1].set_title("Package Embedding Weights")
    ax[0,1].set_xlabel("Aggregate of Embedding Row")
    
    ax[1,0].hist(a, 100, color="green")
    ax[1,0].set_title("API Call Embedding Weights")
    ax[1,0].set_xlabel("Aggregate of Embedding Row")
    
    ax[1,1].hist(a, 100, color="grey")
    ax[1,1].set_title("Block Embedding Weights")
    ax[1,1].set_xlabel("Aggregate of Embedding Row")
    
    figure(num=None, figsize=(100, 20), dpi=80, facecolor='w', edgecolor='k')
    
    fig.text(0.5, 0.95, 'Distribution of Node Embeddings', ha='center')
    fig.text(0.04, 0.5, 'Embedding Weight Count', va='center', rotation='vertical')
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    if display:
        plt.show()
        
    fp=os.path.join(dst,fn)
    os.makedirs(dst, exist_ok=True)
    print("saving to: %s"%fp)
    fig.savefig(fp)

def plot_embeddings(malicious, benign,eda_params, verbose=True):
    model=torch.load(eda_params["full_model_path"])
    save_dir=eda_params["save_dir"]

    if verbose:
        print("Model's state_dict:")
        for param_tensor in model:
            print(param_tensor, "\t", model[param_tensor].size())
        print()
        print("Printing Histogram Stuctured Embedding Weights for above model")

    print_structured_embedding_weights(
        model=model,
        dst=save_dir,
        fn=eda_params["structured_embedding_hist_filename"],
        display=verbose
    )

    if verbose:
        print()
        print("Histogram of Unstructured Embedding Weights")

    print_unstructured_embedding_weights(
        model=model,
        dst=save_dir,
        fn=eda_params["unstructured_embedding_hist_filename"],
        display=True
    )

def analyze_frequencies(src, features):
    #features: ['api','invoke','package']
    benign_apps, malicious_apps=get_apps(src)
    
    distributions=[]
    
    for feature in features:
        distributions.append(get_dataframes(src, malicious_apps, feature, True))
        distributions.append(get_dataframes(src, benign_apps, feature, False))
    
    ix=0
    fx=0
    while ix<len(distributions):
        plt_unique(distributions[ix], distributions[ix+1], features[fx])
        ix+=2
        fx+=1
        
def plot_counts(src):
    print(1)
    benign_apps, malicious_apps=get_apps(src)
    print(2)
    lookup_features={
        'api':'API Calls',
        'invoke':'Invoke Methods',
        'package': 'Packages'
    }
    
    #benign_apps, malicious_apps=get_apps(src)
    
    for value_type in lookup_features.keys():
        benign_data=get_dataframes(src, benign_apps, value_type, False)
        print(3)
        malicious_data=get_dataframes(src, malicious_apps, value_type, True)
        print(4)
        return benign_data
    malware_col = malicious_data["Malware"]#[:N]
    benign_col = benign_data["Benign"]#[:N]
    
    # if proportion:
    #     malware_col=round(malware_col/sum(malware_col)*100,2)
    #     benign_col=round(benign_col/sum(benign_col)*100,2)
    
    #invoke_merge = ben_data.join(mal_data, how='outer').fillna(0).round(decimals=0)
    
    figure(figsize=(15,10)) 
    ind = np.arange(len(malicious_data.index)) 
    width = 0.4    
    plt.bar(ind, benign_col, width, label='Benign', color='white', edgecolor='black', hatch="\\")
    plt.bar(ind + width, malware_col, width, label='Malware', color='black')

    plt.ylabel('Log Scaled Proportion of %s to Number of Apps'%lookup_features[value_type])
    plt.title('Frequency of %s'%value_type)
    plt.yscale('log')

    plt.xticks(ind + width / 2, malicious_data.index, rotation='vertical')
    plt.legend(loc='best')
    plt.show()

def graph_neighbors(benign, malicious, dst, display_out=True):
    '''
    Function to display bar graph of average neighbors per node in <data>
    
    Parameters
    ---------
    benign: Pandas Dataframe, required
        a dataframe of the average benign neighbors per node
        Note <malware> and <benign> both must have the same column names and indices
    malware: Pandas Dataframe, required
        a dataframe of the average benign neighbors per node.
        Note <malware> and <benign> both must have the same column names and indices
    dst: str, reqruied
        string path to save plots to
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None.
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    for col in malicious.columns:
        x1=malicious.loc[:,col].dropna()
        y1=list(x1.values)
        x1=list(x1.index)
        
        x2=benign.loc[:,col].dropna()
        y2=list(x2.values)
        x2=list(x2.index)
        print("test")
        figure(figsize=(25,25)) 
        print("test3")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        print("test2")
        ax1.bar(x1, y1, color='red')
        ax1.set_xlabel("Malicious Nodes")
    
        ax1.set_xticks(x1)
        
        ax2.bar(x2, y2, color='green')
        ax2.set_xlabel("Benign Nodes")
        node=" ".join(col.split("_")[1:-1])#.capitalize()
        # print(node)
        fig.text(0.5, 0.95, 'Mean number of neighbors per %s node'%node, ha='center')
        fig.text(0.04, 0.5, 'Mean Neighbors', va='center', rotation='vertical')
        
        if display_out:
            print()
            display(Markdown("### Plotting average number of neigbors per %s node"%node))
            plt.show()
        
        fn="avg_%s_neighbors.pdf"%node.replace(" ", "_")
        fp=os.path.join(dst,fn)
        os.makedirs(dst, exist_ok=True)
        print("saving to: %s"%fp)
        fig.savefig(fp)
        
def graph_neighbors_subplt(benign, malicious, dst, fn, display=True):
    '''
    Function to display bar graph of average neighbors per node in <data>
    
    Parameters
    ---------
    benign: Pandas Dataframe, required
        a dataframe of the average benign neighbors per node
        Note <malware> and <benign> both must have the same column names and indices
    malware: Pandas Dataframe, required
        a dataframe of the average benign neighbors per node.
        Note <malware> and <benign> both must have the same column names and indices
    dst: str, reqruied
        string path to save plots to
    fn: str, required
        filename to save plot as
    display: bool, optional
        If true display to standard output
    Returns
    -------
    None.
    If <display> is True, then plots are shown in standard output. 
    Plots will be saved to <dst> as .pdf files
    '''
    fn="avg_neighbors.pdf"
    
    figure(figsize=(30,15)) 
    fig, ax = plt.subplots(len(malicious.columns), 2,sharey='row')
    for i, col in enumerate(malicious.columns):
        x1=malicious.loc[:,col].dropna()
        y1=list(x1.values)
        x1=list(x1.index)
        
        x2=benign.loc[:,col].dropna()
        y2=list(x2.values)
        x2=list(x2.index)
        
        node=" ".join(col.split("_")[1:-1]).capitalize()
    
        ax[i, 0].bar(x1, y1, color='red')
        ax[i, 0].set_ylabel("%s neighbors"%node)
    
        ax[i, 0].set_xticks(x1)
        
        ax[i, 1].bar(x2, y2, color='green')
        ax[i, 1].set_ylabel("%s neighbors"%node)
        
    ax[-1,0].set_xlabel("Malicious Nodes")
    ax[-1,1].set_xlabel("Benign Nodes")
    fig.text(0.5, 0.95, 'Mean number of neighbors per node', ha='center')
    fig.text(0.04, 0.5, 'Mean Neighbors', va='center', rotation='vertical')
    
    plt.subplots_adjust(hspace=1, wspace=0.5)
        
    if display:        
        plt.show()
            
    fp=os.path.join(dst,fn)
    os.makedirs(dst, exist_ok=True)
    print("saving to: %s"%fp)
    fig.savefig(fp)

def get_nodes(block, **kwargs):
        '''
        Helper function to get api_call, package, block, and app nodes from a single block in <blocks>.
        Requires key worded arguments.

        Parameters
        ----------
        block: listOfStrings, required
            code block from <blocks> to get nodes from

        Returns
        -------
        Dictionary of api_call, package, block, and app nodes
        '''
        block_str=" ".join(block)

        naming_key=kwargs["naming_key"]
        fn=kwargs["filename"]

        parsed={
            "app":[],
            "block":[],
            "api_call":[],
            "package":[]
        }
        
        for call in block:
            try:
                api_call=call.split("}, ")[1].split(" ")[0].strip()
                package=call.split(";")[0].split(",")[-1].strip()
                
                #make string tag into integer to save space
                get_tag_num=lambda x: int(x[1:])

                app_tag=naming_key["apps"][fn]
                block_tag=naming_key["blocks"][block_str]
                package_tag=naming_key["packages"][package]
                api_call_tag=naming_key["calls"][api_call]
                
                parsed["app"].append(get_tag_num(app_tag))
                parsed["block"].append(get_tag_num(block_tag))
                parsed["package"].append(get_tag_num(package_tag))
                parsed["api_call"].append(get_tag_num(api_call_tag))
            except IndexError:
                pass
        return parsed

def read_data_extract_to_df(args):
    '''
    Function to extract app data from respective json file of parsed api calls into dataframe of 
    each |<app>|<block>|<api_call>|<package>| row combination.

    Parameters
    ----------
    args: tuple, required
        Tuple of two arguments, a key of the names of each <app>, <block>, <api_call>, and <package>

    Returns
    -------
    Dataframe of each |<app>|<block>|<api_call>|<package>| row combination.
    '''
    multiprocessing=args[2]
    naming_key=args[1]
    fp=args[0]
    fn=os.path.basename(fp).replace(".json","")
    
    blocks=np.array(jf.load_json(fp), dtype="object")
    blocks=list(filter(None,blocks))#remove empty blocks
    
    args_mapping={
        "naming_key":naming_key,
        "filename":fn
    }

    if multiprocessing:
        with Pool() as pool:
            parsed=pool.map(partial(get_nodes,**args_mapping), blocks)
    else:
        parsed=[]
        for block in blocks:
            parsed.append(get_nodes(block, **args_mapping))
    parsed=merge_dictionaries(parsed)
    return pd.DataFrame(parsed)#.to_dataframe()

def get_node_data(display_data=True, **kwargs):
    '''
    Function to get dataframes of malicious and benign apps. 
    Dataframes are rowwise, with columns for each |<app>|<block>|<api_call>|<package>|
    combination.

    Parameters
    ----------
    verbose: logical, required. Default True
        If true print updates on eda. Do not print updates otherwise
    lim: int, required. Default None
        If not `None` then limit apps by that ammount 
    multiprocessing: logical, required. Default True
        If true run with multiprocessing. Run in serial otherwise
    save_data: logical, required. Defualt True
        If true save <mal_df> and <ben_df> to <dst>
    mal_filename: str, required
        filename to save malware node data to
    ben_filename: str, required
        filename to save benign node data to
    key_fn: str, required
        Path to json file lookup table of node code for repectives <app>, <block>, <api_call>, and <package>
        strings    
    src: str, required
        Path of parsed smali code in json files
    dst: str, required
        Path to save <mal_df> and <ben_df> as csv files
    Returns
    -------
    Two dataframes. 
    arg1: Dataframe of malicious app-block-api call-package combinations
    arg2: Dataframe of mabenignlicious app-block-api call-package combinations
    '''
    verbose=kwargs["verbose"]
    lim=kwargs["lim"]
    multiprocessing=kwargs["multiprocessing"]
    save_data=kwargs["save_data"]
    mal_filename=kwargs["mal_filename"]
    ben_filename=kwargs["ben_filename"]
    key_fn=kwargs["key_fn"]
    src=kwargs["src"]
    dst=kwargs["dst"]
    
    # print("test11")
    if verbose:
        print("Retrieving naming key from `%s`"%key_fn)
        start=time.time()
        key=jf.load_json(key_fn)["get_key"]
        print("Retrieved naming key in %i seconds"%(time.time()-start))
    else:
        key=jf.load_json(key_fn)["get_key"]
    # print("key",key)
    # mal_apps, ben_apps=get_apps(src, lim)
    if lim==None:
        mal_apps=[(os.path.join(src,file), key, multiprocessing) for file in os.listdir(src) if "_M_" in file]
        ben_apps=[(os.path.join(src,file), key, multiprocessing) for file in os.listdir(src) if "_B_" in file]
    else:
        mal_apps=[(os.path.join(src,file), key, multiprocessing) for file in os.listdir(src) if "_M_" in file][:lim]
        ben_apps=[(os.path.join(src,file), key, multiprocessing) for file in os.listdir(src) if "_B_" in file][:lim]
    # progress=tqdm.tqdm
    mal_data=[]
    ben_data=[]

    if verbose:
        print("Found %i malicious apps and %i benign apps to extract data from"%(len(mal_apps), len(ben_apps)))
        start=time.time()
        for app in tqdm(mal_apps):
            mal_data.append(read_data_extract_to_df(app))
        print("Retrieved data on %i malicious apps in %i seconds"%(len(mal_data),time.time()-start))
        
        start=time.time()
        for app in tqdm(ben_apps):
            ben_data.append(read_data_extract_to_df(app))
        print("Retrieved data on %i benign apps in %i seconds"%(len(mal_data),time.time()-start))

    else:
        for app in tqdm(mal_apps):
            mal_data.append(read_data_extract_to_df(app))
        for app in tqdm(ben_apps):
            ben_data.append(read_data_extract_to_df(app))

    mal_df=pd.concat(mal_data)
    ben_df=pd.concat(ben_data)

    if save_data:
        mal_fn=os.path.join(dst, mal_filename)
        ben_fn=os.path.join(dst, ben_filename)
        if verbose:
            print("Saving malware app data to %s"%mal_fn)
            print("Saving benign app data to %s"%ben_fn)

    if display_data:
        print("Malware data:")
        display(mal_df.head())
        print("\nBenign data:")
        display(ben_df.head())        
    return mal_df, ben_df

def get_m2v_w2v_layers(**kwargs):
    '''
    Function to get embedding layers from metapath2vec walk.
    
    Parameters
    ----------
        Embedding layers always returned
    src: str, required
        path of metapath2vec walk to create embedding layers from
    dst: str, requried
        path to save embedding layers to
    filename: str, required
        name of the file to save embedding layers to
    vocab_size: int, required
        Size of vocabulary to train word2vec
    nworkers: int, required
        Number of workers to run word2vec on
    window_size: int, required
        word2vec sliding window size
    Returns
    -------
    embeddings layers from metapath walk saved to <src>
    '''
    src=kwargs["src"]
    fn=kwargs["filename"]
    dst=kwargs["dst"]

    m2v_embeddings=get_m2v_embeddings(src)
    app_nodes=get_walk_start(m2v_embeddings)
    unique_apps=dict(zip(list(set(app_nodes)), range(1, len(set(app_nodes))+1)))
    w2v_model=Word2Vec(
        m2v_embeddings,
        min_count=1,
        size= kwargs["vocab_size"],
        workers= kwargs["nworkers"],
        window = kwargs["window_size"],
        sg = 1
    )

    word2vec.save_w2v_embedding(
        model=w2v_model,
        corp_size=kwargs["vocab_size"],
        unique_apis=unique_apps,
        embeddings_filename=fn,
        save_unique_apis=False,
        save_dir=dst,
        unique_api_filename=None,
        verbose=True
    )
    return get_w2v_embeddings(os.path.join(dst, fn))

def pca(model, layer, y_actual, num_components, dst, fn, display=True):
    '''
    Function to perform PCA analysis on <model>
    Parameters
    ----------
    model: hashable, required
        A hashable table of embedding layers to run pca on
    layer: string, requried
        the name of the layer to run pca on
    num_components: int, required
        the number of components for pca
    Returns
    -------
    None.
    '''
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(model[layer].numpy())
    df = pd.DataFrame()
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df.to_numpy())
    labs = kmeans.labels_
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=y_app_lab,
        data=df,
        alpha=1,
        cmap='tab10'
    )

    plt.xlabel('pca-one')
    plt.ylabel('pca-two')
    plt.savefig(os.path.join(dst,fn))

    if display:
        plt.show()

def EDA(src, verbose=True, lim=None, multiprocessing=False, save_data=True):
    '''
    Function to run EDA.

    Parameters
    ----------
        verbose: logical, optional. Default True
            If true print updates on eda. Do not print updates otherwise
        lim: int, optional. Default None
            If not `None` then limit apps by that ammount 
        multiprocessing: logical, optional. Default False
            WARNING: DO NOT RUN! Currently not working. It is faster to run in serial than multi. 
            If true run with multiprocessing. Run in serial otherwise         
        save_data: logical, optional. Defualt True
            If true save <mal_df> and <ben_df> to <dst>
        mal_filename: str, required
            filename to save malware node data to
        ben_filename: str, required
            filename to save benign node data to 
        src: str, required
            Path of data to run EDA on
        dst: str, required
            Path to save <mal_df> and <ben_df> as csv files
        key_fn: str, required
            Path to json file lookup table of node code for repectives <app>, <block>, <api_call>, and <package> 
            strings  
    '''

    params=jf.load_json(src)
    params=update_paths(False, params)

    if verbose:
        print()
        print("Parameters:")
        pprint.pprint(params)

    eda_params=params["eda-params"]

    key_fn=os.path.join(eda_params["dict_directory"], eda_params["data_naming_key_filename"])

    malware, benign=get_node_data(
        verbose=verbose,
        lim=lim, 
        multiprocessing=multiprocessing,
        save_data=save_data,
        mal_filename=eda_params["malware_node_filename"],
        ben_filename=eda_params["benign_node_filename"],
        key_fn=key_fn,
        src=eda_params["data_extract_loc"],
        dst=eda_params["save_dir"]
    )
    return malware, benign

if __name__=="__main__":
    param_fp=os.path.join('config','params.json')
    params=jf.load_json(param_fp)
    eda_params=params['eda-params']

    src=eda_params['data_extract_loc']
    key_fn=eda_params["data_naming_key"]

    # pprint.pprint(params)
    # print("test")
    malware, benign=get_node_data(
        key_fn=key_fn, 
        src=src
    )