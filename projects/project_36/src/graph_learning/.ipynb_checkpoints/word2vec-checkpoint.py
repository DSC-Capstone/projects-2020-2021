import os
import sys
import time
import numpy as np
from gensim.models import Word2Vec
from src.data_creation import json_functions
import pickle

def word2vec(walks, emb_type, **params):
    """Trains gensim Word2Vec model on node2vec or metapath2vec walks and saves learned embeddings to .emb file
    
    :param walks : list of lists
        Each list represents a random walk 
        
    :param params : dict
    dict["key"] where dict is global parameter dictionary and key returns word2vec parameter sub-dictionary
    
    :param emb_type : str
        Type of model that generated walks: "node2vec" or "metapath2vec"
    """
    start_learn = time.time()
    print("Starting Word2Vec")
    
    model = Word2Vec(walks, size=params["size"], window=params["window"], min_count=params["min_count"], sg=params["sg"], workers=params["workers"], iter=params["iter"])
    
    if params["verbose"]:
        print("--- Done Learning in " + str(int(time.time() - start_learn)) + " Seconds ---")
        print()
    
   # Store just the words + their trained embeddings.
    fp=os.path.join(params["save_dir"], params["model_filename"])
    model.wv.save(fp)
    if params["verbose"]:
        print("Saved %s to %s" %(params["model_filename"], params["save_dir"]))
    return

# optional if nodes are ints:
# walks = [[str(n) for n in walk] for walk in walks]
# The embedding vectors can be retrieved from model.wv using the node ID
# ex: print(model.wv["p2"].shape)
# Retrieve node embeddings and corresponding subjects
# node_ids = model.wv.index2word  # list of node IDs
# node_targets = node_subjects.loc[[int(node_id) for node_id in node_ids]]
# node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality

#################################
# FOR SHNE EMBEDDINGS
#################################

def create_w2v_embedding(path, path_to_unique_apis, **params):
    print("--- W2V Embedding ---")
    s = time.time()
    
    corp_size=params["size"]
    window_size=params["window"]
    work_size=params["workers"]
#     path_to_unique_apis=os.path.join(params[""])
    api_list = json_functions.load_json(path_to_unique_apis)["get_key"]["calls"].keys()
    unique_apis = dict(zip(api_list,list(range(1, len(api_list)+1)))) #key value pairs
    corpus = []
    
    for root, dirs, lister in os.walk(path):continue
        
    for i in lister:
        if "checkpoint" in i:
            continue
        temp = json_functions.load_json(path+i)
        temp = [item.split(" ")[-1] for sublist in temp for item in sublist]
        corpus.append(temp)
    app_ids = list(range(0,len(corpus)))
    abst = []
    content = []
    for app in corpus:
        try:
            abstracted = []
            for api in app:
                abstracted.append(unique_apis[api.split(" ")[-1]])
            abst.append(abstracted)
        except:
            continue
    content.append(abst)
    content.append(app_ids)
    fp=os.path.join(params["save_dir"], params["content_filename"])
    if params["verbose"]:
        print("Saved %s to %s" %(params["content_filename"], params["save_dir"]))
#     pickle.dump(content, open(fp, "wb"))
    
    print("Corpus construction done in " +str(time.time() -s ) + " seconds with " + str(len(corpus))+ " documents")
    s = time.time()
    
    #Model
    print("corpus",np.array(content).shape)
    print("work_size", work_size)
    print("window_size", window_size)
#     return corpus
    model = Word2Vec(corpus, min_count=1,size= corp_size,workers= work_size, window = window_size, sg = 1)
    return model
    save_w2v_embedding(model, corp_size, unique_apis, **params)
    print("Word2Vec done in " +str(time.time() -s ) + " seconds")

def save_w2v_embedding(model, corp_size, unique_apis, save_unique_apis=True, **params):
    #Remove pre existing 
    os.makedirs(params["save_dir"], exist_ok=True)
    embeddings_path=os.path.join(params["save_dir"],params["embeddings_filename"])
    try:
        os.remove(embeddings_path) #change to config
    except OSError:
        pass
    if save_unique_apis:
        json_functions.save_json(unique_apis,os.path.join(params["save_dir"], params["unique_api_filename"]))
        if params["verbose"]:
            print("Saved %s to %s" %(params["unique_api_filename"], params["save_dir"]))

    #Write to match SHNE intake format
    f = open(embeddings_path, "a") #change to config
    f.write(str(len(unique_apis.keys()))+" ")
    f.write(str(corp_size))
    f.write("\n")
    
    for p in unique_apis.keys():
        f.write(str(unique_apis[p]) +" ")
        for k in model[p]:
            f.write(str(k)+" ")
        f.write("\n")
    f.close()
    if params["verbose"]:
        print("Saved %s to %s" %(params["embeddings_filename"], params["save_dir"]))