import os
import re
import numpy as np
import pandas as pd
import pickle
import json
import random
import threading
import getopt
import sys
import time
import nbconvert
from scipy import sparse
from stellargraph import StellarGraph

from src.data_creation import get_data 
from src.data_creation import json_functions 
from src.data_creation import dict_builder 
from src.data_creation import build_network 
from src.graph_learning import node2vec
from src.graph_learning import metapath2vec
from src.graph_learning import word2vec
from src.data_creation.explore import splitall
from src.SHNE_code import SHNE
from src.eda import eda

def get_app_names(**kwargs):
    '''
    Function to extract file paths of apps
    
    Parameters
    ----------
    
    limiter: logical, required
        boolean value to limit number of app paths extracted. 
        If True benign and malicious apps will be limited by the number given by their 
        respective limit parameter,benign_lim and malignant_lim
    malignant_fp: str, required
        The file path of the malicious apps
    benign_fp: str, required
        The file path of the benign apps
    lim_benign: logical, required
        The number of benign apps that the outputed paths will be limited to
    lim_mal: logical, required
        The number of malicious apps that the outputed paths will be limited to
        
    Returns
    -------
    2 lists
        the first return value being a list of malignant app names found in malignant_fp
        the second return value being a list of benign app names found in benign_fp
    '''
    limiter=kwargs["limiter"]
    malignant_fp=kwargs["mal_fp"]
    benign_fp=kwargs["benign_fp"]
    benign_lim=kwargs["lim_benign"]
    malignant_lim=kwargs["lim_mal"]
    verbose=kwargs["verbose"]
    
    if verbose:
        print("\n--- Starting Malware Detection Pipeline ---")
    start = time.time()
    if limiter:
        if verbose:
            print("Limiting app intake to " + str(malignant_lim + benign_lim) + " apps")
        # print()
        mal_app_names = [[name+"/"+sub_name for sub_name in os.listdir(malignant_fp+"/"+name)] for name in os.listdir(malignant_fp) if os.path.isdir(malignant_fp + "/" + name)]
        benign_app_names = [name for name in os.listdir(benign_fp) if (os.path.exists(benign_fp + "/" + name+"/"+"smali"))]
        flat_list = []
        for sublist in mal_app_names:
            for item in sublist:
                flat_list.append(item)
        mal_app_names = flat_list

        flat_list = []
        mal_app_names = [[name+"/"+sub_name for sub_name in os.listdir(malignant_fp+"/"+name) if os.path.exists(malignant_fp+"/"+name+"/"+sub_name+"/smali")] for name in mal_app_names]
        for sublist in mal_app_names:
            for item in sublist:
                flat_list.append(item)
        mal_app_names = flat_list
        
        
        #get family of malware in its respective app name
        #mal_app_names_full = [family +"_"+ name for family in mal_app_names for name in os.listdir(malignant_fp+"/"+family) if os.path.isdir(malignant_fp +"/"+ family + "/" + name)]
        
        #randomize the list
        random.shuffle(mal_app_names) 
        random.shuffle(benign_app_names)
        
        #limit the apps
        try:
            mal_app_names = mal_app_names[:malignant_lim]
        except:
            mal_app_names = mal_app_names
            
        try:
            benign_app_names = benign_app_names[:benign_lim]
        except:
            benign_app_names = benign_app_names
            
        assert len(set(mal_app_names)) == len(mal_app_names), "DUPLICATE APP NAMES"
        assert len(set(benign_app_names)) == len(benign_app_names), "DUPLICATE APP NAMES"

    else:
        print()
        mal_app_names = [name for name in os.listdir(malignant_fp) if os.path.isdir(malignant_fp + "/" + name)]
        mal_app_names = [[name+"/"+sub_name for sub_name in os.listdir(malignant_fp+"/"+name)] for name in mal_app_names]
        flat_list = []
        for sublist in mal_app_names:
            for item in sublist:
                flat_list.append(item)
        mal_app_names = flat_list
        flat_list = []
        for sublist in mal_app_names:
            for item in sublist:
                flat_list.append(item)
        mal_app_names = flat_list
        
        benign_app_names = [name for name in os.listdir(benign_fp) if os.path.isdir(benign_fp + "/" + name)]
    if verbose:
        print("Found %i malicious apps, %i benign apps in %i seconds"%(len(mal_app_names), len(benign_app_names), time.time()-start))
        print()
    return mal_app_names, benign_app_names
        
def create_dictionary(**kwargs):
    '''
    Create dictionaries of individual app api calls.
    Dictionaries wil be saved to
    
    Parameters
    ----------            
    malignant_apps: listOfStrings, required
        Names of benign apps        
    benign_apps: listOfStrings, required
        Names of benign apps        
    malignant_fp: str, required
        The file path of the malicious apps
    benign_fp: str, required
        The file path of the benign apps
    multithreading: logical, required
        Boolean value to turn on multithreaded processing of data
    out_path: str, required
        File path of outputed parsed apps
    verbose: logical, required
        Boolean value to print progress while building dictionaries
    
    Returns
    -------
    None
    '''
    malignant_apps=kwargs["malignant_apps"]
    benign_apps=kwargs["benign_apps"]
    core_count=kwargs["core_count"]
    malignant_fp=kwargs["mal_fp"]
    benign_fp=kwargs["benign_fp"]
    multi_threading=kwargs["multi_threading"]
    out_path=kwargs["out_path"]
    verbose=kwargs["verbose"]
    
    if '.ipynb_checkpoints' in benign_apps:
        benign_apps.remove('.ipynb_checkpoints')
    if '.ipynb_checkpoints' in malignant_apps:
        malignant_apps.remove('.ipynb_checkpoints')

    start_time = time.time()

    #remove files already in folder
    for file in os.listdir(out_path):
        fp=os.path.join(out_path, file)
        os.remove(fp)

    print("--- Begin Parsing Benign and Malicious Apps ---")
    confirm_exc = get_data.create_app_files(
        benign_fp=benign_fp,
        benign_app=benign_apps,
        mal_fp=malignant_fp,
        mal_app=malignant_apps,
        multi_threading=multi_threading,
        verbose=verbose, 
        out_path=out_path,
        core_count=core_count
    )
    
    if confirm_exc:
        if verbose:
            print("--- All Apps Parsed in " + str(int(time.time() - start_time)) + " Seconds ---")
            print()
    else:
        raise ValueError("ERROR get_data.create_app_files failed")

def build_dictionaries(**params):
    '''
    Function to build dictionaries of api calls:
    dict_A.json contains api calls indexed by the app they appear in
    dict_B.json contains api calls indexed by the method block they appear in
    dict_P.json contains api calls indexed by the package they appear in
    dict_I.json contains api calls indexed by the invocation type they appear in
    api_calls.json contains api calls with the number of times they appear in all apps 
    naming_key.json is a table to look up keys and their relative api calls, apps, code blocks, packages, or invocation types
    
    Parameters
    ----------
    dict_directory: dictionary, required
        File path of dictionary output
    out_path: str, required
        File path to get json files of api calls from parsed apps
    verbose: logical value, required
        Boolean value to print progress while building dictionaries
    truncate: logical value, required
        Boolean value to 
    
    '''
    fp=params["dict_directory"]
    print("--- Starting Dictionary Creation ---")
    start_time = time.time()
    dict_B, dict_P, dict_I, dict_A = dict_builder.fast_dict(**params)
    for t,fname in zip([dict_A, dict_B, dict_P, dict_I],["dict_A", "dict_B", "dict_P", "dict_I"]):
        json_functions.save_json(t, fp+fname)       
    print("--- Dictionary Creation Done in " + str(int(time.time() - start_time)) + " Seconds ---")
    print()

def make_graph(src, dst):
    print()
    print("--- Starting StellarGraph Creation ---")
    start_time = time.time()
    G=build_network.make_stellargraph(src)
    print(G.info())
    print("--- StellarGraph Creation Done in " + str(int(time.time() - start_time)) + " Seconds ---")
    print()
    # list of all node types in the graph
    node_types = ["api_call_nodes", "package_nodes", "app_nodes", "block_nodes"]
    node_dict = {type:len(G.nodes_of_type(type)) for type in node_types}
    json_functions.save_json(node_dict, dst)
    return G

def run_shne(**params):
    #Start SHNE 
    print("--- Starting SHNE ---")
    s_app = time.time()
    SHNE.run_SHNE(**params)
    print("--- SHNE Embedding Layer Created in " + str(int(time.time() - s_app)) + " Seconds ---")

def run_eda(filename):
    '''
    Function to run eda.

    Parameters
    ----------
    verbose: logical, optional. Default True
        If true print updates on eda. Do not print updates otherwise
    limit: int, optional. Default None
        If not `None` then limit apps by that ammount 
    multiprocessing: logical, optional. Default True
        If true run with multiprocessing. Run in serial otherwise
    data_naming_key: str, required
        Path to json file lookup table of node code for repectives <app>, <block>, <api_call>, and <package>
        strings    
    data_extract_loc: str, required
        Path of parsed smali code in json files
    Returns
    -------
    None. Will save notebook of this eda run in the path specified in 
    '''
    print("---RUNNING EDA---")
    file_types=["pdf", "html"]
    cmd="jupyter nbconvert --to notebook %s"%filename
    output=os.popen(cmd)
    print(output)
    for file in file_types:
        cmd="jupyter nbconvert --to %s %s"%(file, filename)
        output=os.popen(cmd)
        print(output)
#     malware, benign=eda.get_node_data(
#         verbose=kwargs["verbose"],
#         lim=kwargs["limit"], 
#         multiprocessing=kwargs["multiprocessing"],
#         key_fn=kwargs["data_naming_key"],
#         src=kwargs["data_extract_loc"]
#     )
#     return malware, benign
    

def run_all(kwargs):
    '''
    Runs the main project pipeline logic, given the targets.
    
    Parameters
    ----------
    cmd_line_args: Dictionary, required
        Arguments passed on in command lines:
            test: run on test est
            node2vec_walk: perfrom node2vec walk instead of word2vec
            embeddings_only: only get word2vec/node2vec embeddings
            skip_embeddings: skip word2vec/node2vec embeddings creation
            skip_shne: skip shne model creation
            parse_only: Only get api dictionaries. Do not create embeddings or models
            overwrite: Overwrite any data that may already exist in out folder
            redirect_std_out: Save cmd line output to text file. Hides console outputs
            time: time how long to run
    params: Dictionary, required
        dictionary of parameters in found in config file in `config/params.json`
            mal_fp: string
                file path of the malignant apps
            benign_fp: string
                file path of the benign apps
            limiter: bool
                Boolean value to dictate if number of apps parsed is to be limited
            lim_mal: int
                Number of malignant apps to limit parsing of if limiter is True 
            lim_benign: int
                Number of benign apps to limit parsing of if limiter is True
            mal_fp_test_loc: string
                file path of test malignant apps            
            benign_fp_test_loc: string
                file path of benign apps
            directory: string
                File path to get json files of api calls from parsed apps
            verbose: bool
                Boolean value to print progress while building dictionaries
            truncate: bool
                Boolean value to
            dict_directory: string
                File path of dictionary output
            multithreading: bool
                Boolean value to turn on multithreaded processing of data    
            out_path: string
                File path of outputed parsed apps            
            verbose: bool
                Boolean value to print progress while building dictionaries            
    '''
    params=kwargs["params"]
    cmd_ln_args=kwargs["cmd_line_args"]

    etl_params=params["etl-params"]
    w2v_params=params["word2vec-params"]

    SKIP_SHNE=cmd_ln_args["skip_shne"]
    EMBEDDINGS_ONLY=cmd_ln_args["embeddings_only"]
    PARSE_ONLY=cmd_ln_args["parse_only"]
    OVERWRITE=cmd_ln_args["overwrite"]
    SAVE_OUTPUT=cmd_ln_args["redirect_std_out"]

    CHECK_FILES=params["check_files"]
    CHECK_FILES2=params["check_files2"]
    SHNE_PATH=params["shne-params"]["datapath"]
    VERBOSE=params["verbose"]
    NUM_CORES=params["core_count"]
    MULTIPROCESS=params["multithreading"]

    DICTIONARY_EXTRACT_DIR=etl_params["dict_directory"]
    ETL_LIMITER=etl_params["limiter"]
    ETL_MALICIOUS_PATH=etl_params["mal_fp"]
    ETL_BENIGN_PATH=etl_params["benign_fp"]
    ETL_LIM=etl_params["lim_apps"]
    ETL_OUT_PATH=etl_params["out_path"]
    ETL_TRUNCATE=etl_params["truncate"]
    API_BOUND=etl_params["lower_bound_api_count"]
    NAMING_KEY=etl_params["data_naming_key_filename"]
    API_CALLS_TABLE=etl_params["api_call_filename"]
    
    #run shne if preprocessing is done
    data_exists=os.path.isdir(SHNE_PATH)
    preprocessing_done=data_exists and all([cf in os.listdir(SHNE_PATH) for cf in CHECK_FILES2])

    #skip preprocessing and just run SHNE
    cmd_line_shne=EMBEDDINGS_ONLY or PARSE_ONLY or OVERWRITE
    if preprocessing_done and not SKIP_SHNE and not cmd_line_shne:
        print("PREPROCESSING DONE, STARTING SHNE")
        run_shne()
        return

    # if app files have already been parsed, then skip dictionary creation
    directory_exists=os.path.isdir(DICTIONARY_EXTRACT_DIR) 
    if directory_exists:
        all_files_in_directory=all([cf in os.listdir(DICTIONARY_EXTRACT_DIR) for cf in CHECK_FILES])
        app_dicts_already_created=directory_exists and all_files_in_directory
    else:
        app_dicts_already_created=False
    
    if app_dicts_already_created and not OVERWRITE:
        print("--- DICTIONARIES ALREADY CREATED, STARTING STELLARGRAPH CREATION ---")

    else:
        #start extracting smali code
        mal_app_names, benign_app_names=get_app_names(
            limiter=ETL_LIMITER,
            mal_fp=ETL_MALICIOUS_PATH,
            benign_fp=ETL_BENIGN_PATH,
            lim_benign=ETL_LIM,
            lim_mal=ETL_LIM,
            verbose=VERBOSE
        )
        dictionary_verbose=VERBOSE and not SAVE_OUTPUT
        create_dictionary(
            malignant_apps=mal_app_names,
            benign_apps=benign_app_names, 
            core_count=NUM_CORES,
            verbose=dictionary_verbose,
            mal_fp=ETL_MALICIOUS_PATH,
            benign_fp=ETL_BENIGN_PATH,
            multi_threading=MULTIPROCESS,
            out_path=ETL_OUT_PATH
        )
        build_dictionaries(
            dict_directory=DICTIONARY_EXTRACT_DIR,
            verbose=dictionary_verbose,
            out_path=ETL_OUT_PATH,
            truncate=ETL_TRUNCATE,
            lower_bound_api_count=API_BOUND,
            data_naming_key_filename=NAMING_KEY,
            api_call_filename=API_CALLS_TABLE
        )
        
    if PARSE_ONLY:
        if VERBOSE:
            print("Done.")
        return
    
    # get StellarGraph Network Graph
    sg_dst=os.path.join(params["shne-params"]["datapath"], params["shne-params"]["node_counts_filename"])
    G=make_graph(DICTIONARY_EXTRACT_DIR, sg_dst)
           
    if cmd_ln_args["node2vec_walk"]:
        #generate node2vec random walks
        node2vec.node2vec_walk(G, params["node2vec-params"])
        params["shne-params"]["datapath"]=params["node2vec-params"]["save_dir"]
    else:
        # generate metapath2vec random walks
        metapath2vec.metapath2vec_walk(G, params["metapath2vec-params"])
        params["shne-params"]["datapath"]=params["word2vec-params"]["save_dir"]
    
    if EMBEDDINGS_ONLY:
        if VERBOSE:
            print("Done.")
        return
    #BRADEN
    unique_api_path=os.path.join(etl_params["data_naming_key_dir"], etl_params["data_naming_key_filename"])
    if not cmd_ln_args["skip_embeddings"]:
        print()
        word2vec.create_w2v_embedding( 
            path=etl_params["data_extract_loc"],
            path_to_unique_apis=unique_api_path,
            **w2v_params
        ) #Config 
    
    if not SKIP_SHNE:
        print()
        run_shne(**params)

    #save final parameters
    out={
        "params":params,
        "command_line_arguments": cmd_ln_args
    }
    out_fn=os.path.join(params["out_path"],"final_"+params["params_name"])
    json_functions.save_json(out, out_fn)
    return