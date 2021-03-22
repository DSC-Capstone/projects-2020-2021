import re, os
import json
import time
import pandas as pd
import numpy as np
import tqdm
import threading
from src.data_creation.explore import printProgressBar
from multiprocessing import Pool, cpu_count


def create_app_files(**kwargs):
    """
    Parses and saves as JSON all supplied apps both benign and mal. Uses multi threading to speed up this process
    
    Parameters
    ----------
    benign_fp : str
        Filepath to benign apps
        
    benign_app : str
        Name of benign app
        
    mal_fp : str
        Filepath to mal apps
        
    mal_app : str
        Name of mal app
        
    multi_threading : boolean
        Boolean value to turn on multithreaded processing of data
        
    Returns
    -------    
    boolean
        True if confirmation of successful excecution, otherwise returns None
        
    
    """    
    benign_fp=kwargs["benign_fp"]
    benign_app=kwargs["benign_app"]
    mal_fp=kwargs["mal_fp"]
    mal_app=kwargs["mal_app"]
    multi_threading=kwargs["multi_threading"]
    verbose=kwargs["verbose"]
    out_path=kwargs["out_path"]
    core_count=kwargs["core_count"]
    
    start = time.time()
    
    app_list = []
    for b in benign_app:
        app_list.append(benign_fp + "/"+b)
    for m in mal_app:
        app_list.append(mal_fp + "/"+m)
        
    tot_length=len(app_list)
    #Multithreading brain builds eight apps at the same time
    if multi_threading == True:
        print("Multithreading Enabled, running on %i processes"%core_count)
        apps=np.array([app_list, [benign_app]*tot_length, [out_path]*tot_length], dtype=object).T
        with Pool(processes=core_count) as pool:
            if verbose:
                for _ in tqdm.tqdm(pool.map(get_json, apps), leave=True): pass
            else:
                pool.map(get_json, apps)
    else:
        if verbose:
            print("Multithreading Disabled, running only single process")
        for i in tqdm.tqdm(app_list):
                get_json([i,benign_app, out_path])               
                
    tot_seconds=int(time.time() - start)
    minutes=tot_seconds/60
    seconds=tot_seconds%60
    if verbose:
        print("Parsed %i apps in %i minutes, %i seconds"%(len(app_list), minutes, seconds))
    return True


def get_json(args):
    """Creates json file for app in 'matricies/data_extract' folder that contains a list of lists, 
    where each inner list is a codeblock of methods 
    
    :param path : str
        Filepath to app directory
    """
    start = time.time()

    path=args[0]
    benign_app=args[1]
    out_path=args[2]

    if path.split("/")[-1] in benign_app:
        typer = "_B_"
    else:
        typer = "_M_"
    
    temp = ""
    
    
    all_rel_info = [] # list of lists with each inner list being a codeblock of methods
    
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if "checkpoint" not in name:
                t = str(os.path.join(root, name))
                if t[-6:] == ".smali":
                    temp = temp + str(open(t, "r").read())
                    
    code_blocks = temp.split(".end method")
    for bloc in code_blocks:
        all_rel_info.append(re.findall('invoke-.+', bloc))
    
    os.makedirs(out_path, exist_ok=True)
    with open(out_path + path.split("/")[-1] +typer+'extract.json', 'w') as f:
        json.dump(all_rel_info, f)
    return