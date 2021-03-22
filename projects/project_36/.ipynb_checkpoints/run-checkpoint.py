import src.main as Main #runs everything
import json
import getopt
import sys
import os
import time
import pprint
from src.data_creation import json_functions as jf

def update_paths(test, params):
    '''
    Function to set the outpath for entire run

    Paramters
    ---------
    test: bool, required
        If true change app location to test app location
    args: dictionary, required
        The key worded arguments passed in `config/params.json`
    Returns
    -------
    <args> with updated paths
    '''
    if test:
        params["out_path"]=params["test_out_path"]
        params["etl-params"]["mal_fp"]=params["etl-params"]["mal_fp_test"]
        params["etl-params"]["benign_fp"]=params["etl-params"]["benign_fp_test"] 
    
    out=params["out_path"]

    verbose=params["verbose"]

    params_to_change=params["update_dirs"]
    verbose_params=params["update_verbose"]

    for key in params_to_change.keys():
        for path in params_to_change[key]:
            new_path=os.path.join(out,params[key][path])
            os.makedirs(new_path, exist_ok=True)
            params[key][path]=new_path

    
    for key in verbose_params:
        for value in verbose_params[key]:
            params[key][value]=verbose
            
    shne_path=params["shne-params"]["model_path"]
    shne_filename=params["shne-params"]["model_filename"]
    params["eda-params"]["full_model_path"]=os.path.join(shne_path, shne_filename)   
    return params

def get_tags(args, tags):
    '''
    Helper function to determine any of <tags> are in <args> passed in command line
    
    Parameters
    ----------
    args: listOfStrings, required
        A list of string command line arguments 
    tags: listOfStrings, required
        A list of string tags to test if they are contained in <tags>
        
    Returns
    -------
    True if any element of <tags> is present in <args>. False otherwise
    '''
    return any([t in args for t in tags])

def get_command_ling_args(args_passed, all_args):
    '''
    Function to get command line arguments

    Parameters
    ----------
    args_passed: listOfStrings, requried
        list of command line arguments passed after `-eda`
    all_args: dictionary, required
        all possible command line arguments from <eda-params> in `config/params.json` 
    Returns
    -------
    Dictionary of all possible command line arguments fouind in <all_args> as keys and either TRUE if they were 
    passed in the command line, or FALSE otherwise
    '''
    cmd_args={}
    for arg in list(all_args.keys()):
        cmd_args[arg]=get_tags(args_passed, all_args[arg])
    return cmd_args

def time_func(func, arguments):
    '''
    Function to time the runtime of another function, <func>.
    Time to run is printed to standard output

    Parameters
    ----------
    func:  pythonFunction, required
        the function to time
    arguments: dictionary, required
        dictionary of the key worded arguments to pass onto <func>
    Returns
    -------
    None    
    '''
    start=time.time()
    func(arguments)
    runtime=time.time()-start

    seconds=runtime%(60)
    minutes=runtime//60%60
    hours=runtime//3600

    print("%s ran in %i hours %i minutes and %i seconds"%(func.__name__, hours,minutes,seconds))
    print()

def get_eda_args(args_passed, all_args):
    '''
    Function to get arguments passed with `-eda` in command line

    Parameters
    ----------
    args_passed: listOfStrings, requried
        list of command line arguments passed after `-eda`
    all_args: dictionary, required
        all possible command line arguments from <eda-params> in `config/params.json` 
    Returns
    -------
    Dictionary of all possible command line arguments fouind in <all_args> as keys and either TRUE if they were 
    passed in the command line, or FALSE otherwise.
    The value for the <limit> in returned dictionart <args> will be either FALSE or an integer 
    value to limit eda apps by.
    '''
    def is_positive_integer(val):
        if val[0]!="-":
            return val.isdigit()
        return False

    args={}
    test_args=args_passed[1:]
    idx=0
    for arg in all_args.keys():
        args[arg]=False
        
    for arg in test_args:        
        if arg in all_args.keys():            
            argument_str=all_args[arg]
            if get_tags(test_args, argument_str):
                if arg=="limit":
                    idx+=1
                    print("args_passed", test_args, "idx", test_args[idx])
                    if is_positive_integer(test_args[idx]):
                        args[arg]=int(test_args[idx])
                    else:
                        raise ValueError("Must pass positive integer after <time> argument!")
                        sys.exit()
                else:
                    args[arg]=True
        else:
            raise ValueError("Invalid value after '-eda'")
            sys.exit()
        idx+=1
    print(args)
    return args

def apply_command_line_args(args, params):
    '''
    Function to apply command line arguments
    Parameters
    ----------
    args: dictionary, required
        dictionary of command line arguments passed
    params: dictionary, required
        parameter configuration dictionary pulled from `config/params.json`
    Returns
    -------
    None if the following parameters are passed:
        <-eda> and <only> will run the eda and then exit
        <-time> will time `run_all()` in `main.py` and exit
    Otherwise returns updated <params> dictionary 
    '''
    TEST=args["test"]
    EDA=args["eda"]
    NODE2VEC=args["node2vec_walk"]
    EMBEDDINGS_ONLY=args["embeddings_only"]
    SKIP_EMBEDDINGS=args["skip_embeddings"]
    SKIP_SHNE=args["skip_shne"]
    SILENT=args["silent"]
    PARSE_ONLY=args["parse_only"]
    OVERWRITE=args["overwrite"]
    LOG=args["redirect_std_out"]
    TIME=args["time"]
    FORCE_MULTI=args["force_multi"]
    FORCE_SINGLE=args["force_single"]
    SHOW_PARAMS=args["show_params"]

    VERBOSE=params["verbose"]

    if SHOW_PARAMS:
        print("Running with current parameters:")
        pprint.pprint(params)

    if SILENT:
        VERBOSE=False
        params["verbose"]=False

    #set outpaths for arguments
    params=update_paths(TEST, params)

    if FORCE_SINGLE and FORCE_MULTI:
        raise ValueError("Pass either `--force-single` or `--force-multi`. Cannot pass both.")

    if FORCE_MULTI:
        if VERBOSE:
            print("Multiprocessing Enabled")
        params["multithreading"]=True
    
    if FORCE_SINGLE:
        if VERBOSE:
            print("Multiprocessing Disabled")
        params["multithreading"]=False

    if LOG:
        fp=os.path.join(params["out_path"], params["log_filename"])
        if VERBOSE:
            print("Saving output to %s"%fp)
        sys.stdout = open(fp, 'w')
        sys.stderr = sys.stdout

    if EDA:
        if VERBOSE:
            print("Running EDA")
        eda_params=params["eda-params"]

        args_passed=eda_params["args_literal"]
        eda_idx=any([args_passed.index(tag) for tag in params["options"]["eda"]])
        eda_args=args_passed[eda_idx:]
       
        eda_args_passed=get_eda_args(eda_args, eda_params["command_line_options"])

        if isinstance(eda_args_passed["limit"], int):
            limit_eda=eda_args_passed["limit"]
        else:
            limit_eda=eda_params["limit_eda"]

#         eda_params={
#             "verbose":params["verbose"],
#             "limit":limit_eda, 
#             "multiprocessing":params["multithreading"],
#             "data_naming_key":eda_params["data_naming_key_filename"],
#             "data_extract_loc":eda_params["data_extract_loc"],
#         }
        
        filepath=os.path.join(eda_params["eda_dir"],eda_params["eda_notebook"])
        if eda_args_passed["time"]:
            if VERBOSE:
                print("Timing EDA run")
            time_func(Main.run_eda, filepath)
        else:
            Main.run_eda(filepath)
        sys.exit()

    # time how long to run
    if TIME:
        if VERBOSE:
            print("Timing `Main.run_all`")
        arguments={
            "cmd_line_args":args,
            "params":params
        }
        time_func(Main.run_all, arguments)
        sys.exit()

    return params

def run(cmd_line_args, params):
    '''
    Function to run entire project

    Parameters
    ----------
    cmd_line_args: listOfStrings, required
        list of command line arguments passed
    params: dictionary, required
        parameter configuration dictionary pulled from `config/params.json`
    Returns
    -------
    None
    '''  
    print()
    #get and apply command line arguments
    args_params=params["options"]
    cmd_line_args_dict=get_command_ling_args(cmd_line_args, args_params)
    params["eda-params"]["args_literal"]=cmd_line_args
    params=apply_command_line_args(cmd_line_args_dict, params)

    out_fn=os.path.join(params["out_path"],params["params_name"])
    jf.save_json(params, out_fn)

    kwargs={
        "cmd_line_args":cmd_line_args_dict,
        "params":params
    }
    
    Main.run_all(kwargs)
    print()    
    sys.exit()

if __name__=="__main__":

    args = sys.argv[1:]
    data_params=jf.load_json("config/params.json")
    run(args, data_params)