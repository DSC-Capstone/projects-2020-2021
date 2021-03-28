### Extracts data from smali files
import json, os

def load_progress(json_obj):
    '''
    `object_hook` function to show progress bar in `save_json`.
    WARNING: showing progress bar will hurt read performance
    
    Parameters
    ----------
    json_obj: json object, required
        json object being loaded to show progress bar for.
    Returns
    -------
    None
    '''
    value = json_obj.get("features")
    if value:
        pbar = tqdm(value)
        for item in pbar:
            pass
            pbar.set_description("Loading")
    return json_obj
    

def save_json(dictionary, filename):
    """
    Saves python dictionary into json file
    
    Parameters
    ----------
    dictionary: dict, required
        Dictionary to save
    filename: str, required
        Name to give json file
    Returns
    -------
    None
    """
    fp="/".join(filename.split("/")[:-1])
    
    try:
        os.makedirs(fp, exist_ok=True)
    except FileNotFoundError:
        pass
    
    if ".json" not in filename:
        filename=filename + ".json"
        
    with open(filename, "w") as outfile:
        json.dump(dictionary, outfile)
    return "Dictionary Saved"


def load_json(json_file, hook=None):
    """
    Loads json file into python dictionary and returns dictionary
    
    Parameters
    ----------
    json_file: str, required
        File containing dictionary to load 
    hook: python function, optional. Default `None`
        If not `None` then should be set to <object_hook> method.
        For example `load_progressbar` will display a progress bar while loading jsoin
        If None apply no <object_hook> method
    """
    
    with open(json_file, "r") as file:
        dictionary = json.load(file, object_hook=hook)
    return dictionary 