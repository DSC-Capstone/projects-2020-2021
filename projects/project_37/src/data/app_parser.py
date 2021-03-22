import sys, os, re

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import find_apps
from p_tqdm import p_umap


API_DATA_COLUMNS =  ["app", "api", "invoke_type", "class", "method", "package", "context"]
PACKAGE_CLEANING_PATTERN = r"[$;].*"

class Application():
    """
    Defines a application/APK.
    """
    smali_class_pattern = r"L[\w/]*;"
    API_call_pattern = r"invoke-.*"
    
    def extract_app_name(self):
        return os.path.basename(self.app_dir)  
    
    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.app_name = self.extract_app_name()
        self.API_data = None
        self.apis = set()
        self.smali_list = []
        self.num_methods = 0
        
        
    def find_smali_filepaths(self):
        """
        Retrieves a list of paths to all smali files in the app directory. 
        Records paths in self.smali_list and returns them.
        """
        # reset current list in case
        self.smali_list = []
        
        for result in os.walk(self.app_dir):
            current_dir = result[0]
            files = result[2]
            for filename in files:
                smali_ext = '.smali'
                if filename[-len(smali_ext):] == smali_ext:
                    self.smali_list.append(os.path.join(current_dir, filename))

        return self.smali_list
    
    def parse_smali(self, filepath):
        """Parses a singluar smali file
        
        filepath: str, path to smali file"""
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if lines:
            # get class name
            current_class = lines.pop(0).split()[-1]

            # scan for code blocks and API calls
            line_iter = iter(lines)
            current_method = ""
            apis_in_method = set()
            for line in line_iter:
                if ".method" in line:
                    current_method = current_class + "->" + line.split()[-1]
                    self.num_methods += 1
                elif "invoke-" in line:
                    split = line.split()
                    invoke_type = (
                        split[0]
                        .split("-")[-1] # remove invoke
                        .split("/")[0] # remove "/range"
                    )
                    api_call = split[-1]
                    self.apis.add(api_call)
                    package = re.sub(PACKAGE_CLEANING_PATTERN, "", api_call)
                    context=line.strip()

                    self.API_data.append([self.app_name, api_call, invoke_type, current_class, current_method, package, context])
    
    def parse(self):
        """
        Parses all smali files within the app.
        """
        self.API_data = []
        
        for file_path in self.find_smali_filepaths():
            self.parse_smali(file_path)
            
        api_data = pd.DataFrame(self.API_data, columns=API_DATA_COLUMNS)
            
        return api_data

            
def get_data(outfolder, data_source=None, nprocs=2, recompute=False):
    '''
    Retrieve data for year/location/group from the internet
    and return data (or write data to file, if `outfolder` is
    not `None`).
    '''
    # setup
    os.makedirs(outfolder, exist_ok=True)
    app_data_path = app_heap_path = os.path.join('data', 'out', 'all-apps', 'app-data')
    os.makedirs(app_data_path, exist_ok=True)
    app_to_parse_path = os.path.join(outfolder, 'app_list.csv')  # location of any predetermined apps

    try:  # search for predetermined list of apps
        apps_df = pd.read_csv(app_to_parse_path)
    except FileNotFoundError:  # if no such file, create one by looking for apps under data_source directory
        apps_df = find_apps(data_source)
        apps_df.to_csv(app_to_parse_path)
        
    def parse_app(app_dir, outfolder):
        app = Application(app_dir)
        outpath = os.path.join(app_data_path, app.app_name+".csv")
        if os.path.exists(outpath) and not recompute:
            return
        else:
            data = app.parse()
            if data.shape[0] == 0:
                print(f'No data for {app.app_name}', file=sys.stdout)
                return
            else:
                data.to_csv(outpath, index=False)

    print("STEP 1 - PARSING APPS")
    # concurrent execution of smali parsing
    app_parser = p_umap(parse_app, 
                        apps_df.app_dir,
                        [outfolder]*len(apps_df.app_dir),
                        num_cpus=nprocs)
