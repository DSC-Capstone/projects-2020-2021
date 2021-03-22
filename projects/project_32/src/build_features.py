import pandas as pd 
import os,glob
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

#constants

def get_smali_files(mp,bp):
    
    #iterate through dir 
    malware_smali_list = []
    for subdir, dirs, files in os.walk(mp):
        for filename in files:
            filepath = subdir + os.sep + filename
            #find smali files
            if filepath.endswith(".smali"):
                malware_smali_list.append(filepath)
    benign_smali_list = []
    for subdir, dirs, files in os.walk(bp):
        for filename in files:
            filepath = subdir + os.sep + filename
            #find smali files
            if filepath.endswith(".smali"):
                benign_smali_list.append(filepath)

    return [malware_smali_list,benign_smali_list]
#helper methods
def create_set_api(def_dict, app_type, app_name):
    def_dict[app_type][app_name]['Combined']['APIs'] = list(set(def_dict[app_type][app_name]['Combined']['APIs']))

def create_set_package(def_dict, app_type, app_name):
    for i in def_dict[app_type][app_name]['Packages']:
        def_dict[app_type][app_name]['Packages'][i] = list(set(def_dict[app_type][app_name]['Packages'][i]))

def helper(app_type, app_name, elem, num_blocks, def_dict, invoke_type):
    tracking_number = 'method' + str(num_blocks)
    final = re.sub(re.compile('^[^}]*}'), '', str(elem))
    api_call = final[2:]
    apiNoParam = re.match(re.compile('[^(]*') , api_call)
    api_call = apiNoParam.group(0) + str('()')
    package = re.search(re.compile('^(.*?)->'), api_call)
    try:
        def_dict[app_type][app_name]['Packages'][package.group(1)].append(api_call)
    except:
        def_dict[app_type][app_name]['Packages'][package].append(api_call)
    def_dict[app_type][app_name]['Combined']['APIs'].append(api_call)
    def_dict[app_type][app_name]['invoke_type'][invoke_type].append(api_call)
    def_dict[app_type][app_name]['blocks'][tracking_number].append(api_call)

def create_struct(mp, bp):
    #constants
    
    num_blocks = 0
    constant = 1
    app_list = {}
    block_list = []
    flag = False
    tracking_number = 'method' + str(num_blocks)
    def_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    inside_block = False
    count = 0
    counter = 0
    api_list = ['invoke-static', 'invoke-virtual', 'invoke-direct', 'invoke-super', 'invoke-interface']
    app_list = []
    malware_smali_list = get_smali_files(mp,bp)[0]
    malware_seen = set()
    for i in malware_smali_list:
        file = open(i, "r")
        app_type = 'malware'
#     app_name = i.split('/')[7]
        app_name = i.split('/')[2]
        if app_name not in malware_seen:
            malware_seen.add(app_name)
        if len(malware_seen) > 40:
            break
        code = file.readlines()
        for index, elem in enumerate(code):
            if '.method' in elem:
                flag = True

            if inside_block and ('.end method' in elem):
                #Assign default values
                block_list = []
                flag = False
                num_blocks += 1

            if flag and (api_list[0] in elem):
                helper(app_type, app_name, elem, num_blocks, def_dict, api_list[0])
            if flag and (api_list[1] in elem):
                helper(app_type, app_name, elem, num_blocks, def_dict, api_list[1])
            if flag and (api_list[2] in elem):
                helper(app_type, app_name, elem, num_blocks, def_dict, api_list[2])
            if flag and (api_list[3] in elem):
                helper(app_type, app_name, elem, num_blocks, def_dict, api_list[3])
            if flag and (api_list[4] in elem):
                helper(app_type, app_name, elem, num_blocks, def_dict, api_list[4])
        create_set_api(def_dict, app_type, app_name)
        create_set_package(def_dict, app_type, app_name)
    #========Benign========#
    count = 0
    counter = 0 
    api_list = ['invoke-static', 'invoke-virtual', 'invoke-direct', 'invoke-super', 'invoke-interface']
    app_list = []
    benign_seen = set()
    benign_smali_list = get_smali_files(mp,bp)[1]
    for i in benign_smali_list:
        file = open(i, "r")
    #     app_type = i.split('/')[4]
        app_name = i.split('/')[2]
        if app_name not in benign_seen:
            benign_seen.add(app_name)
        if len(benign_seen) > 40:
            break
        code = file.readlines()
        for index, elem in enumerate(code):
            if '.method' in elem:
                flag = True

            if inside_block and ('.end method' in elem):
                #Assign default values
                block_list = []
                flag = False
                num_blocks += 1  

            if flag and (api_list[0] in elem): 
                helper('benign', app_name, elem, num_blocks, def_dict, api_list[0])
            if flag and (api_list[1] in elem):
                helper('benign', app_name, elem, num_blocks, def_dict, api_list[1])
            if flag and (api_list[2] in elem): 
                helper('benign', app_name, elem, num_blocks, def_dict, api_list[2])
            if flag and (api_list[3] in elem): 
                helper('benign', app_name, elem, num_blocks, def_dict, api_list[3])
            if flag and (api_list[4] in elem):
                helper('benign', app_name, elem, num_blocks, def_dict, api_list[4])
        create_set_api(def_dict, app_type, app_name)
        create_set_package(def_dict, app_type, app_name)
#     return (pd.Series(def_dict).head())
    return [def_dict, malware_seen, benign_seen]

if __name__ == '__main__':
    API_create_struct()
    

