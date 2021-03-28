import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from src.util import *

invoke_dict = {'static':0,'virtual':1,'direct':2,'super':3,'interface':4}

def smali_data(path):
    """
    Find all the smali files within one app 

    @path: path to the app file 
    @return: dict {"smali_file_path:text"}
    """
    smali_data = {} 
    path = str(path)
    walk_list = os.walk(path)

    # walk through find .smali files 
    for i in walk_list:
        root = i[0]
        for j in i[2]:
            if len(re.findall(r'\.smali$', j)) > 0:
                p = root+'/'+j
                with open(p, 'r') as file:
                    s = file.read()
                    smali_data[p] = s
                    
    return smali_data


def find_app_address(path):
    """
    find all the app paths within the input folder 
    """
    l = []
    if os.path.isdir(path):
        if os.listdir(path) == ['smali'] or 'smali' in os.listdir(path):
            return path
        else:
            for p in os.listdir(path):
                l.append(find_app_address(path + '/' + p))
                
    return list_flatten(l)

def stat_lst(path, app_id, ClassId, PackageId, ApiId, ClassTypeId, ReturnId, MethodId):
    """
    get nested list, where each inner list correspond to one api 

    @path: path to app file 
    """
    
    # get smali file content 
    data = smali_data(path)
    
    result = []

    for _, k in enumerate(data):
        
        v = data[k]
        
        # add class name into class names list
        try:
            classes = re.findall(r'\.class.*;', v)[0].split(' ')
            class_name = classes[-1][:-1]
            if len(classes) > 2:
                class_type = ' '.join(classes[1:-1])
            else:
                class_type = 'default'
                
        except:
            print("path",k)
            continue

        class_id = ClassId.add(class_name)
        class_type_id = ClassTypeId.add(class_type)

        # code block id
        block_id = -1

        # find all the code blocks 
        blocks = re.findall(r'\.method(.+)\s+\.([\s\S]+?)\.end\smethod', v)
        
        
        for b in blocks:
            
            # each code block has an unique id 
            block_id += 1
            
            # find method type
            method_type = ' '.join(b[0].split(' ')[1:-1])
            method_id = MethodId.add(method_type)
            
            
            # find all the api_calls in one block  
            api_calls = re.findall(r'invoke-(\w+) {.+}, (.+);->(.+)\(.*\)(.+);',b[1])   
            
            # in each api_calls extract the detailed information 
            for api in api_calls:
                
                invoke_type, package, api_name, return_type = api
                package_id = PackageId.add(package)
                api_id = ApiId.add(api_name)
                return_id = ReturnId.add(return_type)
                
                # row for each api 
                api_row = [
                    class_id, 
                    api_id, 
                    package_id, 
                    invoke_dict[invoke_type], 
                    block_id, 
                    app_id, 
                    class_type_id, 
                    return_id, 
                    method_id
                ]
                result.append(api_row)
                
    return np.array(result)

def stat_df(path_lst, app_type, output_path, 
    AppId, ClassId, PackageId, ApiId, ClassTypeId, ReturnId, MethodId):
    """
    Save cleaned dataframe to csv file 
    """

    # initialize the DataFrame 
    columns = [
        'class_id', 'api_id', 'package_id', 
        'invoke_type', 'block_id', 'app_id',
        'class_type_id', 'return_id', 'method_id'
    ]
    df = pd.DataFrame(columns=columns)
    
    # go over each app 
    for p in tqdm(path_lst):

        # find app name 
        app_name = re.findall(r'\/([^\/]+)$', p)[0]

        # update AppId 
        app_id = AppId.add(app_name)

        api_array = stat_lst(p, app_id, ClassId, PackageId, ApiId, ClassTypeId, ReturnId, MethodId)

        try:
            df = pd.concat((df,pd.DataFrame(api_array, columns=columns)), ignore_index=True)
        except:
            print(api_array)
        
    df.to_csv(f'{output_path}/{app_type}.csv')
    


def clean_df(input_path, app_type, output_path, sampling_lst):
    """
    method to get a cleaned csv with 6 columns 
    [class, api_name, package, invoke_type, code_block #, app name]
    """
    
    # create app_id, package_id, api_id to track unique app, package and api among all the apps 
    AppId = UniqueIdGenerator(name='app_id', output_path=output_path)
    ClassId = UniqueIdGenerator(name='class_id', output_path=output_path)
    PackageId = UniqueIdGenerator(name='package_id', output_path=output_path)
    ApiId = UniqueIdGenerator(name='api_id', output_path=output_path)
    ClassTypeId = UniqueIdGenerator(name='class_type_id', output_path=output_path)
    ReturnId = UniqueIdGenerator(name='return_id', output_path=output_path)
    MethodId = UniqueIdGenerator(name='method_id', output_path=output_path)
    

    for i in range(len(input_path)):
        p, t, s = input_path[i], app_type[i], sampling_lst[i]

        # find all the app paths within the directory 
        path_lst = find_app_address(p)

        # sampling 
        path_lst = np.random.choice(path_lst, s, replace=False)

        # create the .csv file for each category 
        stat_df(
            path_lst = path_lst, 
            app_type = t, 
            output_path = output_path, 
            AppId = AppId, 
            ClassId = ClassId,
            PackageId = PackageId, 
            ApiId = ApiId,
            ClassTypeId = ClassTypeId,
            ReturnId = ReturnId,
            MethodId = MethodId
        )

    AppId.save_to_csv()
    ClassId.save_to_csv()
    PackageId.save_to_csv()
    ApiId.save_to_csv()
    ClassTypeId.save_to_csv()
    ReturnId.save_to_csv()
    MethodId.save_to_csv()
   
        










