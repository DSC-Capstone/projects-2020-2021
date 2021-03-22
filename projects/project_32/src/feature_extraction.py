import os,glob
import re
from pathlib import Path
import pandas as pd 
import numpy as np
from collections import defaultdict 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.sparse import coo_matrix

def process_ids(def_dict):
    dct = {}
    dct_app = {}
    counter = 0
    counter_app = 0
    lst = []
    lst_app = []
    
    for i in def_dict:
        for j in def_dict[i]:
            lst = lst + list(def_dict[i][j]['Combined']['APIs'])
    lst = list(set(lst))
    for i in lst:
        dct[i] = counter
        counter = counter + 1 
    for app_type in def_dict:
        lst_app = lst_app + list(def_dict[app_type])
    for i in lst_app:
        dct_app[i] = counter_app
        counter_app = counter_app + 1
    return [dct,dct_app]

def a (def_dict, id_api, id_app):
    row_1 = []
    col_1 = []
    counter_list = []
    for i in def_dict:
        for j in def_dict[i]:
            for k in list(def_dict[i][j]['Combined']['APIs']):
                row_1.append(id_app[j])
                col_1.append(id_api[k])
                counter_list.append(1)
    return sparse.coo_matrix((counter_list, (row_1, col_1)))

def b (def_dict, id_api, id_app):
    row_1 = []
    col_1 = []
    counter_list = []
    
    for i in def_dict:
        for j in def_dict[i]:  
            try: 
                blocks = list(def_dict[i][j]['blocks'].keys())
            except:
                continue
            methodList = [[item for item in def_dict[i][j]['blocks'][block]] for block in blocks]
            for i in methodList:
                for j in i:
                    for k in i:
                        row_1.append(id_api[j])
                        col_1.append(id_api[k])
                        counter_list.append(1)

                        row_1.append(id_api[k])
                        col_1.append(id_api[j])
                        counter_list.append(1)
    return sparse.coo_matrix((counter_list, (row_1, col_1)))

           
def p(def_dict, id_api, id_app):

    row_1 = []
    col_1 = []
    counter_list = []
    
    for a in def_dict:
        for j in def_dict[a]:
            keylist = list(def_dict[a][j]['Packages'].keys())
            packageList = [[item for item in def_dict[a][j]['Packages'][key]] for key in keylist]

            for i in packageList:
                for j in i:
                    for k in i:
                        row_1.append(id_api[j])
                        col_1.append(id_api[k])
                        counter_list.append(1)

                        row_1.append(id_api[k])
                        col_1.append(id_api[j])
                        counter_list.append(1)

    return sparse.coo_matrix((counter_list, (row_1, col_1)))

def matrix_a(def_dict):
    id_api = process_ids(def_dict)[0]
    id_app = process_ids(def_dict)[1]
    matrix_a = a(def_dict, id_api, id_app)
    return matrix_a
def matrix_b(def_dict):
    id_api = process_ids(def_dict)[0]
    id_app = process_ids(def_dict)[1]
    matrix_b = b(def_dict, id_api, id_app)
    return matrix_b
def matrix_p(def_dict):
    id_api = process_ids(def_dict)[0]
    id_app = process_ids(def_dict)[1]
    matrix_p = p(def_dict, id_api, id_app)
    return matrix_p