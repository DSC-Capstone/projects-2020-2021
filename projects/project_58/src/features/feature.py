#!/usr/bin/env python
import re
import pandas as pd
import os
import numpy as np

# Transform smali file to api-call anlysis table
def process_smali(sml):
    methods = pd.DataFrame(re.findall(r'method.* (\w+)[(].+[)].+;([\d\D]*?)\.end method', sml))
    if len(methods) == 0:
        return -1

    def process_method(x):
        res = pd.DataFrame(re.findall(r'invoke-(\w{5,9})\s.+}, (.*);->(.+)[(]', x[1]))
        if len(res) == 0:
            return
        res.columns = ['invoke_type', 'package_long', 'call']
        res['method'] = x[0]
        res['package'] = res.package_long.apply(lambda x:x[1:x.find('/')])
        res['type'] = res.package.apply(lambda x:x if x in ['android', 'androidx', 'google', 'java', 'javax', 'kotlin'] else 'self')
        return res

    dfs = methods.apply(process_method, axis=1)
    try:
        return pd.concat(dfs.tolist())
    except:
        return -1


# Apply smali analysis to all smali files within an apk
def process_apk(path, apk_name):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.smali'):
                f = open(os.path.join(root, name))
                sml_df = process_smali(f.read())
                if type(sml_df) == int:
                    pass
                else:
                    df = pd.concat([df, sml_df], ignore_index=True)
                f.close()
    df['name'] = apk_name
    return df


# Parse all training data in directory which contains all malware or benign-ware
def parse_all(path, is_mal):
    fin = pd.DataFrame()
    wares = [i for i in os.listdir(path)]
    if is_mal:
        for d in wares:
            d_path = path + '/' + d
            varieties = [i for i in os.listdir(d_path)]
            for v in varieties:
                v_path = d_path + '/' + v
                try:
                    df = process_apk(v_path, d + ' ' + v)
                    fin = pd.concat([fin, df], ignore_index=True)
                except:
                    pass
    else:
        for d in wares:
            d_path = path + '/' + d
            try:
                df = process_apk(d_path, d)
                fin = pd.concat([fin, df], ignore_index=True)
            except:
                pass
    return fin


# Generate Markov Chain
def generate_chain(df):
    return df.type.value_counts() / len(df)


# Generate Markov chains dataset
def generate_chains(features_mal, features_benign):
    # Extract api-call Statistics from parsed data
    format_ser = pd.Series([0] * 6,['self','java','android','kotlin','androidx','javax'])
    def form(ser):
        return ser.combine(format_ser, max).fillna(0)
    def proc(x):
        return form(x.type.value_counts())
    # Prepare Training data
    chains_mal = features_mal.groupby('name').apply(proc)
    chains_benign = features_benign.groupby('name').apply(proc)
    chains_mal['is_mal'] = [1] * len(chains_mal)
    chains_benign['is_mal'] = [0] * len(chains_benign)
    return pd.concat([chains_mal, chains_benign], ignore_index=True)


# Standardize X
def standardize_X(chain):
    return chains[['android','androidx','java','javax','kotlin','self']].apply(lambda x:x/sum(x),axis=1)


# Generate X, y
def generate_Xy(chains):
    X = chains[['android','androidx','java','javax','kotlin','self']]
    y = chains.is_mal
    return X, y
