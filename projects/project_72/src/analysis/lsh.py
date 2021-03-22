from snapy import MinHash, LSH
import re
import numpy as np 
from format_data_reddit import make_content_text

import copy
def create_lsh(content, n_permutations, n_gram):
    labels = content.keys()
    values = content.values()
    #Create MinHash object
    minhash = MinHash(values, n_gram=n_gram, permutations=n_permutations, hash_bits=64, seed=3)
    #Create LSH model
    lsh = LSH(minhash, labels, no_of_bands=5)
    return lsh

def create_lsh_term(content_term,n_permutations, n_gram):
    lsh_ls = {}
    for i in content_term:
        lsh_ls[i] = create_lsh(content_term[i], n_permutations, n_gram)
    return lsh_ls
def update_lsh_text(lsh_ls,content_text,n_permutations, n_gram):
    for i in content_text:
        if i in lsh_ls:
            labels = content_text[i].keys()
            labels = [i+'test' for i in labels]
            values = content_text[i].values()
            minhash = MinHash(values, n_gram=n_gram, permutations=n_permutations, hash_bits=64, seed=3)
            lsh_ls[i].update(minhash,labels)
        else:
            lsh_ls[i] = create_lsh(content_text[i], n_permutations, n_gram)
    return lsh_ls
def process_all(lsh_ls_term,df,QUERY,n_permutations, n_gram):
    sections = df.shape[0]//1000
    ls_total = []
    for i in range(sections+1):
        dic={QUERY:df[i*1000:(i+1)*1000]}
        content_text = {}
        lsh_ls = copy.deepcopy(lsh_ls_term)
        make_content_text(content_text,dic)
        ls_pre = get_similar_ls(lsh_ls,content_text,n_permutations, n_gram)
        ls_total += ls_pre
    return ls_total
def get_similar_ls(lsh_ls_term,content_text,n_permutations, n_gram):
    lsh_ls = update_lsh_text(lsh_ls_term,content_text,n_permutations, n_gram)
    edge_list = {}
    for i in lsh_ls:
        edge_list[i] = lsh_ls[i].edge_list(jaccard_weighted=True)
    
    similar_ls = []
    for e in edge_list:
        edges = edge_list[e]
        for i in edges:
            if type(i[1])==int and type(i[0])==str:
                similar_ls.append(i)
            elif (type(i[0])==int and type(i[1])==str):
                similar_ls.append(i)
    return similar_ls
def update_df(ls,df,terms,QUERY):
    term_dic={}
    score_dic={}
    l = len(QUERY+'_post_')
    for n in ls:
        word_id = n[0] 
        post_id = int(re.sub('_.+','',word_id[l:]))

        term_id = n[1]
        score = n[2]
        if post_id in term_dic:
            term_dic[post_id].append(terms[term_id])
            score_dic[post_id].append(score)
        else:
            term_dic[post_id] = [terms[term_id]]
            score_dic[post_id]=[score]
            
    def get_drug(x):
        try:
            return term_dic[x.name]
        except:
            return np.nan
    def get_score(x):
        try:
            return score_dic[x.name]
        except:
            return np.nan
    df['matched_drugs'] = df.apply(get_drug,axis=1)
    df['matching_score'] = df.apply(get_score,axis=1)
