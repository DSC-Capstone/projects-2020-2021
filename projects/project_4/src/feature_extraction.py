import sys

sys.path.insert(1, '../src/')

from partyset import PartySet
from utils import data_utils
import numpy as np
import os
import pandas as pd
########################################
"""
run this file by doing python feature_extraction.py <n> <path_to_democrat_data> <path_to_republican_data>
for example: python feature_extraction.py 25 ../data/democrats ../data/republicans
"""


def bag_of_words_features(vocab, document):
    """
    vocab is a list of word stems
    document is a list of word stems
    """
    features = np.zeros(len(vocab))
    for i in range(len(vocab)):
        features[i] = document.count(vocab[i])
    return features

def get_vocab(tfidf_df, n):
    # tfidf_df is the output of stemmed_text in data_utils
    return tfidf_df.head(n)['word'].tolist()

def main(n, dem_folder, rep_folder):
    demset = PartySet(dem_folder)
    repset = PartySet(rep_folder)
    
    dem_docs = demset.get_all_text()
    rep_docs = repset.get_all_text()
    
    # Cleaning Dems & Reps
    print('cleaning data')
    cln_dem = [data_utils.clean_text(doc) for doc in dem_docs]
    cln_rep = [data_utils.clean_text(doc) for doc in rep_docs]
    
    # getting stemmed text for Dems & Reps
    print('stemming data')
    _, dem_stemmed = data_utils.stemmed_text(cln_dem)
    __, rep_stemmed = data_utils.stemmed_text(cln_rep)
    
    # Combining Dems & Reps to get Vocab
    print('stemming combined data')
    all_docs = cln_dem + cln_rep
    count_df, stemmed_docs = data_utils.stemmed_text(all_docs)
    
    print('getting vocab')
    vocab = get_vocab(count_df, n)
    
    print('getting dem features')
    dem_data = []
    for document in dem_stemmed:
        # dem_stemmed is a list of lists where each element is a document
        # document is a list of word stems for a given tweet
        doc_feats = bag_of_words_features(vocab, document)
        dem_data.append((doc_feats, 'D'))
        
    print('getting rep features')
    rep_data = []
    for document in rep_stemmed:
        # rep_stemmed is a list of lists where each element is a document
        # document is a list of word stems for a given tweet
        rep_feats = bag_of_words_features(vocab, document)
        rep_data.append((rep_feats, 'R'))
    
    print('creating dataframes')
    dem_df = pd.DataFrame(data = dem_data, columns = ['BoW_Vector', 'Party_Label'])
    rep_df = pd.DataFrame(data = rep_data, columns = ['BoW_Vector', 'Party_Label'])
    
    print('outputting dataframes to csv')
    dem_out = "../data/dem_bow_{n_words}.csv".format(n_words = n)
    rep_out = "../data/rep_bow_{n_words}.csv".format(n_words = n)
    dem_df.to_csv(dem_out, index=False)
    rep_df.to_csv(rep_out, index=False)
    
    print('done!')
if __name__ == "__main__":
    n_words = sys.argv[1]
    n = int(n_words)
    dem_folder = sys.argv[2]
    rep_folder = sys.argv[3]
    
    main(n, dem_folder, rep_folder)
