import pandas as pd
import numpy as np
import pickle
import os
import sys
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer



def transform_text(configs):
    """ Transform test inside questions using tfidf and save questions and item features """
    # Get unfinished questions
    questions = pd.read_csv(configs["unfinished_questions_file"])
    
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    questions['Body'] = questions['Body'].fillna('')
    
    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(questions['Body'])
    
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    #tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    questions['Tags'] = questions['Tags'].fillna('')
    #valid_filtered_q['Body'] = valid_filtered_q['Body'].fillna('')
    
    # Construct the required TF-IDF matrix by fitting and transforming the data
    tags_matrix = tfidf.fit_transform(questions['Tags'])
    item_dict ={}
    df = questions.sort_values('post_indicies').reset_index()
    for i in range(df.shape[0]):
        item_dict[int(df.loc[i,'post_indicies'])] = int(df.loc[i,'Id'])
        
    # Convert to csr matrix
    books_metadata_csr = csr_matrix(tfidf_matrix)
    tags_csr = csr_matrix(tags_matrix)
    books_metadata_csr = sparse.hstack((books_metadata_csr, tags_csr), format='csr')
    
    # Save to files
    questions.to_csv(configs["questions_file"])
    sparse.save_npz(configs["item_features"], books_metadata_csr)
    with open(configs["item_dict"], 'wb') as dict_fle:
        pickle.dump(item_dict, dict_fle, protocol=pickle.HIGHEST_PROTOCOL)

    return questions

def get_inputs(configs, filtered_a, filtered_q):
    """ Generate and save LightFM inputs using datasets """
    # Create LightFM Dataset and fit it with user and post values and dummy values
    dataset = Dataset()
    dataset.fit((x for x in filtered_a.user_indicies.values),
                (x for x in filtered_a.post_indicies.values))
    dummies = range(max(filtered_a.user_indicies.values)+1, max(filtered_a.user_indicies.values)+100)
    dataset.fit_partial((x for x in dummies))   
    
    # Save user and post indices
    np.save(configs["user_indicies"], filtered_a.user_indicies.values)
    np.save(configs["post_indicies"], filtered_a.post_indicies.values)

    # Build interactions and weights objects
    filtered_a.Score = filtered_a.Score.apply(int)
    (interactions, weights) = dataset.build_interactions(((x[0], x[1], x[2])
                                    for x in filtered_a[['user_indicies', 'post_indicies', 'Score']].values))
    sparse.save_npz(configs["interactions"], interactions)
    sparse.save_npz(configs["weights"], weights)
    print('interections matrix shape: ', dataset.interactions_shape())
    

def main(configs):
    
    # Get answers dataframe
    filtered_a = pd.read_csv(configs["answers_file"])
    
    # Transform text, save item features files, and get questions dataframe
    filtered_q = transform_text(configs)
    
    # Create all other inputs
    get_inputs(configs, filtered_a, filtered_q)
    

if __name__ == "__main__":
    main(sys.argv)