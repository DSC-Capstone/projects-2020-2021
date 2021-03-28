import pandas as pd
import numpy as np
import pickle
import os
import sys
from scipy import sparse
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from lightfm.data import Dataset
from src.models import create_model


def get_model(filtered_a, filtered_q, interactions, weights, parameters, test_percent, split_random_state, epochs):
    """ Create LightFM collaborative filtering model using parameters """
    # Create the hybrid model
    model = LightFM(loss='warp',
                random_state=parameters[0],
                learning_rate=parameters[1],
                no_components=parameters[2],
                #user_alpha=0.000005,
                item_alpha=parameters[3]
               )
    
    # Split the train and test data from interactions and weights
    train, test = random_train_test_split(interactions, test_percentage=test_percent, random_state = split_random_state)
    train_weight, test_weight = random_train_test_split(weights, test_percentage=test_percent, random_state = split_random_state)
    
    # Fit the model
    print('Fitting Baseline Model:')
    model = model.fit(train, sample_weight = train_weight,
         epochs=epochs, verbose=True)
    
    return model



def main(configs):
    # Checks if inputs are already created
    if configs.get(configs["item_features"], False):
        
        # Need to build models in order to get the inputs for the baseline
        create_model.main(configs)
        
    # Load in inputs from their file locations
    filtered_a = pd.read_csv(configs["answers_file"])
    unfinished_q = pd.read_csv(configs["unfinished_questions_file"])
    interactions = sparse.load_npz(configs["interactions"])
    weights = sparse.load_npz(configs["weights"])
    parameters = [configs["model_random_state"], configs["learning_rate"], configs["no_components"], configs["item_alpha"]]
    test_percent = configs["test_percent"]
    split_random_state = configs["split_random_state"]
    epochs = configs["epochs"]
    
    # Replace NaN with an empty string
    unfinished_q['Tags'] = unfinished_q['Tags'].fillna('')
    df = unfinished_q.sort_values('post_indicies').reset_index()

    # Construct post indices by transforming the data
    item_dict = {}
    for i in range(df.shape[0]):
        item_dict[int(df.loc[i,'post_indicies'])] = int(df.loc[i,'Id'])

    # Save baseline questions and item dict
    unfinished_q.to_csv(configs["baseline_questions_file"]) 
    with open(configs["baseline_item_dict"], 'wb') as dict_fle:
        pickle.dump(item_dict, dict_fle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get the desired model
    model = get_model(filtered_a, unfinished_q, interactions, weights, parameters, test_percent, split_random_state, epochs)
    
    # Save the baseline model
    with open(configs["baseline_model"], 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main(sys.argv)