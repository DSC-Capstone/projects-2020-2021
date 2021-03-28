import pandas as pd
import numpy as np
import pickle
import os
import sys
from scipy import sparse
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from . import create_inputs

def get_model(filtered_a, filtered_q, interactions, weights, books_metadata_csr, parameters, test_percent, split_random_state, epochs):
    """ Create LightFM Hybrid model using parameters """
    
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
    print('Fitting Model:')
    model = model.fit(train, item_features = books_metadata_csr, sample_weight = train_weight,
         epochs=epochs, verbose=True)
    
    return model
    

def main(configs):
    # Create inputs and set them to files
    create_inputs.main(configs)
    
    # Load in inputs from their file locations
    filtered_a = pd.read_csv(configs["answers_file"])
    filtered_q = pd.read_csv(configs["questions_file"])
    interactions = sparse.load_npz(configs["interactions"])
    weights = sparse.load_npz(configs["weights"])
    books_metadata_csr = sparse.load_npz(configs["item_features"])
    parameters = [configs["model_random_state"], configs["learning_rate"], configs["no_components"], configs["item_alpha"]]
    test_percent = configs["test_percent"]
    split_random_state = configs["split_random_state"]
    epochs = configs["epochs"]
    
    # Get the desired model
    model = get_model(filtered_a, filtered_q, interactions, weights, books_metadata_csr, parameters, test_percent, split_random_state, epochs)
    
    # Save model
    with open(configs["model"], 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main(sys.argv)