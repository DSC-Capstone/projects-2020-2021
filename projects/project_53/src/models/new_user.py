import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from scipy import sparse
from lightfm.data import Dataset

def main(configs):
    # Get new samples of user chosen questions from the website and epochs
    f = configs["new_sample"]   
    new_epochs = configs["new_epochs"]
    
    # Load user/post indices and post mappings
    user_indicies = np.load(configs["user_indicies"])
    post_indicies = np.load(configs["post_indicies"])
    post_mappings = pd.read_csv(configs["post_mappings"])
    
    # Restructure Post Mappings
    post_mappings.columns = [ 'ParentId', 'post_indicies']
    post_mappings.index = post_mappings['ParentId']
    post_mappings = post_mappings['post_indicies']
    post_ind = lambda x: post_mappings.loc[x]
  
    # Get model
    model = pickle.load(open(configs["model"], "rb"))
    
    # Get dataset and create dummy place
    dataset = Dataset()
    dataset.fit((x for x in user_indicies),
            (x for x in post_indicies))
    dummies = range(max(user_indicies) + 1, max(user_indicies)+100)
    dataset.fit_partial((x for x in dummies))  
    print(dataset.interactions_shape())
    
    # Read in new sample
    new = pd.read_csv(f)
    
    # Apply post mapping
    new['post_indicies'] = new['ParentId'].apply(post_ind)
    
    # Put users in dummies
    new_user_indicies = dict()
    for i in range(len(new.OwnerUserId.unique())):
        new_user_indicies[new.OwnerUserId.unique()[i]] = dummies[i]
    new['user_indicies'] = new.OwnerUserId.apply(lambda x: new_user_indicies[x])
    print(new['user_indicies'].values)
    
    # Fit partial with new dummies
    dataset.fit_partial((x for x in new.user_indicies.values),
             (x for x in new.post_indicies.values))

    # New interactions and weight
    (new_interactions, new_weights) = dataset.build_interactions(((x[6], x[7], x[0]) for x in new.values))

    # Load item features
    item_features = sparse.load_npz(configs["item_features"])
    
    # Get mean embedding score
    for i in new.user_indicies.unique():
          print(i, 'mean user embedding before refitting :', np.mean(model.user_embeddings[i]))
    print(new_interactions.shape)
    
    # Generate new model
    model = model.fit_partial(new_interactions, item_features = item_features, sample_weight = new_weights,
         epochs=configs["new_epochs"], verbose=True)   
    
    # Get new mean embedding score
    for i in new.user_indicies.unique():
          print(i, 'mean user embedding after refitting:', np.mean(model.user_embeddings[i]))      
    
    # Overwrite model
    with open(configs["model"], 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    main(sys.argv)
    
