import os
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
def build_features(inputdir):
    reviews = pd.concat([pd.read_csv(inputdir + '/' + str(f)) for f in os.listdir(inputdir)],
        ignore_index = True)
    dataset = Dataset()
    dataset.fit(users = (x for x in reviews.userID),
                items = (x for x in reviews.productID))
    num_users, num_items = dataset.interactions_shape()
    (interactions, weights) = dataset.build_interactions(list(zip(reviews.userID, reviews.productID)))
    return interactions, weights