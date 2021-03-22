# Brian Cheng
# Eric Liu
# Brent Min

#uploads data onto mongodb

import json 
import csv
import pandas as pd

from tqdm import tqdm

from pymongo import MongoClient
from src.functions import *

def upload_data(data_params, my_client):
    # iterate over every state
    # it is assumed that data is saved and named accoring to get_clean_data.py
    for state in data_params["state"]:
        
        climbs = my_client.MountainProject.climbs
        climbs_data_path = make_absolute(data_params["clean_data_folder"] + state + '_climbs.csv')

        with open(climbs_data_path, encoding="utf-8") as f:
            climbs_df = pd.read_csv(f, encoding="utf-8")
            climbs_data = climbs_df.to_dict('records')

        # replaces the climb, but if the climb doesn't exist in Mongo, it will upload it
        for entry in tqdm(climbs_data):
            climbs.replace_one(entry, entry, upsert=True)
