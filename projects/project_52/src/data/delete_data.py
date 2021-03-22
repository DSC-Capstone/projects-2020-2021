# Brian Cheng
# Eric Liu
# Brent Min

#wipes out data from mongodb

import json 
import csv

from pymongo import MongoClient
from src.functions import *

def delete_data(data_params, my_client):
    #wipe out the climbs data from MongoDB
    climbs = my_client.MountainProject.climbs
    climbs.remove()
    