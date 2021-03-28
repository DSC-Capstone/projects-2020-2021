# Brian Cheng
# Eric Liu
# Brent Min

# top_pop.py contains a top popular recommender, where climbs are first filtered to those over
# 3.5/4 stars, then sorted by number of reviews.

import pandas as pd
from pymongo import MongoClient

from src.functions import make_absolute
from src.model.model_functions import get_mongo_data, format_df, generate_notes

from math import sin, cos, sqrt, atan2, radians

def top_pop(args=None, data_params=None, web_params=None):
    """
    A simple top popular which takes climbs over 3.5/4 stars and returns those climbs with the
    most number of reviews.

    :param:     args            Command line arguments
    :param:     data_params     Data params for running the project from the command line
    :param:     web_params      Params from the website

    :return:    dict            A dictionary in the following format:   
                                {
                                    "recommendations": [{"name": str, "url": int, "reason": str,
                                        "difficulty": str, "description": str}, {}, ...],
                                    "notes": str
                                }
                                Where each item in the "recommendations" list is a singular 
                                recommendation. All recommenders should return in this format
    """
    # get filtered data from mongo based on the input web params
    df = get_mongo_data(web_params)

    # do a simple top popular
    toppop = df[df['avg_rating'] >= 3.5].sort_values('num_ratings', ascending=False)

    # get however many recommendations are requested
    toppop = toppop[:web_params["num_recs"]]

    # generate any generic notes
    notes = generate_notes(toppop, web_params)
    
    # create the formatted recommendations dict based on the number of recommendations to output
    result = format_df(toppop)

    # put results and notes together and return 
    return  {"recommendations": result, "notes": notes}
