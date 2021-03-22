# Brian Cheng
# Eric Liu
# Brent Min

# cosine_rec.py contains all the logic needed to return the routes based on your past ratings

import pandas as pd
from pymongo import MongoClient

from src.functions import make_absolute

from math import sin, cos, sqrt, atan2, radians

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup, Tag
import json

from src.model.model_functions import get_mongo_data, format_df, generate_notes, get_mongo_user_data

def cosine_rec(args=None, data_params=None, web_params=None):
    """
    A recommender which recommends climbs based on the users previously liked climbs

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

    # get user's ticks and ratings
    ticks_ratings = get_user_ticks(web_params['user_url'])

    #get data of user's past climbs
    user_df = get_mongo_user_data(ticks_ratings['climb_id'].tolist())
    user_df = user_df.merge(ticks_ratings, on="climb_id").drop_duplicates(subset=["name"])

    # remove climbs the user has already done from the potential recs df
    df = df[~df["climb_id"].isin(user_df["climb_id"])]

    #defining favorite as highest rated
    fav_routes = user_df[user_df['user_rating'] == user_df['user_rating'].max()]
    
    # only look at the numerical attributes so far (will create more later)
    attributes = ['latitude', 'longitude', 'avg_rating', 'num_ratings', 'height_ft', 'height_m', 
        'pitches', 'grade', 'difficulty']

    # get the cosine sim matrix
    sim_matrix = cosine_similarity(fav_routes[attributes], df[attributes])

    # create a df from the cosine sim matrix
    sim_matrix = pd.DataFrame(data=sim_matrix, columns=df["climb_id"])

    # find the highest similarity score for each climb
    highest_sim = sim_matrix.max(axis=0).sort_values(ascending=False)

    # find the climb_ids of the top N climbs
    highest_sim = highest_sim.index[:web_params["num_recs"]]

    # get the top N climbs
    recommendations = df[df["climb_id"].isin(highest_sim)]

    # generate any generic notes
    notes = generate_notes(recommendations, web_params)
    
    # create the formatted recommendations dict based on the number of recommendations to output
    result = format_df(recommendations)

    # put results and notes together and return 
    return  {"recommendations": result, "notes": notes}

def get_user_ticks(user_url):
    """
    This function takes the input user MountainProject url and gets the climbs the user has done
    (aka ticks) and the rating the user gave the climb.

    :param:     user_url            The url where the users profile can be found
    :return:    [[int], [float]]    The first list contains the ticks of the user and the second 
                                    list contains the ratings for the ticks. The two lists should
                                    be the same length
    """
    # store ticks and ratings here
    ticks_ratings = pd.DataFrame(columns=["climb_id", "user_rating"])

    # get the users profile page
    user_page = requests.get(user_url + '/ticks').text
    soup = BeautifulSoup(user_page, 'html.parser')

    # if the user has multiple pages of ticks, iterate over them
    num_pages = int(soup.find_all('a', {"class":"no-click"})[-1].contents[0].strip()[-1])
    for i in range(num_pages):

        # get the page of ticks
        page = requests.get(user_url + '/ticks?page=' + str(i + 1)).text
        soup = BeautifulSoup(page, 'html.parser')

        # get all the links on the page and iterate over them
        all_links = soup.find_all('a')
        for link in all_links:

            # store the climb_id and rating here
            climb_id = None
            rating = 0            

            # get the id of the climb
            if(len(link) > 1 and len(link.find_all('strong')) > 0):
                climb_id = int(link.get("href").split("/")[4])

            # only get the rating if we got the climb_id
            if(climb_id is not None):
                # get the star rating for the climb
                ratings_list = link.find_all('span', {"class":"scoreStars"})
                if len(ratings_list) > 0:
                    for element in ratings_list[0].contents:
                        if isinstance(element, Tag):
                            image = element['src']
                            if image == '/img/stars/starBlue.svg':
                                rating += 1
                            if image == '/img/stars/starBlueHalf.svg':
                                rating += 0.5

                # only add the climb if we got the climb_id
                ticks_ratings = ticks_ratings.append({"climb_id": climb_id, 
                    "user_rating": rating}, ignore_index=True) 

    # return the ticks and ratings
    return ticks_ratings
