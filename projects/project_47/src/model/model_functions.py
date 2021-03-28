# Brian Cheng
# Eric Liu
# Brent Min

# model_functions.py contains various functions useful across all recommender models.

import pandas as pd

from pymongo import MongoClient

import math

import time

def get_mongo_data(web_params):
    """
    Get data from mongo based on the web params. This function is mostly here to limit the amount
    of data being transfered, and to collate logic across different models

    :param:     web_params      Paramters from the website. Only needed ones are location

    :return:    pd.df       A pandas DataFrame containing approximate data for the users requested
                            location.
    """
    # get a connection to mongo
    client = MongoClient('mongodb+srv://DSC102:Kw4ngOgiLl6mPi5r@cluster0.4gstr.mongodb.net/MountainProject?retryWrites=true&w=majority')

    # create an approximation lat/lng square from the web params
    
    # first convert the maximum distance to approximate lat/lng differences, with a 5% margin of
    # error
    # note 1 degree lat ~= 69 miles, and 1 degree lng ~= 54.9 miles
    lat_distance = (web_params["max_distance"] / 69) * 1.05
    lng_distance = (web_params["max_distance"] / 54.6) * 1.05

    # then get the min/max lat/lng values that form the square
    lat_min = float(web_params["location"][0]) - lat_distance
    lat_max = float(web_params["location"][0]) + lat_distance
    lng_min = float(web_params["location"][1]) - lng_distance
    lng_max = float(web_params["location"][1]) + lng_distance

    # then query mongo with these parameters
    climbs = client.MountainProject.climbs
    climbs = climbs.find({"latitude": {"$gte": lat_min, "$lte": lat_max}, 
        "longitude": {"$gte": lng_min, "$lte": lng_max}})
    df = pd.DataFrame.from_records(list(climbs))

    # do some basic cleaning 
    df = df.fillna(-1)
    df['climb_type'] = df['climb_type'].apply(lambda x: x.strip('][').split(', '))   

    # filter based on the rest of the web params
    df = filter_df(df, web_params["location"], web_params["max_distance"], 
        web_params["difficulty_range"])

    # return the df
    return df

def get_mongo_user_data(climb_ids):
    """
    Get data from mongo based on the climb_ids. This function is mostly here to limit the amount
    of data being transfered, and to collate logic across different models

    :param:     climb_ids      A list of climb IDs

    :return:    pd.df       A pandas DataFrame containing approximate data for the users requested
                            location.
    """
    # get a connection to mongo
    client = MongoClient('mongodb+srv://DSC102:Kw4ngOgiLl6mPi5r@cluster0.4gstr.mongodb.net/MountainProject?retryWrites=true&w=majority')

    # then query mongo with these parameters
    climbs = client.MountainProject.climbs
    climbs = climbs.find({"climb_id": {"$in": climb_ids}})
    df = pd.DataFrame.from_records(list(climbs))

    # do some basic cleaning 
    df = df.fillna(-1)
    df['climb_type'] = df['climb_type'].apply(lambda x: x.strip('][').split(', '))   

    # return the df
    return df

def generate_notes(rec_df, web_params):
    """
    This function generates a list of notes based on the recommendations and the web_params. Note 
    this function should be called on the same input to format_df(), not on the output of 
    format_df()

    :param:     rec_df      The df of recommendations. It is assumed that this df contains all 
                            columns from the original cleaned data
    :param:     web_params  Params from the website

    :return     [str]   A list of strings where each string is a note. Can be an empty list
    """
    # store notes here
    notes = []

    # make sure the correct number of recommendations were generated
    if(len(rec_df.index) < web_params["num_recs"]):
        note_str = f"Could not generate {web_params['num_recs']} recommendations based on the " \
            "selected options."
        notes.append(note_str)

    # make sure that at least some boulders were recommended if the user wanted boulders
    if((web_params["difficulty_range"]["boulder"][0] != -1) and (len(rec_df.index) > 0)):
        # sum up the boulder_climb column
        # if this column has at least on non zero value, then at least one boulder was recommended
        # note that here we use len(rec_df.index) instead of web_params["num_recs"], since it is
        # possible to have less than web_params["num_recs"] recommendations at this point
        num_boulders = rec_df["boulder_climb"].sum()
        if(num_boulders == 0):
            notes.append(f"The top {len(rec_df.index)} recommendations are all routes. " \
                "To generate boulders, turn off routes or increase the number of recommendations.")

    # make sure that at least some routes were recommended if the user wanted routes
    if((web_params["difficulty_range"]["route"][0] != -1) and (len(rec_df.index) > 0)):
        # use the same logic as above
        num_routes = rec_df["rock_climb"].sum()
        if(num_routes == 0):
            notes.append(f"The top {len(rec_df.index)} recommendations are all boulders. " \
                "To generate routes, turn off boulders or increase the number of recommendations.")

    # return the notes
    return notes

def format_df(rec_df):
    """
    This function takes the input df and formats it so that django can easily display it.

    :param:     rec_df      The df of recommendations. It is assumed that this df contains all 
                            columns from the original cleaned data

    :return:    [{}]        A list of dictionaries in the following format: 
                            [{"name": str, "url": int, "reason": str, "difficulty": str, 
                            "description": str}, {}, ...]
                            Each dictionary is a singular recommendation
    """
    # make sure there are recommendations
    if(len(rec_df.index) == 0):
        return []

    # create and return a list of the correct format
    formatted = rec_df.apply(lambda row: {"name": row["name"], "url": row["climb_id"],
        "difficulty": row_to_difficulty(row), "reason": "", "description": row["description"]}, 
        axis=1)
    return list(formatted)

def row_to_difficulty(row):
    """
    This function takes a single row of a df and converts the difficulty integer to the correct
    string based on whether the climb is a boulder or route

    :param:     row     A row of the df. This should have at minimum columns "difficulty" and
                        "boulder_climb"

    :return:    str     A string representation of the climb difficulty (V_ or 5._)
    """
    # check if the climb is a boulder
    if(row["boulder_climb"] == 1):
        # return the proper string
        mapping = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", 
            "V13", "V14", "V15", "V16"]
        return mapping[row["difficulty"]]
    # if the climb is not a boulder, it is assumed to be a route
    else:
        mapping = ['3rd', '4th', 'Easy 5th', '5.0', "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", 
            "5.7", "5.8", "5.9", "5.10a", "5.10b", "5.10c", "5.10d", "5.11a", "5.11b", "5.11c", 
            "5.11d", "5.12a", "5.12b", "5.12c", "5.12d", "5.13a", "5.13b", "5.13c", "5.13d", 
            "5.14a", "5.14b", "5.14c", "5.14d", "5.15a", "5.15b", "5.15c", "5.15d"]
        return mapping[row["difficulty"]]

def filter_df(df, location, distance, diff_ranges):
    """
    This function filters the input df based on the other three paramters, using the functions 
    below. This function additionally serves as a check to make sure duplicate entires do not exist

    :param:     df              The input df to filter. It is expected at minimum to have the 
                                columns "latitude", "longitude", "boulder_climb", "rock_climb", 
                                and "difficulty"
    :param:     location        The center location. A lat/lng pair
    :param:     distance        The distance from the center location in miles
    :param:     diff_ranges     A dictionary of the following format: {"boulder": [int, int],
                                "route": [int, int]}. Typically, these values are generated by the
                                website. If the user does not want a type of climb, the list will
                                be [-1, -1]

    :return:    pd.DataFrame    The input df filtered based on the parameters
    """
    # in order to reduce compute time, first filter by difficulty
    df = filter_type_difficulty(df, diff_ranges)

    # make sure that the df is not empty before filtering by location
    # an exception is raised when you try to apply from an empty df
    if(len(df.index) == 0):
        return df

    # disable a warning that does not apply here
    pd.options.mode.chained_assignment = None

    # now filter by location
    df["dis_mi"] = df.apply(lambda x: distance_lat_lng(location, (x["latitude"], x["longitude"])), 
        axis=1)
    df = df[df["dis_mi"] <= distance]

    # ensure that there are no duplicate entries
    df = df.drop_duplicates(subset=["name"])

    # return the filtered df
    return df

def distance_lat_lng(start_lat_lng, end_lat_lng):
    """
    This function returns the distance between two lat/lng pairs in miles
    TODO: use a approximation function to cut down on compute time?

    :param:     start_lat_lng   The first lat/lng pair. Any of list, tuple, iterable, etc.
    :param:     end_lat_lng     The second lat/lng pair. Any of list, tuple, iterable, etc.
        
    :return:    float           The distance between the two lat/lng pairs in miles
    """
    # earth radius in miles
    _radius = 3958.8

    # convert all lat/lng to radians
    start_lat_lng = (math.radians(start_lat_lng[0]), math.radians(start_lat_lng[1]))
    end_lat_lng = (math.radians(end_lat_lng[0]), math.radians(end_lat_lng[1]))

    # get the differences between the lats/lngs
    lat_distance = end_lat_lng[0] - start_lat_lng[0]
    lng_distance = end_lat_lng[1] - start_lat_lng[1]

    # apply the haversine formula: https://www.movable-type.co.uk/scripts/latlong.html
    a = (math.sin(lat_distance / 2) ** 2) + \
            (math.cos(start_lat_lng[0]) * math.cos(end_lat_lng[0]) * \
            (math.sin(lng_distance / 2) ** 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = _radius * c

    # since the initial radius was in miles, the final distance should also be in miles
    return d

def filter_type_difficulty(df, diff_ranges):
    """
    This function takes the input df and filters it by climb type and by difficulty

    :param:     df              A pandas dataframe. It is expected at minimum to have the columns 
                                "boulder_climb", "rock_climb", and "difficulty"
    :param:     diff_ranges     A dictionary of the following format: {"boulder": [int, int],
                                "route": [int, int]}. Typically, these values are generated by the
                                website. If the user does not want a type of climb, the list will
                                be [-1, -1]

    :return:    pd.DataFrame    The input df filtered based on the parameters
    """
    # store booleans to keep track if the user wants routes/boulders
    boulders = diff_ranges["boulder"][0] != -1
    routes = diff_ranges["route"][0] != -1

    # if the user wants boulders
    if(boulders):
        boulder_mask = (df["boulder_climb"] == 1) & \
            (df["difficulty"] >= diff_ranges["boulder"][0]) & \
            (df["difficulty"] <= diff_ranges["boulder"][1])

    # if the user wants routes
    if(routes):
        route_mask = (df["rock_climb"] == 1) & \
            (df["difficulty"] >= diff_ranges["route"][0]) & \
            (df["difficulty"] <= diff_ranges["route"][1])

    # combine the masks are necessary 
    if(boulders and routes):
        return df[boulder_mask | route_mask]
    elif(boulders):
        return df[boulder_mask]
    elif(routes):
        return df[route_mask]
