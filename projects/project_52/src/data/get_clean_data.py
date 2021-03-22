# Brian Cheng
# Eric Liu
# Brent Min

# get_clean_data.py collates all the logic needed to clean and save data

import json
import csv

from tqdm import tqdm

from src.functions import make_absolute

def get_clean_data(data_params):
    """
    This function collates all cleaning logic

    :param:     data_params     A dictionary containing all data parameters. The only ones used are
                                the location at which to download raw data and the location at which
                                to save clean data
    """
    # iterate over every state
    # it is assumed that data is saved and named accoring to get_raw_data.py
    for state in data_params["state"]:
        # get the url at which raw data will be found
        raw_data_path = make_absolute(data_params["raw_data_folder"] + state + ".json")
        print(raw_data_path)
        
        # get the data
        with open(raw_data_path, "r") as f:
            raw_data = json.load(f)

        # store all clean data as a list of lists
        # note that the first input row is the column names
        climb_data = [["climb_id", "name", "description", "image_url", "latitude", "longitude", 
            "avg_rating", "num_ratings", "url", "climb_type", "height_ft", "height_m", "pitches",
            "grade", "protection", "difficulty", 'rock_climb', 'boulder_climb']]
        #user_data = [["user_id", "climb_id", "rating"]]

        # process the data
        for climb in raw_data:
            # get the climb/user data and add it to the list of lists
            climb_row = split_into_user_climb(climb)
            climb_data.append(climb_row)
            #for user_row in user_rows:
            #    user_data.append(user_row)

        # save the lists of lists as csv data in the proper location
        clean_data_path = str(make_absolute(data_params["clean_data_folder"])) + "/"
        with open(clean_data_path + state + "_climbs.csv", "w", encoding="utf-8", 
            newline="") as f:
            writer = csv.writer(f)
            writer.writerows(climb_data)

def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val

def split_into_user_climb(climb_dict):
    """
    This function takes the json for a single climb and splits the data into a primary key climb_id
    dataset and a primary key user_id dataset

    :param:     climb_dict      The full scraped contents of one climb in dict form

    :return:    ([], [[]])      A tuple containing a list, and a list of lists. The first list   
                                contains a row of the climb.csv file, and the second list of lists
                                contains user_ids, climb_ids, and user_ratings
    """
    # all the info for the climb row
    climb_id = climb_dict["route_url"].split("/")[-2]
    try:
        image_url = climb_dict["image"]
    except KeyError:
        image_url = "N/A"

    #extract climb types, height (ft and m), pitches, and grade
    type_data = climb_dict['climb_type'].split(',')
    height_ft = None
    height_m = None
    pitches = None
    grade = None
    climb_types = []
    for data in type_data:
        if 'ft' in data:
            height_ft = int(data.split()[0])
            height_m = int(data.split()[2][1:])
        elif 'pitches' in data:
            pitches = int(data.split()[0])
        elif 'Grade' in data:
            grade = roman_to_int(data.split()[1])
        else:
            climb_types.append(data)
    
    #ordering difficulty
    difficulty_mapping = ['3rd', '4th', 'Easy 5th', '0', "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10a", "10b", "10c", "10d", "11a", "11b", "11c", "11d", "12a", "12b", "12c", "12d", "13a", 
        "13b", "13c", "13d", "14a", "14b", "14c", "14d", "15a", "15b", "15c", "15d"]
    route_str = climb_dict['difficulty_rating'].strip()
    
    #if it's a normal rock climbing route
    if route_str[0] != 'V':
        if '.' in route_str:
            route_str = route_str.split('.')[1].split('/')[0]
        rock = 1
        boulder = 0
        if route_str[-1] == '+' or route_str[-1] == '-':
            route_str = route_str[:-1]
        if route_str == '10':
            route_str = '10a'
        if route_str == '11':
            route_str = '11a'
        if route_str == '12':
            route_str = '12a'
        if route_str == '13':
            route_str = '13a'
        if route_str == '14':
            route_str = '14a'
        if route_str == '15':
            route_str = '15a'
        difficulty = difficulty_mapping.index(route_str)
    #if it's a bouldering climb
    else:
        for i in range(1, len(route_str)):
            if route_str[i].isnumeric() == False:
                route_str = route_str[:i]
                break
        rock = 0
        boulder = 1
        if len(route_str) < 2:
            difficulty = 0
        else:
            difficulty = int(route_str[1:])

    #all the info for the climb row
    climb_row = [climb_id, climb_dict["name"], climb_dict["description"], image_url,
        climb_dict["geo"]["latitude"], climb_dict["geo"]["longitude"], 
        climb_dict["aggregateRating"]["ratingValue"], climb_dict["aggregateRating"]["reviewCount"],
        climb_dict["route_url"], climb_types, height_ft, height_m, pitches, grade, climb_dict['protection'],
        difficulty, rock, boulder]

    # all the info for the user row
    # user_rows = list(map(list, climb_dict["user_ratings"].items()))
    #for user_row in user_rows:
    #    user_row.insert(1, climb_id)

    # return the info as a tuple
    return climb_row
