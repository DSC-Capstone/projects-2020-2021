# Brian Cheng
# Eric Liu
# Brent Min

# get_raw_data.py contains all the logic needed to scrape data from the mountainproject website

import requests
import json
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.functions import make_absolute

def get_raw_data(data_params):
    """
    This function collates all scraping logic

    :param:     data_params     A dictionary containing all data parameters. The only one used is
                                the location at which to save raw data, and the states to scrape
    """
    # store raw data here
    raw_data = []

    # iterate over every state
    for state, url in data_params["states"]:
        all_routes = find_all_routes_in_area(url)

        # for every route in the state, get the route data
        for route_url in tqdm(all_routes):
            route_data = get_route_data(route_url)
            if route_data:
                raw_data.append(route_data)

        # save the raw data
        with open(make_absolute(data_params["raw_data_folder"] + state + ".json"), "w") as f:
            json.dump(raw_data, f)

        # after saving the raw data, clear the raw data list
        raw_data = []

def get_route_data(route_url):
    """
    Get all route data for a single route

    :param:     route_url   The URL at which the route lives

    :return:    dict        A dictionary containing the following information:
                            "@context": N/A
                            "@type": N/A
                            "name": Name of the climb
                            "description": Description of the climb
                            "image": link to the climb image
                            "geo": dict with lat/long
                            "aggregateRating": dict with average rating/number of ratings
                            "route_url": url at which the route can be found
                            "user_ratings": dict with key user_id and value user_rating (0-4)
    """
    # get the climb description
    text = ''
    while text == '':
        try:
            text = requests.get(route_url).text
            break
        except:
            print('too fast, sleeping...')
            time.sleep(5)
            print('finished sleeping')
            continue
    soup = BeautifulSoup(text, 'html.parser')
    
    #climb type
    climb_type = soup.find('td', string='Type:').next_sibling.next_sibling.contents[0].strip()
    if 'Aid' in climb_type or 'Ice' in climb_type or 'Mixed' in climb_type:
        return None

    #difficulty rating and difficulty rating system sections
    difficulty_section = soup.find('h2', {'class': 'inline-block mr-2'})
    if difficulty_section is None or len(difficulty_section) == 0:
        difficulty_rating = 'NA'
        difficulty_rating_system = 'NA'
    else:
        if isinstance(difficulty_section.contents[0], str):
            difficulty_rating = difficulty_section.contents[0]
            difficulty_rating_system = 'NA'
        else:
            difficulty_rating = difficulty_section.contents[0].contents[0]
            difficulty_rating_system = difficulty_section.contents[0].contents[1].contents[0].contents[0]
    if difficulty_rating_system != 'YDS':
        return None
            
    # split up the stuff we want
    data = json.loads("".join(soup.find("script", {"type":"application/ld+json"}).contents))
    
    #description + protection sections
    description_section = soup.find_all('div', {'class': 'fr-view'})
    description = description_section[0].contents
    if len(description_section) == 3:
        protection = description_section[2].contents
    elif len(description_section) == 2:
        protection = description_section[1].contents
    else:
        protection = 'No protection data'
        
    data['climb_type'] = climb_type
    data['description'] = ''.join([x for x in description if isinstance(x, str)])
    data['protection'] = ''.join([x for x in protection if isinstance(x, str)])
    data['difficulty_rating'] = difficulty_rating
    data['difficulty_rating_system'] = difficulty_rating_system
    data['route_url'] = route_url

    # if there exists more than zero ratings for this climb, then get those user(s)
    if(int(data["aggregateRating"]["reviewCount"]) > 0):
        data["user_ratings"] = get_route_rating_data(route_url)
    # otherwise just add an empty dict
    else:
        data["user_ratings"] = {}

    return data

def get_route_rating_data(route_url):
    """
    Get all user_ids and the ratings for the input route

    :param:     route_url   The URL at which the route lives. NOTE (!) this is not the URL at which
                            rating data can be found, it is modified in this function

    :return:    dict        A dictionary containing the key of user_id and value of user rating
                            for the input climb url
    """
    # first modify the route_url to access the stats
    route_stats_url = route_url.split("/")
    route_stats_url.insert(4, "stats")
    route_stats_url = "/".join(route_stats_url)

    # get the html of the stats page
    text = requests.get(route_stats_url).text
    soup = BeautifulSoup(text, 'html.parser')

    # get the first table which should contain ratings
    ratings_table = soup.find("table", attrs={"class": "table"})

    # make sure that a table exists 
    if(ratings_table == None):
        print(route_stats_url)
        return {}

    # get the rows of the rating table
    rows = ratings_table.find_all("tr")
    
    # store ratings here
    user_ratings = {}

    # every single row is a users rating
    for row in rows:
        # first col contains the user_id
        cols = row.find_all("td")
        user_url = cols[0].find("a").get("href")
        user_id = user_url.split("/")[4]

        # second col contains the user rating (0-4 stars)
        stars = len(cols[1].find_all("img"))

        # the above interprets avoid (single image of a bomb) the same as one star (single image
        # of a star)
        # adjust for that
        if(stars == 1 and ("bomb" in cols[1].find("img")["src"])):
            stars = 0
        
        # store the rating
        user_ratings[user_id] = stars
            
    # return the ratings
    return user_ratings

def find_all_routes_in_area(area_url):
    """
    Get all the URLS of all the routes that exist in the input area

    :param:     area_url    The URL of the area to look in. It does not matter how high level of 
                            an area this URL points to, as long as it is an area of some sort
    """
    def add_links(soup):
        """
        Recursive function that adds newly found links under an area into the global list 'links'.

        :param:     soup    the parsed document of the given area URL
        """
        # locates the box containing links and add the links to the global list 'links'
        div = soup.find('div', {'class': 'max-height max-height-md-0 max-height-xs-400'})
        if div:
            a_hrefs = div.find_all('a')
            for link in a_hrefs:
                links.append(link.get('href'))
    # goes through the links until we went through all of them
    # if a link is still an area, we go through its sublinks again recursively
    # if a link is a route, we store it in our route_links variable
    # we return route_links variable in the end as a list of route links
    text = requests.get(area_url).text
    soup = BeautifulSoup(text, 'html.parser')
    links = []
    route_links = []
    add_links(soup)
    while len(links) > 0:
        link = links[0]
        if '/area/' in link:
            text = requests.get(link).text
            new_soup = BeautifulSoup(text, 'html.parser')
            add_links(new_soup)
        else:
            if link != '#':
                route_links.append(link)
        links.remove(link)
    return route_links
