import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import requests

def read_github_json_file(url):
    resp = requests.get(url)
    nearby_data = json.loads(resp.text)
    return nearby_data
def find_FIPs(county_name,us_confirmed_df,state_name = "California"):
    fips = us_confirmed_df[(us_confirmed_df.Admin2 == county_name) & (us_confirmed_df.Province_State == state_name)]["FIPS"].values[0]
    return str(fips)
def generate_sc_neighbor_dictionary(us_confirmed_df):
    '''
    return a dictionary of southern california counties,county FIP code being the key
    each value will be another dictionary, in the order of west,east,north, and south
    '''
    # main key = center county
    dic = {}
    # inner keys: west,east,north,south
    dic["San Bernardino"] = {"west":"Kern","east":"border","north":"border","south":"Riverside"}

    dic["Kern"] = {"west":"San Luis Obispo","east":"San Bernardino","north":"border","south":"Los Angeles"}

    dic["San Luis Obispo"] = {"west":"border","east":"Kern","north":"border","south":"Santa Barbara"}

    dic["Santa Barbara"] = {"west":"border","east":"Ventura","north":"San Luis Obispo","south":"border"}


    dic["Ventura"] = {"west":"Santa Barbara","east":"Los Angeles","north":"Kern","south":"border"}

    #Los Angeles County
    dic["Los Angeles"] = {"west":"Ventura","east":"San Bernardino","north":"Kern","south":"Orange"}
    #Orange County
    dic['Orange'] = {"west": "border", "east": "Riverside","north":"Los Angeles", "south": "San Diego"}

    # Riverside County
    dic['Riverside'] = {'west':"Orange",'east':"border",'north':"San Bernardino", 'south':"San Diego"}
    # San Diego County
    dic['San Diego']={'west':"border",'east':"Imperial","north":'Riverside',"south":"border"}
    # Imperial County
    dic['Imperial']={'west':"San Diego",'east':"border","north":'Riverside',"south":"border"}
    
    fips_neighbor_dict = {}
    for i in dic.keys():

        new_key = find_FIPs(i,us_confirmed_df)
        fips_neighbor_dict[new_key] = dic[i]
        for k in dic[i].keys():
            if dic[i][k] == "border":
                continue
            fips_neighbor_dict[new_key][k] = find_FIPs(dic[i][k],us_confirmed_df)
    return fips_neighbor_dict


def find_population_for_county(county_FIPS,us_death_df):
    population_df = us_death_df[us_death_df.columns[:12]]

    return population_df.loc[population_df.FIPS == county_FIPS]["Population"].values[0]

def calculate_lat_long_difference(county_FIP1,county_FIP2,us_confirmed_df):
    '''
    returns a tuple, where the first entry is the difference in lat
    the second entry is the difference in long
    '''
    lat_1 = us_confirmed_df.loc[us_confirmed_df.FIPS == county_FIP1]["Lat"].values[0]
    long_1 = us_confirmed_df.loc[us_confirmed_df.FIPS == county_FIP1]["Long_"].values[0]
    lat_2 = us_confirmed_df.loc[us_confirmed_df.FIPS == county_FIP2]["Lat"].values[0]
    long_2 = us_confirmed_df.loc[us_confirmed_df.FIPS == county_FIP2]["Long_"].values[0]
    return lat_1-lat_2,long_1-long_2   

    
def FIP_to_county_name(FIP,us_confirmed_df):
    df = us_confirmed_df.loc[us_confirmed_df.FIPS == FIP]
    county = df.Admin2.values[0]
    state = df.Province_State.values[0]
    return county+", "+state

def interpret_dictionary_FIP(FIP_dict,us_confirmed_df):
    li = list(FIP_dict.keys())
    for i in range(len(li)):

        key = li[i]
        new_key = FIP_to_county_name(key,us_confirmed_df)
        FIP_dict[new_key] = FIP_dict[key]
        del FIP_dict[key]
    return FIP_dict

def collect_data(path):
    '''
    a composite method that does everything! Collects the data and saves susceptible, infected, removed, and population
    given the params in the config/data_params.json file
    takes in the path where the person wants s,i,r,p to be stored
    returns the name of the path (folder)
    '''
    us_confirmed_df,us_death_df,global_recover_df,mobility = retrieve_data()
    with open('config/data_params.json') as fh:
        data_cfg =json.load(fh)
    start = data_cfg["start"]
    days = data_cfg["days"]
    admin_level = data_cfg["admin_level"]
    admin_name = data_cfg["admin_name"]
    df_list = [us_confirmed_df,us_death_df,global_recover_df]
    if admin_level == "state":
        s,i,r,p = get_state(start,days,df_list,admin_name)
    elif admin_level == "county":
        s,i,r,p = get_county(start,days,df_list,admin_name)
    elif admin_level == "country":
        s,i,r,p = get_country(start,days,df_list,admin_name)
    s_path = path+"s.txt"
    i_path = path+"i.txt"
    r_path = path+"r.txt"
    p_path = path+"p.txt"
    write_list_to_txt(s_path,s)
    write_list_to_txt(i_path,i)
    write_list_to_txt(r_path,r)
    write_list_to_txt(p_path,[p])
    
    return path
def standardize_FIPS(df):
    '''
    return: dataframe after changing the FIP code from float to string (zfilled) to standardize
    '''
    df["FIPS"] = df.loc[~df.FIPS.isna()].FIPS.apply(lambda x:str(int(x)).zfill(5))
    return df
def retrieve_data():
    '''
    retrieves information from the JHU and mobility data repository
    return a list of dataframes, including confirmed, death, recover (only global) and mobility data
    these dataframe will be used for generating sequential data limited to a geographical location, with a time frame.
    '''
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    us_confirmed_df = pd.read_csv(url, error_bad_lines=False)

    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    us_death_df = pd.read_csv(url, error_bad_lines=False)



    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    global_recover_df = pd.read_csv(url, error_bad_lines=False)



    url = "https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-m50.csv"
    mobility = pd.read_csv(url, error_bad_lines=False)
    mobility_standardized = standardize_mobility_df(mobility)
    mobility_standardized = standardize_FIPS(mobility_standardized)
    
    us_confirmed_df = us_confirmed_df.loc[us_confirmed_df.Admin2 != "Unassigned"]
    us_confirmed_df = standardize_FIPS(us_confirmed_df)
    
    us_death_df = us_death_df.loc[us_death_df.Admin2 != "Unassigned"]
    us_death_df = standardize_FIPS(us_death_df)

    return us_confirmed_df,us_death_df,global_recover_df,mobility_standardized

def standardize_mobility_df(mobility_df):
    mobility_standardized = change_col_name(mobility_df) 
    return mobility_standardized
def change_col_name(df):
    columns_dict = {}
    df.columns = [i.capitalize() for i in df.columns]
    
    for i in range(len(df.columns)):
        col = df.columns[i]
        if col == "Fips":
            columns_dict[col] = "FIPS"
        if (col.split("-")[0] == "2020") | (col.split("-")[0] == "2021"):
            date_time_obj = datetime.strptime(col, '%Y-%m-%d')
            date_time_str = date_time_obj.strftime("%-m/%-d/%y")
            columns_dict[col] = date_time_str
    df2 = df.rename(columns=columns_dict)
    df_dates_only = df2[df2.columns[5:]]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df2_standardized = scaler.fit_transform(df_dates_only)
    df2[df2.columns[5:]] = df2_standardized
    return df2



def get_county(start,days,df_list,county_code = 1001):
    '''
    get county level data
    start: starting day since patient 0 in Wuhan
    days: duraton
    df_list: the list of dataframes (confirmed, death, recovered, mobility)
    county_code: FIP of the county, defatul to 1001
    '''
    us_confirmed_df = df_list[0]
    us_death_df = df_list[1]
    global_recover_df = df_list[2]
    county_confirmed = us_confirmed_df.loc[us_confirmed_df.FIPS == county_code].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1).sum(axis=0).values
    county_death = us_death_df.loc[us_death_df.FIPS == county_code].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key',"Population"],axis=1).sum(axis=0).values
    county_population = us_death_df.loc[us_death_df.FIPS ==county_code].Population.sum()
    county_infected = county_confirmed
    county_removed = county_death
    county_susceptible = county_population - county_infected
    return county_susceptible[start:days+start],county_infected[start:start+days],county_removed[start:start+days],county_population

def get_state(start,days,df_list,state_name = "Washington"):
    '''
    get state level data
    start: starting day since patient 0 in Wuhan
    days: duraton of sequential data
    df_list: the list of dataframes (confirmed, death, recovered, mobility)
    county_code: FIP of the county, defatul to Washington since it is the first patient
    '''
    us_confirmed_df = df_list[0]
    us_death_df = df_list[1]
    global_recover_df = df_list[2]    
    county_confirmed = us_confirmed_df.loc[us_confirmed_df.Province_State == state_name].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1).sum(axis=0).values
    county_death = us_death_df.loc[us_death_df.Province_State == state_name].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key',"Population"],axis=1).sum(axis=0).values
    county_population = us_death_df.loc[us_death_df.Province_State ==state_name].Population.sum()
    county_infected = county_confirmed
    county_removed = county_death
    county_susceptible = county_population - county_infected
    return county_susceptible[start:start + days],county_infected[start:start + days],county_removed[start:days+start],county_population
def get_country(start,days,df_list,country_name = "US"):
    '''
    get country level data
    start: starting day since patient 0 in Wuhan
    days: duraton of sequential data
    df_list: the list of dataframes (confirmed, death, recovered, mobility)
    county_code: FIP of the county, defatul to Washington since it is the first patient
    '''
    us_confirmed_df = df_list[0]
    us_death_df = df_list[1]
    global_recover_df = df_list[2]   
    county_confirmed = us_confirmed_df.loc[us_confirmed_df.Country_Region == country_name].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1).sum(axis=0).values
    county_death = us_death_df.loc[us_death_df.Country_Region == country_name].drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State',
           'Country_Region', 'Lat', 'Long_', 'Combined_Key',"Population"],axis=1).sum(axis=0).values
    county_population = us_death_df.loc[us_death_df.Country_Region ==country_name].Population.sum()
    country_recovered = global_recover_df.loc[global_recover_df["Country/Region"] == country_name].drop(
    ["Province/State","Country/Region","Lat","Long"],axis= 1).values[0]
    county_infected = county_confirmed
    county_removed = county_death + country_recovered
    county_susceptible = county_population - county_infected
    return county_susceptible[start:start+days],county_infected[start:start+days],county_removed[start:start+days],county_population

def write_list_to_txt(filename,my_list):
    '''
    write sequential data into txt file for model building
    filename: the output text file name. s,i,r, or p.txt
    my_list: the sequential data to be written. 
    '''

    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)
    print("txt file at:  "+filename)
    

def get_california_counties(county_list):
    california = []
    for county in county_list:
        if "".join(county[:2]) == "06":
            california.append(county)
    return california
def read_nearby_file(file_name):
    '''
    file_name:the file location of the nearby neighbors json file
    returns: filtered dictionary, containing only california counties, also replace counties in the other state with "border"
    '''
    nearby_dict = read_github_json_file(file_name)
    
    county_list = list(nearby_dict.keys())
    california_county_list = get_california_counties(county_list)
    california_county_dict = {}

    for county in nearby_dict.keys():
        if county in california_county_list:
            california_county_dict[county] = nearby_dict[county]
    for key in california_county_dict.keys():
        for subkey in california_county_dict[key].keys():
            if list(california_county_dict[key][subkey])[:2]!=["0","6"]:
                #print(ca_county_neighbors[key][subkey])
                california_county_dict[key][subkey] = "border"
    return california_county_dict

            

