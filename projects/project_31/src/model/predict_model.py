import sys
from src.data.etl import *
from src.model.train_model import *
import json
def print_prediction_error():
    with open('config/model_params.json') as fh:
        model_cfg =json.load(fh)
    curr_day = model_cfg["curr_day"]
    prediction_day = model_cfg["prediction_day"]
    test_beta = model_cfg["test_beta"]
    test_d = model_cfg["test_d"]
    us_confirmed_df,us_death_df,global_recover_df,mobility = retrieve_data()
    fips_neighbor_dict = generate_sc_neighbor_dictionary(us_confirmed_df)
    d1 = calculate_i_t1(curr_day,fips_neighbor_dict,us_confirmed_df,test_beta,test_d,us_death_df,mobility,n=5)
    li = list(d1.keys())
    
    prediction_df =  check_prediction(li,d1,us_confirmed_df,prediction_day)
    print(prediction_df)
    return prediction_df.percent_difference.mean()

def get_prediction_df(nearby_file,date1,date2,beta,d,us_confirmed_df,us_death_df,mobility,n=5):
    '''
    nearby_file:the file location of nearby file
    date1: the previous day
    date2: the second day (prediction day)
    beta: infection rate
    d: infection duration
    n: default at 5, the dt
    '''
    
    california_county_dict = read_nearby_file(nearby_file)
    fips_neighbor_dict =california_county_dict
    d1 = calculate_i_t1(date1,fips_neighbor_dict,us_confirmed_df,beta,d,us_death_df,mobility,n=5)
    li = list(d1.keys())
    return_df = check_prediction(li,d1,us_confirmed_df,date2).rename({"Admin2":"county_name",date2:"actual"},axis=1).reset_index(drop=True)
    
    return return_df