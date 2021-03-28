import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
from numpy import linalg as LA
import sys
from src.data.etl import *

def build_model(path):
    '''
    the overall method that will collect data from local files, and read params from json file, and do the model training
    path: takes in the path where the s,i,r,p text files are located (a folder name)
    returns: model parameters beta and d
    beta is the infection rate
    d is the infectious duration
    '''
    
    s_path = path+"s.txt"
    i_path = path+"i.txt"
    r_path = path+"r.txt"
    p_path = path+"p.txt"
    
    with open(s_path,"r") as f:
        s = f.read().split("\n")
        s = [int(x) for x in s if x!=""]
    with open(i_path,"r") as f:
        i = f.read().split("\n")
        i = [int(x) for x in i if x!=""]
    with open(r_path,"r") as f:
        r = f.read().split("\n")
        r = [int(x) for x in r if x!=""]
        
    with open(p_path,"r") as f:
        p = f.read().split("\n")[0]
        p = int(p)
    with open('config/model_params.json') as fh:
        model_cfg = json.load(fh)

    iterations = model_cfg["iterations"]

    
    learning_rate = tune_learning_rate(s,i,r,p)
    betas,d = calculate(s,i,r,p,learning_rate,iterations)
    return betas,d
    
def calculate_hessian(s,i,r,population):
    '''
    create hessian matrix (second derivatives of beta and d)
    '''
    result_beta_second = 0
    result_epsilon_second = 0
    result_both_second = 0
    for n in range(len(s)-1):
        result_beta_second += 4*(s[n] * i[n]/population) **2 
        result_epsilon_second += 4*i[n]
        result_both_second += -2*s[n]*i[n]**2/population
    return result_beta_second/len(s),result_epsilon_second/len(s),result_both_second/len(s)
            
def tune_learning_rate(s,i,r,p):
    '''
    calculate 0.1/eigenvalue of hessian matrix, our learning rate
    '''
    
    top_left,bottom_right,the_other_two = calculate_hessian(s,i,r,p)
    w, v = LA.eigh(np.array([[top_left, the_other_two], [the_other_two, bottom_right]]))
    lip_constant = w[w>0][0]
    return 0.1/lip_constant

def calculate_gradient(s,i,r,population,beta,epsilon):
    '''
    get gradients at each iteration of gradient descent
    '''
    result1 = 0 #continue adding to solve for beta
    result2 = 0 #continue adding to solve for 1/D aka epsilon
    for n in range(len(s)-1):
        result1 += 2*(s[n+1]-s[n]+beta*s[n]*(i[n]/population))*(s[n]*i[n]/population)
        result1 += 2*(i[n+1]-i[n]-beta*s[n]*(i[n]/population) + i[n]*epsilon)*(-s[n]*i[n]/population)
        
        result2 += 2*(i[n+1]-i[n]+i[n]*epsilon-beta*i[n]*s[n]/population)*(i[n])
        result2 += 2*(r[n+1]-r[n]-i[n]*epsilon)*(-i[n])
        
    return result1,result2
def calculate(s,i,r,population,learning_rate,iterations):
    '''
    gradient descent method
    returns: beta and d
    we initialized beta to be 0.2 and D to 14 from existing knowledge so we can get to convergence faster
    '''
    beta = 0.2
    epsilon = 1/14
    
    loss = 0
    length = len(s)
    betas = []
    ds = []
    
    for itera in range(iterations): # do it for 10 iterations.
        
        loss1,loss2 = calculate_gradient(s,i,r,population,beta,epsilon)
        beta_new = beta - learning_rate* loss1/length #0.001 is the learning rate
        epsilon_new = epsilon - learning_rate * loss2/length
        if (beta_new == beta) & (epsilon_new == epsilon):
            return beta_new,1/epsilon_new
           
        beta = beta_new
        epsilon = epsilon_new
        
        betas.append(beta)
        ds.append(1/epsilon)
        
    return betas,ds

def calculate_i_delta(infection,dic,county_name,us_confirmed_df):
    center_infected = infection[county_name]
    north_county = dic[county_name]["north"]
    south_county = dic[county_name]["south"]
    
    infected_dic = {} 
    a = np.array(list(dic[county_name].values()))
    li = ["north","south","west","east"]
    if np.count_nonzero(a == "border") == 3:
        
        for side in li:
            if dic[county_name][side]!="border":
                only_neighbor_county = dic[county_name][side]
                infected_dic[side] = infection[only_neighbor_county]
                li.remove(side)
                for k in li:
                    infected_dic[k] = center_infected
                    lat_difference = calculate_lat_long_difference(county_name,only_neighbor_county,us_confirmed_df)[0] * 2
                    long_difference = calculate_lat_long_difference(county_name,only_neighbor_county,us_confirmed_df)[1] * 2
        first_term = (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
        second_term =  (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
        
        return first_term+second_term
    
    
    
    if (north_county == "border") & (south_county!="border"):
        infected_dic["south"] = infection[south_county]
        infected_dic["north"] = center_infected
        long_difference = calculate_lat_long_difference(county_name,south_county,us_confirmed_df)[1] * 2 #times two, mirror the lat and long of opposite
    elif (north_county != "border") & (south_county =="border"):
        infected_dic["south"] = infection[north_county]
        infected_dic["north"] = center_infected
        long_difference = calculate_lat_long_difference(north_county,county_name,us_confirmed_df)[1] * 2 #times two, mirror the lat and long of opposite
    elif (north_county != "border") & (south_county !="border"):
        infected_dic["south"] = infection[south_county]
        infected_dic["north"] = infection[north_county]
        long_difference = calculate_lat_long_difference(north_county,south_county,us_confirmed_df)[1] #times two, mirror the lat and long of opposite
        
        
    west_county = dic[county_name]["west"]
    east_county = dic[county_name]["east"]
    
    
    if (west_county == "border") & (east_county!="border"):
        infected_dic["east"] = infection[east_county]
        infected_dic["west"] = center_infected
        lat_difference = calculate_lat_long_difference(county_name,east_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    elif (west_county != "border") & (east_county =="border"):
        infected_dic["west"] = infection[west_county]
        infected_dic["east"] = center_infected
        lat_difference = calculate_lat_long_difference(county_name,west_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    elif (west_county != "border") & (east_county !="border"):
        infected_dic["west"] = infection[west_county]
        infected_dic["east"] = infection[east_county]
        lat_difference = calculate_lat_long_difference(west_county,east_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    
    
    if "north" not in infected_dic.keys():
        return (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
    if "west" not in infected_dic.keys():
        return (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
    
    
    first_term = (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
    second_term =  (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
    return first_term+second_term

def check_prediction(FIP_list,prediction_dic,us_confirmed_df,date):
    df = us_confirmed_df.loc[us_confirmed_df.FIPS.isin(FIP_list)]
    
    df["predicted"] = df["FIPS"].replace(prediction_dic)
    df["percent_difference"] = (df["predicted"] - df[date]) / df[date]
    df["predicted"] = df.predicted.astype(int)
    return df[["Admin2",date,"predicted","percent_difference"]]
def calculate_delta_initialization(dic,county_name,us_confirmed_df,t2_date): 
    '''
    dic: the dictionary containing neighboring county FIPs
    county_name: the FIP of the county of interest
    us_confirmed: dictionary containing infected case numbers for us to initialize our dictionary
    t2_date: the previous day (our prediction is based on), so we can look up the infection stats from the confirmed df
    '''
    
    infected_dic = {}
    
    center_infected = us_confirmed_df.loc[us_confirmed_df.FIPS == county_name][t2_date].values[0]
    infected_dic["center"] = center_infected
    north_county = dic[county_name]["north"]
    south_county = dic[county_name]["south"]
    
    
    a = np.array(list(dic[county_name].values()))
    li = ["north","south","west","east"]
    if np.count_nonzero(a == "border") == 3:
        
        for side in li:
            if dic[county_name][side]!="border":
                only_neighbor_county = dic[county_name][side]
                infected_dic[side] = us_confirmed_df.loc[us_confirmed_df.FIPS == only_neighbor_county][t2_date].values[0]
                li.remove(side)
                for k in li:
                    infected_dic[k] = center_infected
                    lat_difference = calculate_lat_long_difference(county_name,only_neighbor_county,us_confirmed_df)[0] * 2
                    long_difference = calculate_lat_long_difference(county_name,only_neighbor_county,us_confirmed_df)[1] * 2
        first_term = (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
        second_term =  (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
        
        return first_term+second_term
    if (north_county == "border") & (south_county!="border"):
        infected_dic["south"] = us_confirmed_df.loc[us_confirmed_df.FIPS == south_county][t2_date].values[0]
        infected_dic["north"] = center_infected
        long_difference = calculate_lat_long_difference(county_name,south_county,us_confirmed_df)[1] * 2 #times two, mirror the lat and long of opposite
    elif (north_county != "border") & (south_county =="border"):
        infected_dic["south"] = us_confirmed_df.loc[us_confirmed_df.FIPS == north_county][t2_date].values[0]
        infected_dic["north"] = center_infected
        long_difference = calculate_lat_long_difference(north_county,county_name,us_confirmed_df)[1] * 2 #times two, mirror the lat and long of opposite
    elif (north_county != "border") & (south_county !="border"):
        infected_dic["south"] = us_confirmed_df.loc[us_confirmed_df.FIPS == south_county][t2_date].values[0]
        infected_dic["north"] = us_confirmed_df.loc[us_confirmed_df.FIPS == north_county][t2_date].values[0]
        long_difference = calculate_lat_long_difference(north_county,south_county,us_confirmed_df)[1] #times two, mirror the lat and long of opposite
        
        
    west_county = dic[county_name]["west"]
    east_county = dic[county_name]["east"]
    
    
    if (west_county == "border") & (east_county!="border"):
        infected_dic["east"] = us_confirmed_df.loc[us_confirmed_df.FIPS == east_county][t2_date].values[0]
        infected_dic["west"] = center_infected
        lat_difference = calculate_lat_long_difference(county_name,east_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    elif (west_county != "border") & (east_county =="border"):
        infected_dic["west"] = us_confirmed_df.loc[us_confirmed_df.FIPS == west_county][t2_date].values[0]
        infected_dic["east"] = center_infected
        lat_difference = calculate_lat_long_difference(county_name,west_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    elif (west_county != "border") & (east_county !="border"):
        infected_dic["west"] = us_confirmed_df.loc[us_confirmed_df.FIPS == west_county][t2_date].values[0]
        infected_dic["east"] = us_confirmed_df.loc[us_confirmed_df.FIPS == east_county][t2_date].values[0]
        lat_difference = calculate_lat_long_difference(west_county,east_county,us_confirmed_df)[0] * 2 #times two, mirror the lat and long of opposite
    
    #if at this point, north is still border, means that both north and south didn't have neighbors
    if "north" not in infected_dic.keys():
        return (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
    if "west" not in infected_dic.keys():
        return (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
    
    first_term = (infected_dic["west"] + infected_dic["east"] - 2 * center_infected)/lat_difference ** 2
    second_term =  (infected_dic["north"] + infected_dic["south"] - 2 * center_infected)/lat_difference ** 2
    return first_term+second_term

def calculate_i_t1(t2_date,dic,us_confirmed_df,beta,d,us_death_df,us_mobility_df,n=10):
    '''
    t2_date: date of the previous day
    dic: dictionary containing each county's four neighboring counties. e.g. dic["San Diego"]["east"] = "Imperial"
    beta: 0.2
    d: 14
    '''
    dic_predictions = {}
    mobility_dic = {}
    population_dic = {}

    dt = 1.0/n
    
    #initialization, initialize the dictionary, the values will be the infected cases in t2
    for county_name in dic.keys():
        infected_t2 =  us_confirmed_df.loc[us_confirmed_df.FIPS == county_name][t2_date].values[0]
        try:
            m_t2 = us_mobility_df.loc[us_mobility_df.FIPS == county_name][t2_date].values[0]
        except:
            m_t2 = us_mobility_df[t2_date].mean()
        mobility_dic[county_name] = m_t2
        
        #death_t2 = us_death_df.loc[us_death_df.FIPS == county_name][t2_date].values[0]
        population = find_population_for_county(county_name,us_death_df)
        population_dic[county_name] = population
        s_t2 = population - infected_t2
        
        first_term = infected_t2 #previous day number, infected number at t0
        
    
        second_term = -infected_t2/d
        third_term = beta * (infected_t2/population) * s_t2
        fourth_term = calculate_delta_initialization(dic,county_name,us_confirmed_df,t2_date) * m_t2
        
        
            
        dic_predictions[county_name] = sum([first_term,sum([second_term,third_term,fourth_term])*dt])
    for i in range(n-1):
        def calculate_for_dt(dic_predictions,population_dict,mobility_dict,d,beta):
            for county_name in dic_predictions.keys():
                infected_prev = dic_predictions[county_name]
                m = mobility_dict[county_name]
                population = population_dict[county_name]
                s = population - infected_prev
                first_term = infected_prev #t(n-1) infected number of the previous t
                second_term = -infected_prev/d
                third_term = beta * (infected_prev/population) * s
                fourth_term = calculate_i_delta(dic_predictions,dic,county_name,us_confirmed_df) * m
                fourth_term = 0
                dic_predictions[county_name] = sum([first_term,sum([second_term,third_term,fourth_term])*dt])
            return dic_predictions
        dic_predictions = calculate_for_dt(dic_predictions,population_dic,mobility_dic,d,beta)
        #calculate_for_dt, run for n-1 times
        
    
    
    
    return dic_predictions


