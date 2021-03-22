import random
import numpy as np
import sys, os



import json

import pandas as pd
from scipy.integrate import quad
import assumptions
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600
FEET_TO_METERS = 0.3048
SQ_FT_TO_SQ_M = 0.092903
CUBIC_FT_TO_METERS = 0.0283168
CUBIC_CM_TO_METERS = 1e-6
CUBIC_μM_TO_CUBIC_CM = 1e-12
CUBIC_M_TO_ML = 1e6

var = assumptions.var


































#Used to generate values with normal distribution

#Helper Functions
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    sys.stdout = sys.__stdout__

def get_air_changes_per_hour(cfm, room_volume):
    if str(cfm) == 'nan':
        #impute unknown cfm with arbitrary cfm
        print('VAV unknown. Imputed with arbitrary VAV of 800 CFM')
        cfm = 800
    elif cfm == 0:
        #Natural ACH of .15 if CFM is off provided
        return 0
    # if CFM is provided
    return (cfm * 60) / room_volume

def get_room_data(filepath, building_id, room_id):
    CUBIC_FT_TO_METERS = 0.0283168
    room_table = pd.read_csv(filepath)
    room_table = room_table[room_table["Building"] == building_id]
    room_dic = {}
    if len(room_table.loc[room_table['Room'] == room_id]) == 0:
        return print("User input error: Room " + room_id + " not found.")
    
    room_table.loc[room_table['Room'] == room_id]['Area']
    #Room Area in square ft.
    room_dic['room_area'] = room_table.loc[room_table['Room'] == room_id]['Area'].item()
    #Room Height. Average room height of 10 ft is chosen if nan
    room_hght = room_table.loc[room_table['Room'] == room_id]['Height'].item()
    if np.isnan(room_hght):
        print(room_id + ' Room height not found. Average room height of 10 ft imputed.')
        room_hght = 10
    room_dic['room_hght'] = room_hght
    
    #CFM range. If no CFM is provided min is chosen by default
    cfm_min = room_table.loc[room_table['Room'] == room_id]['VAVmin'].item()
    cfm_max = room_table.loc[room_table['Room'] == room_id]['VAVmax'].item()
    cfm_recommended = room_table.loc[room_table['Room'] == room_id]['VAVrecommended'].item()
    room_dic["cfm_min"] = float(cfm_min)
    room_dic["cfm_recommended"] = float(cfm_recommended)
    room_dic["cfm_max"] = float(cfm_max)
     
    #V is volume of room
    room_dic['room_volume'] = room_dic['room_area'] * room_hght
    #Unit Conversion
    
    room_dic['room_volume_m'] = room_dic['room_area'] * room_hght * CUBIC_FT_TO_METERS
    
    return room_dic

def get_quanta_emmission_rate(activity, expiratory_activity, mask_tf, var = var):
    CUBIC_μM_TO_CUBIC_CM = 1e-12
    CUBIC_M_TO_ML = 1e6
    Dc = var['droplet_conc'][expiratory_activity]
    Dv = var['droplet_vol']
    if mask_tf:
       #Because of the varying quality and effectiveness in masks for the general public 
       #we assume a conservatively low mask efficacy for smaller particles and increased
       #efficacy for larger particles. 
       #Source: https://doi.org/10.1016/j.ajic.2007.07.008
        summation = sum([var['mask_efficacy']['.8μm'] * Dc['.8μm'] * (Dv['.8μm'] * CUBIC_μM_TO_CUBIC_CM),
                     var['mask_efficacy']['1.8μm'] * Dc['1.8μm'] * (Dv['1.8μm'] * CUBIC_μM_TO_CUBIC_CM),
                     var['mask_efficacy']['3.5μm'] * Dc['3.5μm'] * (Dv['3.5μm'] * CUBIC_μM_TO_CUBIC_CM),
                     var['mask_efficacy']['5.5μm'] * Dc['5.5μm'] * (Dv['5.5μm'] * CUBIC_μM_TO_CUBIC_CM)])
    else:
        summation = sum([Dc['.8μm'] * (Dv['.8μm'] * CUBIC_μM_TO_CUBIC_CM),
                     Dc['1.8μm'] * (Dv['1.8μm'] * CUBIC_μM_TO_CUBIC_CM),
                     Dc['3.5μm'] * (Dv['3.5μm'] * CUBIC_μM_TO_CUBIC_CM),
                     Dc['5.5μm'] * (Dv['5.5μm'] * CUBIC_μM_TO_CUBIC_CM)])
    #Convert IR from cubic meters to mililiter
    return var['cv'] * var['ci'] * (var['IR'][activity] * CUBIC_M_TO_ML) * summation
      
#Infection Risk Calculator
def infection_risk(t, building_id, room_id, n_occupants, activity, expiratory_activity, room_data_path, mask_tf, cfm, var = var):
    ERq = get_quanta_emmission_rate(activity, expiratory_activity, mask_tf)
    room_dic = get_room_data(room_data_path, building_id, room_id)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Air Changes per Hour
    cfm = get_vav(room_data_path, building_id, room_id, cfm)
    air_change_rate = get_air_changes_per_hour(cfm, room_dic['room_volume'])
    
    
    
    
    ##To calculate infection rate we will aggregate the past week of testing for UC San Diego (last updated: 12/10/20)
    #Source: https://returntolearn.ucsd.edu/dashboard/index.html
    n_infected = var['infection_rate'] * n_occupants # probability of getting infected given number of occupants
    #if n_infected < 1:
        #n_infected = 1
    #Infectious virus removal rate
    ivrr = air_change_rate + var['deposition_rate'] + var['viral_inactivation']
    
    def quanta_concentration(t, I = n_infected, ERq = ERq, IVRR = ivrr, V = room_dic['room_volume_m'], n0 = var['initial_quanta']):
        return ((ERq * I) / (IVRR * V)) + (n0 + ((ERq * I) / IVRR)) * ((np.e**(-IVRR * t)) / V)

    ans, err = quad(quanta_concentration, 0, t)
    
    risk = 1 - np.e**(-var['IR'][activity] * ans)
    
    print('It is estimated that an individual has ' + str(risk * 100) +'% chance to be infected')
    #print('It is estimated that ' + str(risk) + ' x ' + str(n_occupants) + ' = ' + str(int(risk * n_occupants)) + ' susceptible occupants will be infected')
    
    
    return risk

#For user interface
def ui_calc(activity_dropdown, building_input, room_input, time_input, occupant_input, mask_tf, rid_path, cfm_max = "max"):
    
    
    
    
    
    print(activity_dropdown)
    #Given the user inputted activity we must assume inhalation rate and expiratory activities in 
    #order to accurately provide a quantum emmission rate.
    if activity_dropdown == 'Lecture':
        print("lecture")
        #Simulate lecture with average of resting/whispering and speaking/standing
        #perhaps make this information available to users by providing  a drop down that allows user to 
        #choose ratio of two actions/exp_actions during the events
        act1 = 'resting'
        act2 = 'standing'
        #The expiratory action is assumed to be 
        exp_act1 = 'whispering'
        exp_act2 = 'speaking'
        ir1 = infection_risk(time_input, building_input, room_input, occupant_input, act1, exp_act1, rid_path, mask_tf, cfm_max)
        ir2 = infection_risk(time_input, building_input, room_input, occupant_input, act2, exp_act2, rid_path, mask_tf, cfm_max)
        total_ir  = (ir1 + ir2) / 2
    elif activity_dropdown == 'Studying':
        print("studying")
        #Simulate studying with average of resting/whispering and speaking/standing
        act1 = 'resting'
        act2 = 'standing'
        exp_act1 = 'speaking'
        exp_act2 = 'whispering'
        ir1 = infection_risk(time_input, building_input, room_input, occupant_input, act1, exp_act1, rid_path, mask_tf, cfm_max)
        ir2 = infection_risk(time_input, building_input, room_input, occupant_input, act2, exp_act2, rid_path, mask_tf, cfm_max)
        total_ir  = (ir1 + ir2) / 2
    elif activity_dropdown == 'Singing':
        print("singing")
        #Simulate singing by assuming occupants are singing and standing
        act1 = 'standing'
        exp_act1 = 'singing'
        total_ir = infection_risk(time_input, building_input, room_input, occupant_input, act1, exp_act1, rid_path, mask_tf, cfm_max)
    elif activity_dropdown == 'Social':
        print("social")
        #Simulate singing by assuming occupants are doing light exercise and talking
        act1 = 'light_exercise'
        exp_act1 = 'speaking'
        total_ir = infection_risk(time_input, building_input, room_input, occupant_input, act1, exp_act1, rid_path, mask_tf, cfm_max)
    else:
        #Simulate singing by assuming occupants are doing heavy exercise and talking
        print("else")
        act1 = 'heavy_exercise'
        exp_act1 = 'speaking'
        total_ir = infection_risk(time_input, building_input, room_input, occupant_input, act1, exp_act1, rid_path, mask_tf, cfm_max)
    return total_ir

def get_vav(room_data_path, building_id, room_id, cfm): 
    room_dic = get_room_data(room_data_path, building_id, room_id)
        
    if cfm == "max":
        cfm = room_dic["cfm_max"]
    elif cfm == "min":
        cfm = room_dic["cfm_min"]
        
    elif cfm == "median":
        cfm = (room_dic["cfm_max"] + room_dic["cfm_min"]) / 2 
        
    elif cfm == "recommended": 
        
        if room_dic["cfm_recommended"] is not None: 
            cfm = room_dic["cfm_recommended"]
        
        
        
        else: 
            cfm = "unavailable"
    else: 
        cfm = float(cfm)
    return cfm
