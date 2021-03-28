from mesa_geo import GeoAgent

import util

import warnings

#Shapely Imports
from shapely.geometry import Point
import shapely


#Data Analysis 
import geopandas as gpd
import pandas as pd
import numpy as np
import random

# Configuration Data and Files
import configparser





# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# school config data


school_params_ini = 'schoolparams.ini'
parser_school = configparser.ConfigParser()
parser_school.read(config_file_path_prefix + school_params_ini)
cohort_config = parser_school['COHORT']



# UPDATE Christmas add cohort agent
class Cohort(GeoAgent):
    # dummy config for data collection
    health_status = None
    symptoms = None
    x = None
    y = None
    viral_load = None
    
    # get cohort area config
    max_intersection_prob = eval(cohort_config['max_intersection_prob'])
    
    def __init__(self, unique_id, model, shape, students, recess_yard, radius):
        super().__init__(unique_id, model, shape)
        self.radius = radius
        self.students = students
        self.recess_yard = recess_yard
        for student in self.students:
            student.cohort = self
        
    def step(self):
        # currently does nothing
        if self.model.schedule_type != "Simultaneous":
            self.advance()
    
    def advance(self):
        # generate a random cohort recess location at the beginning of day
        if self.model.step_count == 0:
            while True:
                self.shape = self.__update_cohort_location()
                if self.__check_intersection():

                    break
                
            
    def __update_cohort_location(self):
        
        #UPDATE: add max toleration for intersection check
        # avoid crowded recess yard which 
        #makes random generation of cohort location with given intersection prob impossible
        max_toleration = 10
        for i in range(max_toleration):
            center = util.generate_random(self.recess_yard.shape)
            shape = center.buffer(self.radius)
            if self.recess_yard.shape.contains(shape):

                return shape
        return shape
    
    
    def __check_intersection(self):
        other_cohorts = [a for a in list(self.model.grid.get_intersecting_agents(self)) if issubclass(type(a), Cohort)]
        for a in other_cohorts:
            if a.unique_id != self.unique_id:
                if self.shape.intersection(a.shape).area > self.max_intersection_prob * self.shape.area:

                    return False   
        return True
            
            
            
    

