from mesa_geo import GeoAgent

import util
import transmission_rate as trans_rate

from scipy import stats 




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

#Plot
import matplotlib.pyplot as plt



# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# parser viz config data
viz_ini_file = 'vizparams.ini'

parser_viz = configparser.ConfigParser()
parser_viz.read(config_file_path_prefix + viz_ini_file)

default_section = parser_viz['DEFAULT_PARAMS']


# parser disease config data

disease_params_ini = 'diseaseparams.ini'
parser_dis = configparser.ConfigParser()
parser_dis.read(config_file_path_prefix + disease_params_ini)
incubation = parser_dis['INCUBATION']


# NPI config data


npi_params_ini = 'NPI.ini'
parser_npi = configparser.ConfigParser()
parser_npi.read(config_file_path_prefix + npi_params_ini)






# infectious curve config
###################################### 
# based on gamma fit of 10000 R code points

shape, loc, scale = (float(incubation['shape']), float(incubation['loc']), float(incubation['scale']))

# infectious curve
range_data = list(range(int(incubation['lower_bound']), int(incubation['upper_bound']) + 1))
infective_df = pd.DataFrame(
    {'x': range_data,
     'gamma': list(stats.gamma.pdf(range_data, a=shape, loc=loc, scale=scale))
    }
)
#########################################



class Human(GeoAgent):
            
    # plot config

    marker = default_section['marker']
    colordict = {"healthy": default_section['healthy'], 'exposed': default_section['exposed'], 'infectious': default_section['infectious']}
    edgedict = {"healthy": default_section['healthy_edge'], 'exposed': default_section['exposed_edge'], 'infectious': default_section['infectious_edge']}
    sizedict = {"healthy": default_section['healthy_size'], 'exposed': default_section['exposed_size'], 'infectious': default_section['infectious_size']}
    
    
    
    
    # dummy config for data collection
    viral_load = None
    
    
    
    # UPDATE 10/16: move stats to class level
     

    
    
    
    


    
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy'):
        super().__init__(unique_id, model, shape)
        
        # mask setup
        # defualt to no mask
        self.mask_type = None
        self.mask_passage_prob = trans_rate.return_mask_passage_prob(self.mask_type)
        
        # disease config

        self.health_status = health_status
        prevalence = float(parser_dis['ASYMPTOMATIC_PREVALENCE']['prevalence'])
        self.asymptomatic = np.random.choice([True, False], p = [prevalence, 1-prevalence])
        self.symptoms = False


        # TODO: vaccination should be parameterized (effective rate, etc.)
        self.tested = False
        self.vaccinated = False
        
        self.infective = False
        
        # symptom onset countdown config
        ##########################################
        # From 10000 lognorm values in R
        countdown = parser_dis['COUNTDOWN']
        shape, loc, scale =  (float(countdown['shape']), float(countdown['loc']), float(countdown['scale']))

        lognormal_dist = stats.lognorm.rvs(shape, loc, scale, size=1)

        
        num_days = max(int(countdown['lower_bound']), 
                       min(np.round(lognormal_dist, 0)[0], 
                           int(countdown['upper_bound'])))# failsafe to avoid index overflow
        self.symptom_countdown = int(num_days)
        #######################################
        
        
        # breathing attributes for transmission models
        self.breathing_rate = None
        self.breathing_activity = None

        
        self.room = room
        self.x = self.shape.x
        self.y = self.shape.y
        
        
        


        
        

 
    def update_shape(self, new_shape):
        self.shape = new_shape
        self.x = self.shape.x
        self.y = self.shape.y
        
    
    def __update(self):
        # UPDATE 10/16: reorganized things from Bailey's update
        # TODO: currently mask has no functionality other than reducing transmission distance, is this faithful?

        # mask wearing reduces droplet transmission max range
        # infection above max range is considered as aerosal transmission

        
        #if self.mask and not (self.model.activity[self.room.schedule_id] == 'lunch'):
        
        # chu distance multiplier would be < 0.001 above 30 feet for infection
        max_infect_dist = 30
        neighbors = self.model.grid.get_neighbors_within_distance(self, max_infect_dist)
        
        #else:
        #    neighbors = self.model.grid.get_neighbors_within_distance(self, int(parser_npi['NO_NPI']['infection_distance']))

        

                        
                        
                        
        if self.health_status == 'exposed' and self.infective:



            for neighbor in neighbors:

                # Check class is Human and are within the same room                           
                if issubclass(type(neighbor), Human) and self.__check_same_room(neighbor) :
                    if neighbor.unique_id != self.unique_id and (neighbor.health_status == 'healthy'):                   
                        # Call Droplet transmission function
                        temp_prob = self.droplet_infect(self, neighbor)
                        infective_prob = np.random.choice ([True, False], p = [temp_prob, 1-temp_prob])
                        if infective_prob and not neighbor.vaccinated:
                            neighbor.health_status = 'exposed'

    
    @staticmethod
    def droplet_infect(infected, uninfected):
        '''
        baseline transmission rate
        '''
        feet_to_meter = 1/3.2808
        distance = infected.shape.distance(uninfected.shape)*feet_to_meter
        
        # normalize symptom countdown value to infectious distribution value
        # 0 being most infectious
        # either -10 or 8 is proven to be too small of a chance to infect others, thus covering asympotmatic case
        
        countdown_norm = min(int(incubation['upper_bound']), max(int(incubation['lower_bound']), 0 - infected.symptom_countdown))
        transmission_baseline = infective_df[infective_df['x'] == countdown_norm]['gamma'].iloc[0]

     
        # Use Chu distance calculation ## see docs
        chu_distance_multiplier = 1/2.02
        distance_multiplier = (chu_distance_multiplier)**distance                                                        

        
        # approximate student time spent breathing vs talking vs loudly talking
        # upperbound baseline (worst case) for breathing activity is moderate_excercise and talking loud
        base_bfr = trans_rate.return_breathing_flow_rate('moderate_exercise')
        base_eai = trans_rate.return_exhaled_air_inf('talking_loud')
        
        
        inf_bfr_mult = trans_rate.return_breathing_flow_rate(infected.breathing_rate)/base_bfr 
        inf_eai_mult = trans_rate.return_exhaled_air_inf(infected.breathing_activity)/base_eai
        
        uninf_bfr_mult = trans_rate.return_breathing_flow_rate(uninfected.breathing_rate)/base_bfr 
        
        # take average of breathing flow rate of two agents
        bfr_multiplier = np.mean([inf_bfr_mult, uninf_bfr_mult])
        # we dont think the uninfected air exahale rate should be a factor here 
        breathing_type_multiplier = bfr_multiplier*inf_eai_mult
        
        

        # Mask Passage: 1 = no masks, .1 = cloth, .05 = N95
        mask_multiplier = np.mean([infected.mask_passage_prob, uninfected.mask_passage_prob])

        
        # Lunch special case: mask off during lunch time
        if infected.model.activity[infected.room.schedule_id] == 'lunch':
            mask_multiplier = 1
            
        
        # convert transmission rate / hour into transmission rate / step
        hour_to_fivemin_step = 5/60

        return transmission_baseline * distance_multiplier * breathing_type_multiplier * mask_multiplier * hour_to_fivemin_step


                        
    def __check_same_room(self, other_agent):
        '''
        check if current agent and other agent is in the same room
        
        the purpose of this function is to make sure to eliminate edge cases that one agent near the wall of its room
        infects another agent in the neighboring room
        
        this is at this iteration of code only implemented for class purpose, as unique id check is way more efficient
        
        later implementation should add attribute to human agent for current room
        
            other_agent: other agent to check
            returns: boolean value for if the two agents are in the same room
        '''
        same_room = True
        if self.model.activity[self.room.schedule_id] == 'class':
            same_room = (self.room.unique_id == other_agent.room.unique_id)
        return same_room
    
    
    def __move(self, move_spread = 4, location = None):
        '''
        Checks the current location and the surrounding environment to generate a feasbile range of destination (the area
        of a circle) for agent to move to.
        The radius of the circle is determined by agent's move_factor.
        Assigns new point to override current point.
        '''   
        
        if not location:
            location = self.room
        move_spread = location.shape.intersection(self.shape.buffer(move_spread))
        if hasattr(location, 'seating_pattern'):
            if location.seating_pattern == 'circular':
                move_spread = shapely.ops.cascaded_union(list(location.desks.values())).difference(move_spread)
        try:
            minx, miny, maxx, maxy = move_spread.bounds 
                
            while True:
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))            
                # check if point lies in true area of polygon
                if move_spread.contains(pnt):
                    self.update_shape(pnt)
                    break
        except:
            pass


    
    
    def plot(self):
        plt.plot(
            self.shape.x, self.shape.y, 
            marker=self.marker, 
            mec = self.edgedict[self.health_status],
            color = self.colordict[self.health_status],
            markersize = self.sizedict[self.health_status]
                )
        

                        
        
        
        





class Student(Human):
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy', mask_on=False):
        super().__init__(unique_id, model, shape, room, health_status)

        viz_ini_file = 'vizparams.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file_path_prefix + viz_ini_file)
        student_viz_params = parser['STUDENT']

        self.grade = self.room.room_type.replace('classroom_', '')
        
        
        # mask setup
        self.mask = mask_on      
        mask_probs = [eval(parser_npi['MASK_PROB']['cotton']), 
                      eval(parser_npi['MASK_PROB']['multilayer']),
                      eval(parser_npi['MASK_PROB']['surgical']),
                      eval(parser_npi['MASK_PROB']['n95'])]
        if mask_on:
            self.mask_type = np.random.choice(['Cotton', 'Multilayer', 'Surgical', 'N95'], p = mask_probs)
        else:
            self.mask_type = None
        self.mask_passage_prob = trans_rate.return_mask_passage_prob(self.mask_type)
        
        
        self.seat = Point(self.shape.x, self.shape.y)
        self.marker = student_viz_params['marker']
        
        
        

        self.out_of_place = False        
        self.prev_activity = None
        self.lunch_count = 0
        self.desks = None
        self.breathing_rate = None
        self.breathing_activity = None
        
        
        
        
        
                

    def step(self):
        self._Human__update()
        
        
        if self.model.schedule_type != "Simultaneous":
            self.advance()                

                
                
    def advance(self):
        activity = self.model.activity[self.room.schedule_id]
        # case 1: student in class
        if activity == 'class': 
            if self.prev_activity != activity:
                self.prev_activity = activity
                self.update_shape(self.seat)
                self.breathing_rate = 'resting'
                
                #TODO: The probability should be in init file
                self.breathing_activity = np.random.choice(
                    ['talking_whisper', 'talking_loud', 'breathing_heavy'], 
                    p=[0.2, 0.05, 0.75]
                )

                
            if self.room.prob_move:
                self.out_of_place = True
                
                #TODO: The probability should be in init file
                self.breathing_rate = 'light_exercise'
                self.breathing_activity = np.random.choice(
                    ['talking_whisper', 'talking_normal', 'talking_loud', 'breathing_heavy'], 
                    p=[0.05, 0.2, 0.6, 0.15]
                )
                
                
                self._Human__move()
            else:
                if self.out_of_place:
                    self.update_shape(self.seat)
                    self.out_of_place = False
                    
                
                
        # case 2: student in recess            
        elif activity == 'recess':
            
            #TODO: The probability should be in init file
            self.breathing_rate = np.random.choice(
                ['resting', 'moderate_exercise', 'light_exercise'],
                p = [0.2, 0.5, 0.3]
            )
            
            self.breathing_activity = np.random.choice(
                ['talking_whisper', 'talking_normal', 'talking_loud', 'breathing_heavy'], 
                p=[0.05, 0.2, 0.6, 0.15]
            )

            
            if self.prev_activity != activity:
                self.update_shape(util.generate_random(self.cohort.shape))
                self.prev_activity = activity
            
            self._Human__move(move_spread=5, location = self.cohort)
        
        
        
        # case 3: student having lunch
        elif activity == 'lunch':
            self.breathing_rate = 'resting'
            self.breathing_activity = 'breathing_heavy'
            #in class lunch case
            if self.model.inclass_lunch or self.grade != 'grade':
                if self.prev_activity != activity:
                    self.update_shape(self.seat)
                    self.prev_activity = activity
                    self.out_of_place = True
                    self._Human__move()
                else: 
                    if self.out_of_place:
                        self.update_shape(self.seat)
                        self.out_of_place = False

                    
            #in cafeteria lunch case
            else:
                if self.prev_activity != activity:
                    self.update_shape(util.generate_random(self.model.lunchroom.shape))
                    self.prev_activity = activity

                # enter lunch cafeteria, move free for 2 iteration
                if self.lunch_count < 2:
                    self._Human__move(move_spread=10, location = self.model.lunchroom)

                # finds seat, settle in seat until finish lunch
                elif self.lunch_count == 2:
                    self.update_shape(self.model.lunchroom.seats[0])
                    # remove seat from seat list if the student occupies it
                    self.model.lunchroom.seats = self.model.lunchroom.seats[1:]

                # release seat back to seat list
                elif self.lunch_count == 7:
                    self.model.lunchroom.seats.append(self.shape)
                    self.lunch_count = -1

            
                self.lunch_count += 1
    
                        
                            
                            
                            




                
                
    
class Teacher(Human):
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy',mask_on=True):
        super().__init__(unique_id, model, shape, room, health_status)
        
        
        # mask setup
        self.mask = mask_on      
        mask_probs = [eval(parser_npi['MASK_PROB']['cotton']), 
                      eval(parser_npi['MASK_PROB']['multilayer']),
                      eval(parser_npi['MASK_PROB']['surgical']),
                      eval(parser_npi['MASK_PROB']['n95'])]
        if mask_on:
            self.mask_type = np.random.choice(['Cotton', 'Multilayer', 'Surgical', 'N95'], p = mask_probs)
        else:
            self.mask_type = None
        self.mask_passage_prob = trans_rate.return_mask_passage_prob(self.mask_type)
        
        
        
        self.classroom = self.room # TODO: for future development enabling Teachers to move to other room during non-class time
        self.breathing_rate = None
        self.breathing_activity = None

        viz_ini_file = 'vizparams.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file_path_prefix + viz_ini_file)
        teacher_viz_params = parser['TEACHER']
        
        self.marker = teacher_viz_params['marker']
        self.edgedict = {"healthy": teacher_viz_params['healthy_edge'], 'exposed': teacher_viz_params['exposed_edge'], 'infectious': teacher_viz_params['infectious_edge']}
        self.sizedict = {"healthy": teacher_viz_params['healthy_size'], 'exposed': teacher_viz_params['exposed_size'], 'infectious': teacher_viz_params['infectious_size']}
    
    
    def step(self):
        self.breathing_rate = 'light_exercise'
        self.breathing_activity = np.random.choice(['talking_loud', 'talking_normal', 'breathing_heavy'], p=[0.5, 0.25, 0.25])
        self._Human__update()
        
                
        if self.model.schedule_type != "Simultaneous":
            self.advance()
            
    def advance(self):
        self._Human__move()
                
        