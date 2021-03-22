from mesa_geo import GeoAgent, GeoSpace
from mesa.time import BaseScheduler
from mesa.time import SimultaneousActivation
from mesa import datacollection
from mesa.datacollection import DataCollector
from mesa import Model
import pandas as pd
import numpy as np
import random
import shapely
from shapely.geometry import Polygon, Point, LineString
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go



class RunAll:

    def __init__(self):
        a = 2

    # runs the program by using the parameters to set of the simulation, and simulating each step
    def run_program(self, params):
        # saving the parameters
    	student_num = params['student_num']
        # initializing the simulation
    	mod = CollectableModel(Student, params)
        # simulating each step
    	for _ in range(int(params['steps'])):
    		mod.step()
        # collecting agent data results for visualisation purposes
    	agent_data = mod.datacollector.get_agent_vars_dataframe().reset_index()
        # reformatting the collected agent data
    	df = agent_data[['Step','sick?','present']].groupby(['Step']).sum().reset_index()
    	df.index = df.Step
    	df = df[['Step', 'sick?','present']]
    	df['healthy'] = df['present'] - df['sick?']
    	df = df[['sick?','healthy','present']]
        #returning the reformatted data back to run.py
    	return agent_data, df

class NaiveAgent(GeoAgent):
    def __init__(self, unique_id, model, shape):

        super().__init__(unique_id, model, shape)

class Student(NaiveAgent):
    """Student Class
    
    this class defines a student for our model. This student has many parameters such as location,
    whether present in bus or not, whether sick or not, wheter they are spreading the virus or not,
    which bus stop they belong to, and some not yet completed parameters such as mask wearing.
    
    """

    # initializing a student with given parameters
    def __init__(self, unique_id, model, shape, sick, mask, params, spreads, bus_stop, breathe_rate):
        super().__init__(unique_id, model, shape)
        self.present = False
        self.stop_counter = 0
        self.breathe = 0
        self.spreads = spreads
        self.params = params
        self.sick = sick
        self.mask = mask
        self.x = self.shape.x
        self.y = self.shape.y
        self.bus_stop = bus_stop
        self.breathe_rate = breathe_rate

    def step(self):
        #place holder to let us change the seats of passangers in the future
        a = 0
        #self.sit_down(np.random.randint(bus_cols), np.random.randint(bus_rows))


    # this method is not activated since the students are not changing seats in the middle of trip
    def sit_down (self, newx, newy):

        self.x, self.y = newx, newy # update the location
        self.shape = Point(self.x, self.y) # update shape


    # this happens every step
    def advance(self):
        self.take_ride()

    
    def take_ride(self):

        # adds one minute every time, this method helps realize when the bus reaches the stop of the student
        self.stop_counter += 1
        # total amount of air breathed is updated
        self.breathe += self.breathe_rate[0]
        # if the student had hopped onto the bus
        if self.stop_counter > self.bus_stop:
            # the student is not marked present in the bus
            self.present = True
            # if the student is sick and spreading
            if self.sick == True and self.spreads == True:
                # affect neighbors is different radiuses with different rates (the closer, the higher the danger)

                # get neighbors in different radiuses
                highriskneighbors = self.model.grid.get_neighbors_within_distance(self, self.params['highrisk_radius'])
                medriskneighbors = self.model.grid.get_neighbors_within_distance(self, self.params['medrisk_radius'])
                lowriskneighbors = self.model.grid.get_neighbors_within_distance(self, self.params['lowrisk_radius'])


                # affect the neighbors in differnt radiuses
                highrisk = self.params['highrisk'] * self.breathe
                medrisk = self.params['medrisk'] * self.breathe
                lowrisk = self.params['lowrisk'] * self.breathe
                for n in highriskneighbors:
                    if random.random() < highrisk and n.present:
                        n.sick = True

                for n in medriskneighbors:
                    if random.random() < medrisk and n.present:
                        n.sick = True

                for n in lowriskneighbors:
                    if random.random() < lowrisk and n.present:
                        n.sick = True

class MyModel(Model):
    def __init__(self, student_class, params):#student_num, sick_num, bus_cols, bus_rows, highrisk_radius, highrisk, lowrisk_radius, lowrisk, masks):

        # mesa required attributes
        self.running = True # determines if model should keep on running
        # should be specified to false when given conditions are met

        self.grid = GeoSpace() # To learn more about other type of space grid, check mesa documentation
        self.schedule = SimultaneousActivation(self) # scheduler dictates model level agent behavior, aka. step function
        # Here we are using a BaseScheduler which computes step of the agents by order
        # To learn more about other type of scheduler, check mesa documentation
        student_num = params['student_num']
        sick_num = params['sick_num']
        bus_cols = params['bus_cols']
        bus_rows = params['bus_rows']
        breathe_rate = params["breathe_rate"],
        windows_open = params["windows_open"],
        seating_array = params["seating_array"],
        highrisk_radius = params['highrisk_radius']
        highrisk = params['highrisk']
        lowrisk_radius = params['lowrisk_radius']
        lowrisk = params['lowrisk']
        masks = params['masks']
        bus_stop_student_count = params['bus_stop_student_count']
        bus_stop_minutes_count = params['bus_stop_minutes_count']

        sick = 0
        locs = []


        # initializing the model by making the students and putting them into their seats
        stop_counter = 0
        for i in range(student_num):
            bus_stop = bus_stop_minutes_count[bus_stop_student_count.index(min(filter(lambda x: x > stop_counter,bus_stop_student_count)))]
            stop_counter +=1
            # finding an empty seat for the next student
            new = False
            while (new == False):
                loc = (np.random.randint(bus_cols),np.random.randint(bus_rows))
                if loc not in locs:
                    new = True
                    locs.append(loc)
            pnt = Point(loc)
             # adding sick and healthy students
            if sick < sick_num:
                sick += 1
                a = Student(model=self, shape=pnt, unique_id="Passenger #" + str(i), sick = True, mask = False, params=params, spreads = True, bus_stop = bus_stop, breathe_rate = breathe_rate )
            else:
                a = Student(model=self, shape=pnt, unique_id="Passenger #" + str(i), sick = False , mask = False, params=params, spreads = False, bus_stop = bus_stop, breathe_rate = breathe_rate)

            self.grid.add_agents(a)
            self.schedule.add(a)

    def step(self):
        '''
        step function of the model that would essentially call the step function of all agents
        '''

        self.schedule.step()
        #self.schedule.advance()
        self.grid._recreate_rtree() # this is some history remaining issue with the mesa-geo package
        # what this does is basically update the new spatial location of the agents to the scheduler deliberately

class CollectableModel(MyModel):
    def __init__(self, student_class, params):#student_num, sick_num, bus_cols, bus_rows, highrisk_radius, highrisk, lowrisk_radius, lowrisk, masks):
        '''
        initialize the model with a GeoSpace grid
        agent_class: the type of agent you want to initialize in this model
                     normally not an input parameter as models are mostly tied to specific agent types
                     here we want to reuse thi model later
        agent_N: number of agents to intialize the model with
        '''
        super().__init__(student_class, params)#student_num, sick_num, bus_cols, bus_rows, highrisk_radius, highrisk, lowrisk_radius, lowrisk, masks)


        #data collect init
        # the model reporter specifies the data that should be collected from the model in each collect call
        # since we have no model data to be collected in this example, it is not specified
        # the format is the same as the agent_reporters

        # the agent reporter specifies the data that should be collected from the agents in each collect call
        agent_reporters = {"x": "x",
                           "y": "y",
                           "present": "present",
                           "sick?": "sick",
                           "spreads": "spreads"}


        # both dictionaries can take in lambda function as well
        # however, for efficiency purposes, getting attribute is recommanded
        # check mesa documentation for more details in the DataCollector page


        self.datacollector = datacollection.DataCollector(agent_reporters=agent_reporters)#, model_reporters=model_reporters)

    # simulate step by step
    def step(self):
        '''
        step function of the model that would essentially call the step function of all agents
        '''
        self.schedule.step()
        self.grid._recreate_rtree()
        self.datacollector.collect(self)
