from mesa_geo import GeoAgent

import human_agent
import util
# Aerosol Transmission
import aerosol_new
import transmission_rate as trans_rate


import math

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

#Plot
import matplotlib.pyplot as plt


# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# school config data
school_params_ini = 'schoolparams.ini'
parser_school = configparser.ConfigParser()
parser_school.read(config_file_path_prefix + school_params_ini)
population_config = parser_school['SCHOOL_POPULATION']
school_intervention_params = parser_school['INTERVENTION']
    

class Classroom(GeoAgent):
    
    
    # dummy config for data collection
    health_status = None
    symptoms = None
    x = None
    y = None
    
    
    def __init__(self, unique_id, model, shape, room_type):
        super().__init__(unique_id, model, shape)
        #self.occupants = []
        self.aerosol_transmission_rate = []
        #self.occupants = occupants #List of all occupants in a classroom
        #self.barrier = barrier_type
        self.room_type = room_type
        self.seating_pattern = None
        self.viral_load = 0
        self.prob_move = False
        self.schedule_id = None
        
        
        # volume
        self.floor_area = shape.area
        self.height = 12

        # airflow ventiliation type
        self.environment = eval(school_intervention_params['ventilation_type'])
    
    def step(self):
        # roll weighted probability for current step having in-class activity
        self.prob_move = np.random.choice([True, False], p = [0.2, 0.8])

                
        if self.model.schedule_type != "Simultaneous":
            self.advance()
        
        
        
        
    def advance(self):
        """
            #TODO: needs comment
        """
        # UPDATE 3/1
        # recess yards should have no airosol transmission
        if self.room_type ==  'recess_yard':
            transmission_rate = 0
            self.aerosol_transmission_rate.append(transmission_rate)
            return 
            
        occupants = [a for a in list(self.model.grid.get_intersecting_agents(self)) if issubclass(type(a), human_agent.Human)]
        exposed = [a for a in occupants if a.infective]

        num_occupants = len(occupants)
        num_exposed = len(exposed)

        # change accordingly (bus in 1 minute step size: 1/60)
        exposure_time = 5/60

        mean_breathing_rate = np.mean([trans_rate.return_breathing_flow_rate(a.breathing_rate) for a in occupants])
        mean_infectivity = np.mean([trans_rate.return_exhaled_air_inf(a.breathing_activity) for a in occupants])
        ACH = trans_rate.return_air_exchange_rate(self.environment)
        floor_area = self.floor_area
        
        
        # TODO: take mean of mask_prob of all human agents in the room
        mask_passage_prob = np.mean([a.mask_passage_prob for a in occupants])
        height = self.height


        transmission_rate = aerosol_new.return_aerosol_transmission_rate(floor_area=floor_area, room_height=height,
        air_exchange_rate=ACH, aerosol_filtration_eff=0, relative_humidity=0.69, breathing_flow_rate=mean_breathing_rate, exhaled_air_inf=mean_infectivity,
        mask_passage_prob=mask_passage_prob)

        transmission_rate *= exposure_time
        transmission_rate *= num_exposed #To be changed to some proportion to get infectious
        


        self.aerosol_transmission_rate.append(transmission_rate)
        

        

        
    def generate_desks(self, shape, radius_desk=5, spacing_desk=5):
        
        desks = []
        minx, miny, maxx, maxy = shape.bounds

        # Starting from left to right start assigning centers of desks
        x_start, y_start = minx + radius_desk, maxy - radius_desk
        
        # Small offsets to account for imperfect corners in digitization.
        x_start += 0.2
        y_start -= 0.2

        # Create the desks using the right buffers.
        desk = Point(x_start, y_start).buffer(radius_desk)
        net_space = radius_desk + spacing_desk
        
        x_coordinate = x_start
        y_coordinate = y_start
        
        # Loop to create the desks
        while True:
            desks.append(desk)        
            
            x_coordinate += net_space
            
            if (y_coordinate < miny) and (x_coordinate > maxx):         
                break
                        
            if x_coordinate > maxx:
                x_coordinate -= net_space
                x_coordinate = x_start 
                y_coordinate = y_coordinate - net_space
                
            
            desk = Point(x_coordinate, y_coordinate).buffer(radius_desk)    
        
        desks = gpd.GeoSeries(desks)

        # Figure out the desks which intersect the classroom and reject those. 
        # Note that we do not use contains operation. 

        desks  = desks[desks.apply(lambda desk: shape.intersects(desk))]
        
        return desks


    def generate_seats_circular(self, shape, desks, N, num_children_per_desk=5):

        # Helper function to help generate seat positions given a set of desks.
        def return_coords(desk, num_children_per_desk):
            boundary = list(desk.boundary.coords)
            step = len(boundary) // num_children_per_desk
            boundary = pd.Series(boundary[::step])
            boundary = boundary[:num_children_per_desk]
            return boundary.apply(Point).values.tolist() 

        # Return results using efficient vectorized apply functions.
        result = desks.apply(return_coords, args=(num_children_per_desk,))
        dataframe = pd.DataFrame((desks, result)).T
        dataframe = dataframe.rename({0:'desk', 1:'seats'}, axis=1)
        desk_series = dataframe.apply(lambda row: [row['desk'] for i in range(len(row['seats']))], axis=1)
        
        desk_series = desk_series.sum()
        result = result.sum()
        
        final_df = pd.DataFrame()
        final_df['desk'] = desk_series
        final_df['seat'] = result
        
        final_df['desk_id'] = final_df['desk'].apply(str)
        
        #Check those seats which are extremely close to the boundary and retain them. This is to compensate for digitization errors.
        to_drop = final_df[(gpd.GeoSeries(final_df['seat']).distance(shape) >= 0.1) & (~final_df['seat'].apply(lambda seat: shape.contains(seat)))].drop_duplicates(subset=['desk_id'])
        to_drop = to_drop.desk_id.values
        
        self.num_children_per_desk = num_children_per_desk

        # Only return N seating positions.
        return final_df[~final_df.desk_id.isin(to_drop)][:N]


    def generate_seats(self, N, width, style='individual', num_children_per_desk=None):
        
        self.seating_pattern = style
        self.seats = []
        shape = self.shape
        
        
        # generate grid seating that seperates each student by fixed amount
        if style == 'individual':         
            center = shape.centroid
            md = math.ceil(N**(1/2))
            pnt = Point(center.x - width*md//2, center.y - width*md//2)
            for i in range(md):
                for j in range(md+1):
                    self.seats.append(Point(pnt.x + i*width, pnt.y + j*width))
        
        
        # generate circular seating
        elif style == 'circular':
            
            # set default to number of children per desk
            if not num_children_per_desk:
                num_children_per_desk = 5
            
    
            # while loop to accomandate all children if seats genereated by default is not enough 
            # UPDATE 2/18: moved from school to classroom
            
            while len(self.seats) < N:
                dataframe_seats_desks = self.generate_seats_circular(shape, self.generate_desks(shape), N, num_children_per_desk=num_children_per_desk)
                self.seats = dataframe_seats_desks.seat.tolist()
                num_children_per_desk += 1
                
            
            self.desks = dict(dataframe_seats_desks[['desk_id', 'desk']].drop_duplicates(['desk_id']).set_index('desk_id')['desk'])
                                
    def generate_seats_lunch(self, xwidth, ywidth):
        
        self.seats = []
        xmin, ymin, xmax, ymax = self.shape.bounds
        xcoords = xmin + xwidth
        ycoords = ymin + ywidth
        
        y_pointer = ycoords
        x_pointer = xcoords
        
        while (xcoords < xmax):
            
            while (ycoords < ymax):
                self.seats.append(Point(xcoords, ycoords))
                ycoords += ywidth
                
            xcoords += xwidth
            ycoords = y_pointer
            
        np.random.shuffle(self.seats)