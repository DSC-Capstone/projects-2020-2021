from mesa_geo import GeoAgent, GeoSpace
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from mesa import datacollection
from mesa import Model

import room_agent
import human_agent
import cohort_agent
import util

from scipy import stats 
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

# Aerosol Transmission
import aerosol_new
import transmission_rate as trans_rate

# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# school config data


school_params_ini = 'schoolparams.ini'
parser_school = configparser.ConfigParser()
parser_school.read(config_file_path_prefix + school_params_ini)
population_config = parser_school['SCHOOL_POPULATION']
cohort_config = parser_school['COHORT']



class School(Model):
    
        
    schedule_types = {"Sequential": BaseScheduler,
                      "Random": RandomActivation,
                      "Simultaneous": SimultaneousActivation}

    
    def __init__(self, map_path, schedule_path, grade_N, KG_N, preschool_N, special_education_N, 
                 faculty_N, seat_dist, init_patient=3, attend_rate=1, mask_prob=0.516, 
                 inclass_lunch=False, student_vaccine_prob = 0, student_testing_freq = 14,
                 teacher_vaccine_prob = 1, teacher_testing_freq = 7, teacher_mask = 'N95', 
                 schedule_type="Simultaneous"):
        # zipcode etc, for access of more realistic population from KG perhaps
        
        
        
        # model param init
        self.__mask_prob = mask_prob
        self.inclass_lunch = inclass_lunch
        self.seat_dist = math.ceil(seat_dist/(attend_rate**(1/2)))
        self.idle_teachers = [] # teachers to be assigned without a classroom
        self.init_patient = init_patient
        
        
        # testing param init
        self.teacher_testing_freq = teacher_testing_freq
        self.student_testing_freq = student_testing_freq

        
        


        # mesa model init
        self.running = True
        self.grid = GeoSpace()
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        
        
        
        #data collect init
        model_reporters = {"day": "day_count",
                           "cov_positive": "infected_count"}
        agent_reporters = {"unique_id": "unique_id",
                           "health_status": "health_status",
                           "symptoms": "symptoms",
                           "x": "x",
                           "y": "y",
                           "viral_load": "viral_load"}
        self.datacollector = datacollection.DataCollector(model_reporters=model_reporters, agent_reporters=agent_reporters)
        
        

        
        
        
        school_gdf = gpd.read_file(map_path)
        # minx miny maxx maxy
        # use minx maxy
        # gdf.dessolve 
        # minx miny maxx maxy = geometery.bounds

        # for loop:
        #   bus = bus(shape(minx,maxy))
        #    minx = minx - width
        #    maxy = maxy + length

        
        
        # room agent init
        self.room_agents = school_gdf.apply(lambda x: room_agent.Classroom(
        unique_id = x["Id"], 
        model = self,
        shape = x["geometry"], 
        room_type = x["room_type"]),
                     axis=1
                    ).tolist()

        self.grid.add_agents(self.room_agents)
        
        
        
        # stats tracking init
        self.infected_count = 0
        self.step_count = 0
        self.day_count = 1
        self.num_exposed = 0
        
        
        
        # student activity init
        self.schoolday_schedule = pd.read_csv(schedule_path)
        self.activity = None
        
        
        # id tracking init
        self.__teacher_id = 0
        self.__student_id = 0
        self.__cohort_id = 0
        self.__faculty_N = faculty_N
        self.schedule_ids = self.schoolday_schedule.columns
        
        
        
        # geo-object tracking init
        self.recess_yards = util.find_room_type(self.room_agents, 'recess_yard')
        self.cohorts = []
        
        
        # UPDATE Christmas cohort generation
        def generate_cohorts(students, N):
            '''
            generate cohorts with within/out-of classroom probability, cohort size probablity
            example: students have 80% chance to have a friend in same room, 20% chance to have a friend in different room
            and 50% to have a cohort size of 5, 20% size 2, 15% size 4, 10% size 3, 5% size 1
            
            students: a 2d list containing list of students in each room
                students[k] is a list of student agents (in same classroom room)
                
            '''

            size_prob = eval(cohort_config['size_prob'])
            same_room_prob = eval(cohort_config['same_room_prob'])
            
            
            
            radius = eval(cohort_config['radius'])
            
            
            
            size_val_list = list(size_prob.keys())
            size_prob_list = list(size_prob.values())
            same_room_val_list = list(same_room_prob.keys())
            same_room_prob_list = list(same_room_prob.values())
            
            
            # start at room 0
            cur_room = 0

            while N > 0:

                cur_size = np.random.choice(size_val_list, p = size_prob_list)
                if N <= max(size_val_list):
                    cur_size = N

                # get same-room cohort size 
                cur_same = sum(np.random.choice(same_room_val_list, size = cur_size, p = same_room_prob_list))

                # add students from current room to current cohort
                cur_same = min(cur_same, len(students[cur_room]))          
                cohort = students[cur_room][:cur_same]
                students[cur_room] = students[cur_room][cur_same:]


                room_idx = list(range(len(students)))

                other_room = room_idx[:]
                other_room.remove(cur_room)

                # add students from other rooms to cohort
                if not len(other_room):
                    rand_room = [cur_room]*(cur_size - cur_same)
                else:
                    rand_room = np.random.choice(other_room ,size = (cur_size - cur_same))
                    
                for r in rand_room:
                    # update and remove r if r is an empty room
                    while True:
                        try:
                            cohort.append(students[r][0])
                            students[r] =students[r][1:]
                            break
                        except:
                            if r in other_room:
                                other_room.remove(r)
                            if not len(other_room):
                                r = cur_room
                            else:
                                r = np.random.choice(other_room)


                            
                            
                            
                # TODO: recess yard is current hard coded             
                recess_yard = self.recess_yards[0]
                if cohort[0].grade != 'grade':
                    recess_yard = self.recess_yards[1]
                    
                    
                # make cohort agent with dummy shape
                cur_cohort = cohort_agent.Cohort("Cohort" + str(self.__cohort_id), self, Point(0,0), cohort, recess_yard, cur_size*radius)
                self.grid.add_agents(cur_cohort)
                self.schedule.add(cur_cohort)
                self.cohorts.append(cur_cohort)
                self.__cohort_id += 1
                
                
                # remove empty rooms
                students = [room for room in students if len(room) > 0]

                # rolling update to minimize student pop edge cases 
                # fail safe break
                if not len(students):
                    break
                cur_room = (cur_room + 1)%len(students)

                # update student population
                N -= cur_size
            
            
            


        def init_agents(room_type, N, partition=False):
            '''
            batch initialize human agents into input room type rooms with equal partition size
            
            room_type: a valid string of room type: [None, 'restroom_grade_boys', 'lunch_room', 'classroom_grade',
               'restroom_all', 'restroom_grade_girls', 'restroom_KG',
               'classroom_KG', 'community_room', 'library',
               'restroom_special_education', 'restroom_faculty',
               'classroom_special_education', 'health_room', 'faculty_lounge',
               'classroom_preschool', 'restroom_preschool']
            '''

                
            rooms = util.find_room_type(self.room_agents, room_type)
            
            
            # if student group should be seperated to different day schedules
            # assigning schedule_id to equally partitioned rooms
            # currently only grade 1-5 "grade" students need to be partitioned, 
            partition_size = len(rooms)
            if partition:
                partition_size = math.ceil(partition_size/len(self.schedule_ids))
                
            class_size = N//len(rooms)
            remaining_size = N%len(rooms)
            
            #track all students of same grade type
            all_students = []
            for i, classroom in zip(range(len(rooms)), rooms):
                
                
                
                                
                # spread remaining student into all classrooms
                c_size = class_size
                if remaining_size > 0:
                    remaining_size -= 1
                    c_size += 1
                    
                    
                
                #each classroom has its own possibility to have circular desks instead of normal grid seating
                #TODO: strongly believe this is subject to change
                prob_circular = eval(population_config['circular_desk_prob'])
                
                if np.random.choice([True, False], p=[prob_circular, 1-prob_circular]):
                    classroom.generate_seats(c_size, self.seat_dist, style='circular')
                else:
                    classroom.generate_seats(c_size, self.seat_dist)
                
                
                classroom.schedule_id = self.schedule_ids[i//partition_size]

                
            

                #track students within the same room
                students = []
                for idx in range(c_size): 
                    pnt = classroom.seats[idx]
                    mask_on = np.random.choice([True, False], p=[mask_prob, 1-mask_prob])
                    agent_point = human_agent.Student(model=self, shape=pnt, unique_id="S"+str(self.__student_id), room=classroom, mask_on=mask_on)
                    # vaccinate students accordingly
                    agent_point.vaccinated = np.random.choice([True, False], p = [student_vaccine_prob, 1- student_vaccine_prob])
                    
                    
                    if classroom.seating_pattern == 'circular':
                        desks = gpd.GeoSeries(classroom.desks)
                        agent_point.desk = desks[desks.distance(agent_point.shape).sort_values().index[0]]
                        
                        
                    self.grid.add_agents(agent_point)
                    self.schedule.add(agent_point)
                    self.__student_id += 1
                    
                    # add student to room temp list
                    students.append(agent_point)

                    
                
                #add teacher to class
                pnt = util.generate_random(classroom.shape)
                agent_point = human_agent.Teacher(model=self, shape=pnt, unique_id="T"+str(self.__teacher_id), room=classroom)
                
                # teacher mask/vaccination protocol 
                agent_point.vaccinated = np.random.choice([True, False], p = [teacher_vaccine_prob, 1- teacher_vaccine_prob])
                agent_point.mask_type = teacher_mask
                agent_point.mask_passage_prob = trans_rate.return_mask_passage_prob(teacher_mask)
                
                
                self.grid.add_agents(agent_point)
                self.schedule.add(agent_point)
                self.__teacher_id += 1
                self.__faculty_N -= 1
                
                # add room students list to all students 
                # shuffle students for efficiency improvement
                np.random.shuffle(students)
                all_students.append(students)
            
            
            
            
            #UPDATE Christmas
            #generate cohort with temp student list
            generate_cohorts(all_students, N)
            
            


        
        # initialize all students and teachers in classrooms
        init_agents("classroom_grade", int(grade_N*attend_rate), partition=True)        
        # keep track of student types
        #self.grade_students = [a for a in list(self.schedule.agents) if isinstance(a, Student)]        
        init_agents("classroom_KG", int(KG_N*attend_rate))
        init_agents("classroom_preschool", int(preschool_N*attend_rate))        
        #self.pkg_students = [a for a in list(set(self.schedule.agents).difference(self.grade_students)) if isinstance(a, Student)]
        init_agents("classroom_special_education", int(special_education_N*attend_rate))
  
            

        # dump remaining teacher to faculty lounge
        for f_lounge in util.find_room_type(self.room_agents, "faculty_lounge"):
            f_lounge.schedule_id = self.schedule_ids[0]
            
            for i in range(self.__faculty_N):

                pnt = util.generate_random(f_lounge.shape)
                agent_point = human_agent.Teacher(model=self, shape=pnt, unique_id="T" + str(self.__teacher_id), room=f_lounge)
                
                # teacher mask/vaccination protocol 
                agent_point.vaccinated = np.random.choice([True, False], p = [teacher_vaccine_prob, 1- teacher_vaccine_prob])
                agent_point.mask_type = teacher_mask
                agent_point.mask_passage_prob = trans_rate.return_mask_passage_prob(teacher_mask)
                
                self.grid.add_agents(agent_point)
                self.schedule.add(agent_point)
                
                #teacher from faculty lounge can be used later if on duty teachers test positive
                self.idle_teachers.append(agent_point) 
                
                self.__teacher_id += 1

        
        # add rooms to scheduler at last 
        for room in self.room_agents:
            self.schedule.add(room)
            
            
            
         
            
            
        self.lunchroom = util.find_room_type(self.room_agents, 'lunch_room')[0]
        self.lunchroom.generate_seats_lunch(1, 4)
            
    
    
    
    
    
    def small_step(self):
        self.schedule.step()
        self.grid._recreate_rtree() 



    def add_N_patient(self, N): 
        patients = random.sample([a for a in self.schedule.agents if isinstance(a, human_agent.Student)], N)
        for p in patients:
            p.health_status = "exposed"
            p.asymptomatic = True
            p.infective = True


    def show(self):
        '''
        plot current step visualization
        deprecated since end of model visualization update
        '''

        # UPDATE 10/16: add deprecation warning
        message  = "this function is no longer used for performance issues, check output_image.py for end of model visualization"
        warnings.warn(message, DeprecationWarning)


        school_geometry = gpd.GeoSeries([a.shape for a in self.room_agents])
        school_map = gpd.GeoDataFrame({"viral_load" : [min(a.viral_load, 5) for a in self.room_agents]})
        school_map.geometry = school_geometry
        basemap = school_map.plot(column = "viral_load", cmap="Reds", alpha = 0.5, vmin = 0, vmax = 5)
        school_map.boundary.plot(ax = basemap, color='k', linewidth=0.2)

        list(map(lambda a: a.plot(), [a for a in self.schedule.agents if issubclass(type(a), human_agent.Human)]))

        hour = 9 + self.step_count*5//60 # assume plot start at 9am
        minute = self.step_count*5%60
        plt.title("Iteration: Day {}, ".format(self.day_count) + "%d:%02d" % (hour, minute), fontsize=30)



    def __update_day(self):
        '''
        update incubation time, reset viral_load, remove symptomatic agents, aerosol transmission etc for end of day
        '''
        for a in self.schedule.agents[:]:
            # update human agent disease stats
            if issubclass(type(a), human_agent.Human):

                if a.symptoms:
                    # remove agent if symptom onset
                    if isinstance(a, human_agent.Teacher) and len(self.idle_teachers) > 0:
                        # assign a new teacher to position
                        new_teacher = self.idle_teachers.pop()
                        new_teacher.shape = a.shape
                        new_teacher.room = a.room
                        new_teacher.classroom = a.classroom
                    self.schedule.remove(a)
                    self.grid.remove_agent(a)

                # UPDATE 10/16: infectious made obsolete, end of day update rework
                elif a.health_status == "exposed":
                    
                    # UPDATE 2/28: merge testing implementation 
                    # test student and teacher accordingly
                    # Q: why testing is here under exposed case?: 
                    # testing only matters if infect the result is a hit
                    # therefore the agnent gets removed only if two conditions are met
                    # 1.testing is arranged; 2. testing result the agent is indeed exposed 
                    if isinstance(a, human_agent.Teacher) and (self.day_count % self.teacher_testing_freq == 0):
                        # if hit teacher, try to assign a new teacher to position
                        if len(self.idle_teachers) > 0:
                            new_teacher = self.idle_teachers.pop()
                            new_teacher.shape = a.shape
                            new_teacher.room = a.room
                            new_teacher.classroom = a.classroom
                        #remove teacher if testing conditions are met
                        self.schedule.remove(a)
                        self.grid.remove_agent(a)   

                    elif isinstance(a, human_agent.Student) and (self.day_count % self.student_testing_freq == 0):
                        #remove student if testing conditions are met
                        self.schedule.remove(a)
                        self.grid.remove_agent(a)   
                    
                    # UPDATE 10/17: update infective delay if agent is not infective by end of day
                    a.infective = True
                    a.symptom_countdown -= 1
                    # calculate when symptoms begin to show using 0-15 density
                    if a.symptom_countdown <= 0:
                        if a.symptom_countdown == 0: 
                            self.infected_count += 1
                        # update model stat for total infected
                        # negative countdown means this agent is asymptomatic

                        if not a.asymptomatic:
                            # this is a really small chance, however possible
                            # set symtoms to true
                            # next day this agent will be removed from the model
                            a.symptoms = True
                            
                            
                            
                

            # update room agent aerosal stats 
            elif issubclass(type(a), room_agent.Classroom):
                room = a
                mean_aerosol_transmissions = sum(room.aerosol_transmission_rate)
                if np.isnan(mean_aerosol_transmissions):
                    mean_aerosol_transmissions = 0

                occupants = [a for a in list(self.grid.get_intersecting_agents(room)) if issubclass(type(a), human_agent.Human)]
                healthy_occupants = [a for a in occupants if a.health_status == 'healthy']                  
                    
                # failsafe for rare case where this can exceed one
                mean_aerosol_transmissions = min(mean_aerosol_transmissions, 1)
                
                # treating aerosal transmissions as a probability for each healthy occupant in this room to get sick
                
                for healthy_occupant in healthy_occupants:
                    if np.random.choice([True, False], p =[mean_aerosol_transmissions, 1-mean_aerosol_transmissions]):
                        if not healthy_occupant.vaccinated:
                            healthy_occupant.health_status = 'exposed'
                        
                    
    def step(self):
        '''
        simulate a day with school day schedule
        '''
        if not self.schedule.steps:
            self.add_N_patient(self.init_patient)



        for i, row in self.schoolday_schedule.iterrows():
            self.activity = row
            self.datacollector.collect(self)
            self.schedule.step()
            self.grid._recreate_rtree() 
            self.step_count += 1


        self.__update_day()  
        self.grid._recreate_rtree() 
        self.day_count += 1
        self.step_count = 0

            
            
