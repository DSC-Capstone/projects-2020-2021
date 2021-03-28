## TODO
# Make this file just functions
# make plots better:
# run 10k make raster is in 'air_flow'
#     This saves the grid overlay to a file
# ventilation calculation is in 'vent_setup'
#     This plots a line between the edges of the vent and the edges of the windows: directionally influences aerosol
# Implement masks as a function of this ventilation / time: i.e. reduced airflow


import math
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import json

# TODO:
# add viz for 1-room infection
# plot infections / time

def init_positions(floor_area, n_students):
    # generate size of room based on circular/square desks
    positions = {}
    # grid desks
    rows = int(math.sqrt(n_students))
    count = 0
    for i in range(rows):
        for j in range(rows):
            positions[count] = [i, j]
            count += 1
    inf_index = random.sample(set(range(n_students)), 2)
    uninf_index = list(range(n_students))
    inf_array = []


    # double check this
    for i in inf_index:
        uninf_index.remove(i)
        inf_array.append(i)
    return uninf_index, inf_index, inf_array, positions




def return_aerosol_transmission_rate(floor_area, mean_ceiling_height,
                            air_exchange_rate,
                            aerosol_filtration_eff, relative_humidity, breathing_flow_rate,
                            exhaled_air_inf, max_viral_deact_rate, mask_passage_prob,
                            max_aerosol_radius=2, primary_outdoor_air_fraction=0.2):
    # Function to return transmission % from smaller droplets dispersing around the room
    # Same as SchoolABM

    mean_ceiling_height_m = mean_ceiling_height * 0.3048 #m3
    room_vol = floor_area * mean_ceiling_height  # ft3
    room_vol_m = 0.0283168 * room_vol  # m3

    fresh_rate = room_vol * air_exchange_rate / 60  # ft3/min
    recirc_rate = fresh_rate * (1/primary_outdoor_air_fraction - 1)  # ft3/min
    air_filt_rate = aerosol_filtration_eff * recirc_rate * 60 / room_vol  # /hr
    eff_aerosol_radius = ((0.4 / (1 - relative_humidity)) ** (1 / 3)) * max_aerosol_radius
    viral_deact_rate = max_viral_deact_rate * relative_humidity
    sett_speed = 3 * (eff_aerosol_radius / 5) ** 2  # mm/s
    sett_speed = sett_speed * 60 * 60 / 1000  # m/hr
    conc_relax_rate = air_exchange_rate + air_filt_rate + viral_deact_rate + sett_speed / mean_ceiling_height_m  # /hr
    airb_trans_rate = ((breathing_flow_rate * mask_passage_prob) ** 2) * exhaled_air_inf / (room_vol_m * conc_relax_rate)

    return airb_trans_rate #This is mean number of transmissions per hour between pair of infected / healthy individuals


def get_distance(infect_id, uninfect_id, student_pos):
    x1y1 = student_pos[infect_id]
    x2y2 = student_pos[uninfect_id]
    # print(student_pos)
    return math.sqrt(((x2y2[0]-x1y1[0])**2) + ((x2y2[1] - x1y1[1])**2))

def directional_air(matrix, row, col, direction='back'): # back of the bus = down
    # dir_matrix is a np.array
    # risk_matrix is a pd.dataframe


    # is 1 good for ACH? use as proxy for now
    # proxy for now is 1 for both aerosol emission and air flow (on bus)
    # TODO: classroom
    # TODO: this can probably be EASILY optimized numerically
    # ETA: 3/14

    # 2 steps:
    # 1. make +/- matrix:
    # 2. loop through and combine original and +/-
    # plus_matrix

    # fix this function now

    if direction == 'back':
        matrix[row][col] - 1
        matrix[row-1][col] + 1
        return matrix

    if direction == 'forward':
        #
        matrix[row][col] - 1
        matrix[row+1][col] + 1
        return matrix

    if direction == 'left':
        #
        matrix[row][col] - 1
        matrix[row][col-1] + 1
        return matrix

    if direction == 'right':
        matrix[row][col] - 1
        matrix[row][col+1] + 1
        return matrix

    if direction == 'left-back':
        #
        matrix[row][col] - 1
        matrix[row-1][col-1] + 1
        return matrix

    if direction == 'left-forward':
        matrix[row][col] - 1
        matrix[row+1][col-1] + 1
        return matrix

    if direction == 'right-back':
        matrix[row][col] - 1
        matrix[row-1][col+1] + 1
        return matrix

    if direction == 'right-forward':
        matrix[row][col] - 1
        matrix[row+1][col+1] + 1
        return matrix
    return



def matrix_avg(input_matrices):
    '''
    calculate mean for every mutual [i,j] in input
    list of np.arrays
    '''
    return



def create_color_viz(infect_times, num_students = 25):
    # implement having more students in the class

    # initialize positions of students
    positions = {}
    x_arr = []
    y_arr = []
    rows = int(math.sqrt(num_students))
    count = 0
    for i in range(rows):
        for j in range(rows):
            positions[count] = [i, j]
            x_arr.append(i)
            y_arr.append(j)
            count += 1

    inf_x_arr = []
    inf_y_arr = []
    for inf in infect_times.keys():
        # print(positions[inf])
        inf_x_arr.append(positions[inf][0])
        inf_y_arr.append(positions[inf][1])

        # color dot red

    # plot uninfected in blue
    plt.scatter(x_arr, y_arr, color='blue')
    # plot infected in red
    # only output last frame for now:
    plt.scatter(inf_x_arr, inf_y_arr, color='red')
    plt.show()

    # for i in range(len(inf_x_arr)):
    #     # scatter
    #     print(inf_x_arr[i])

    # TODO: loop through timesteps and/or skip to next timestep and replot
    # i.e. step 0: 2 infected, step 27: 3 infected, etc etc


    return

def create_plot_viz(infect_times, infect_distrib):
    # convert infection timestamps to step function plot
    times = infect_times.values()
    #
    # print('times', times)
    step_x_array = times
    step_y_array = [i for i in range(len(times))]
    plt.step(step_x_array, step_y_array) # label timestep above each infection
    plt.show()

    # plot distribution of infection rate over course of model run
    plt.hist(infect_distrib, bins=50)
    plt.gca().set(title='Transmission likelihood distribution')
    plt.show()

    return

# Go by standards mandated by school district in terms of masks and setup
def one_room(input_dir, output_dir, viz_checkd, num_people=25, mask_type='cloth', num_days=5, num_class=3, vent_test=False):


    uninf_index, inf_index, inf_array, student_pos = init_positions(900, num_people)

    # setup variables and functions# output graphs
    # countdown until symptoms appear: probability curve
    shape, loc, scale =  (0.6432659248014824, -0.07787673726582335, 4.2489459496009125)
    x = np.linspace(0, 17, 1000)
    countdown_curve = stats.lognorm(s=shape, loc=loc, scale=scale)

    infected = {i: int(np.round(stats.lognorm.rvs(shape, loc, scale, size=1)[0], 0)) for i in inf_index}

    # bound the days for overflow errors
    for i in infected:
        if infected[i] > 17:
            infected[i] = 17
        if infected[i] < -10:
            infected[i] = -10
    # create infectiveness reference dataframe
    shape, loc, scale = (20.16693271833812, -12.132674385322815, 0.6322296057082886)
    x = np.linspace(-10, 8, 19)
    infective_df = pd.DataFrame({'x': list(x), 'gamma': list(stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))})

    uninfected = {i: 0 for i in uninf_index}

    ##### TODO: Get these inputs from default only if not available in user input
    deff = open('src/data/default.json',)
    floor_plan = json.load(deff)
    # for i in floor_plan:
    #     print(i)
    # deff.close()
    floor_area = floor_plan['floor_area']  # ft2
    mean_room_height = floor_plan['mean_ceiling_height']  # ft
    air_exchange_rate = floor_plan['air_exchange_rate']  # /hr (air changes per hour (ACH))

    ##Assumed Parameter Identified as 0.2 for Classrooms, 1.0 for outdoors##
    primary_outdoor_air_fraction = floor_plan['primary_outdoor_air_fraction']  # 1.0 = natural ventilation
    aerosol_filtration_eff = floor_plan['aerosol_filtration_eff']  # >0.9997 HEPA, =0.2-0.9 MERVs, =0 no filter

    #Average daily RH for San Diego is 69%
    relative_humidity = floor_plan['relative_humidity']
    physical_params = [floor_area, mean_room_height, air_exchange_rate, primary_outdoor_air_fraction,
                            aerosol_filtration_eff, relative_humidity]

    # Physiological Parameters
    breathing_flow_rate = floor_plan['breathing_flow_rate']  # m3/hr
    max_aerosol_radius = floor_plan['max_aerosol_radius']  # micrometers
    physio_params = [breathing_flow_rate, max_aerosol_radius]

    # Disease Parameters
    exhaled_air_inf = floor_plan['exhaled_air_inf']  # infection quanta/m3, changes with acitivity type.
    max_viral_deact_rate = floor_plan['max_viral_deact_rate']  # /hr
    disease_params = [exhaled_air_inf, max_viral_deact_rate]

    # Precautionary Parameters
    mask_passage_prob = floor_plan['mask_passage_prob'] # 1 = no masks, ~0.1 cloth, <0.05 N95
    risk_tolerance = floor_plan['risk_tolerance']  # expected transmissions per infector
    prec_params = [mask_passage_prob, risk_tolerance]


    # Use Chu
    chu_distance_curve = 1/2.02
    # loop through day


    days = num_days
    classes = num_class
    steps = 12 # 5 mins / step = 1 hour / class

    # use these to generate plot of num_infected by step count, assuming all else equal
    step_count = 0
    infect_plot_dict = {}

    # temp
    min = 1
    max = 0
    trans_array = []
    infection_timesteps = {}
    for i in inf_index:
        infection_timesteps[i] = 0
    # print(infection_timesteps)

    # generate plotline of num_infected using minimum default input: no masks, bad airflow, etc

    # generate plotline of num_infected using maximum default input: n-95 masks, barriers, vents, etc

    num_infected = 2

    # Aerosol transmission in this room (assumes well-mixed room)
    air_transmission = return_aerosol_transmission_rate(floor_area, mean_room_height,
                            air_exchange_rate,
                            aerosol_filtration_eff, relative_humidity, breathing_flow_rate,
                            exhaled_air_inf, max_viral_deact_rate, mask_passage_prob=.1,
                            max_aerosol_radius=2, primary_outdoor_air_fraction=0.2)
    # print('air', air_transmission)

    # Loop through days in a week (5 days)
    for i in range(days):

        # Loop through classes in each day (3 classes)
        for c in range(classes):

            # Loop through 5 minute steps in each class (total of 12 steps / class)
            for s in range(steps):
                step_count += 1

                ## Droplet Infection
                # Loop through infected students
                for i_student in inf_index:
                    # Loop through uninfected and calculate likelihood of infection
                    for u_student in uninf_index:
                        try:
                            transmission = droplet_infect(i_student, u_student, infected, student_pos, infective_df, chu_distance_curve).iloc[0]
                            trans_array.append(transmission)
                        except:
                            transmission = droplet_infect(i_student, u_student, infected, student_pos, infective_df, chu_distance_curve)
                            # print(transmission)
                            # break
                            transmission = 0 # Out of bounds of infectivity

                        # print(transmission)
                        if np.random.choice([True, False], p=[transmission, 1-transmission]):


                            # add u to infected
                            inf_array.append(u_student)

                            infection_timesteps[u_student] = step_count
                            # also calculate time until symptoms for u_student

                            uninf_index.remove(u_student)


            # Aerosol Infection
            #Loop through infected and uninfected for pairwise calculations

            # this is calculated after every class (hourly transmission rate in that space)
            for i in inf_index:

                for u in uninf_index:
                    a_transmit = air_transmission
                    if np.random.choice([True, False], p=[a_transmit, 1-a_transmit]):
                        # add u to infected
                        inf_array.append(u)
                        # also calculate time until symptoms for u_student

                        uninf_index.remove(u)
                    pass

    out = 2 # count number students in inf_index
    print('this is the number of infected students: ' + str(len(inf_array)))
    print('these are the infected student IDs: ' + str(inf_array))
    print('this is when they were infected: ' + str(infection_timesteps))
    # print(np.min(trans_array), np.max(trans_array), 'min, max')
    # print(np.mean(trans_array), 'mean')

    # '''
    if viz_checkd:

        create_color_viz(infection_timesteps, 25)
        print(infection_timesteps, 'time')

        create_plot_viz(infection_timesteps, trans_array)
    # '''

    # also return list of IDs infected
    return inf_array

def scatter_collect(input_dir, output_dir, viz_checkd, num_people=25, mask_type='cloth',
 num_days=5, num_class=3, vent_test=False, n_runs=10): # also add one_room inputs, use this as run.py command
    '''
    run one_room n=1000 times and take avg of # times each id is stored

    divide by # runs = % chance of each seat being infected in each seat:

    Y = # times infectious/infective/infected by the model
    X = avg distance from other points

    i.e. Y =
    a dictionary with {ID: % of time infected}
    X =
    a dictionary with {ID: Avg distance from other individuals}

    ### TODO: think of better X variable
    '''
    id_counts = {i: 0 for i in range(num_people)}
    for run in range(n_runs):
        infected = one_room("src/data/default", "src/data", viz_checkd=False, num_people=25, mask_type='cloth', num_days=5, num_class=3, vent_test=False)
        for id in infected:
            id_counts[id] += 1
    x = id_counts.values()
    y = id_counts.values()
    print(len(y))
    plt.scatter(x, y)
    plt.show()


    # temp = one_room
    # get one_room to output which IDs are infected and their distance from other individuals

    return

def get_dist_multiplier(distance, chu_distance_curve):
# TODO: update this function with correct Chu Calculation from SchoolABM
    return distance * chu_distance_curve

def droplet_infect(infect_id, uninfect_id, infected, student_pos, infective_df, chu_distance_curve, mask_passage_prob=.1):
    # Function to return transmission % from larger droplets
    # TODO: update breathing rate calculation w/ SchoolABM version

    distance = get_distance(infect_id, uninfect_id, student_pos)
    time = infected[infect_id]
    try:
        transmission_baseline = infective_df[infective_df.x == -1 * time]['gamma']
    except:
        print('out of range')
        return 0

    # get distance from chu
    distance_multiplier = get_dist_multiplier(distance, chu_distance_curve)
    # approximate student time spent breathing vs talking vs loudly talking
    breathing_type_multiplier = np.random.choice([.1, .5, 1], p=[.2, .05, .75])#############################
    # whisper, loud, heavy

    mask_multiplier = mask_passage_prob # equivalent to aerosol masks

    # convert transmission rate / hour into transmission rate / step
    hour_to_fivemin_step = 5/60
    # test if necessary

    return transmission_baseline * distance_multiplier * breathing_type_multiplier * mask_multiplier * hour_to_fivemin_step
