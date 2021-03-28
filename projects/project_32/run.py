import sys
import json
import shutil

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from etl import get_data
from scale_model import init_positions, droplet_infect, return_aerosol_transmission_rate, get_distance, get_dist_multiplier, one_room, scatter_collect

# Targets:
DATA_PARAMS = 'config/default_config.json'
SCALE_MODEL_PARAMS = 'config/scale_room.json'
VIZ_PARAMS = 'config/viz.json' ## make this file

def load_parameters(filepath):
    '''
    Loads input and output directories
    '''
    with open(filepath) as fp:
        parameter = json.load(fp)

    return parameter


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    '''

    if 'test' in targets: # Visualize the time until infectiveness of a newly infected individual
        # plot temporal distributions and chu effect given baseline of .2, .1, .05
        temp = load_parameters(SCALE_MODEL_PARAMS)


        # one_room(temp['input_dir'], temp['output_dir'], False) #implement additional input variable in one_room()
        return

    if 'bus_flow' in targets:
        # output quiver plot and ??contour plot?? for bus of input type
        temp = load_parameters(BUS_PARAMS)
        # one_room(temp['input_dir'], temp['output_dir'], False, )

        return

    if 'scatter' in targets:
        temp = load_parameters(SCALE_MODEL_PARAMS)
        scatter_collect(temp['input_dir'], temp['output_dir'], False)
        return

    if 'clear' in targets:
        # remove files from scatter_folder
        return

    if 'visualize' in targets:
        print('targets: ')
        print(targets)
        # one room viz using website / airavata inputs
        temp = load_parameters(SCALE_MODEL_PARAMS)
        temp_viz = load_parameters(VIZ_PARAMS)
        one_room(temp['input_dir'], temp['output_dir'], True) #implement viz params and viz code
        # visualize_infection() takes in timesteps of infection and returns a plotted visualization w/ red = inf, green = not inf

        # plot_infection() demonstrates the 'animations' with an XY plot of number individuals infected (X) vs timesteps of 5 mins (Y)


        # TODO: get help from Kaushik w/ Airavata inputs

        # TODO: get help from team for website integration
        return

        # set up variables

    return


if __name__ == '__main__':
    # run via:
    # python run.py analysis
    targets = sys.argv[1:]
    main(targets)
