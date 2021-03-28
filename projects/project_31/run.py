from mesa_geo import GeoAgent, GeoSpace
from mesa.time import BaseScheduler
from mesa.time import SimultaneousActivation
from mesa import datacollection
from mesa.datacollection import DataCollector
from mesa import Model
import pandas as pd
import numpy as np
import random
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sys
import shapely
from shapely.geometry import Polygon, Point, LineString

import os.path
import json

from src.data.build_datasets import DatasetMaker
from src.models.run_model import RunAll
from src.visualization.viz import Visualize
from src.visualization.viz_gif import GifMaker

# silencing all warnings
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('> using given parameters')
        # extracting parameters from default config file
        params = DatasetMaker().make_data('bus_params')

    else:
        try:
            print('> using test parameters')
            # extracting parameters from given test file
            params = DatasetMaker().make_data('test')

        except:
            print('invalid data')

    # initiating the model, running the model, and recording agent data for each step
    agent_data, results = RunAll().run_program(params)

    print('> Working on Visualization \n .')
    # making the results graph showing number of healthy and sick passangers in each step
    Visualize().make_viz(results,params)
    print(" .")
    #making the gif of the bus showing the location of healhty and sick people in each step
    GifMaker().make_gif(agent_data,params)
    print(" .")
    print('> Visualization Done')
    print('> results saved in figures folder')