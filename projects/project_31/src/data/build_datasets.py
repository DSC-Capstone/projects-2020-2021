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
import os.path
import json

class DatasetMaker:
    # dummy initialization
    def __init__(self):
        a=0
        
    def make_data(self,inputt):
        # in case the input is test, use test parameters in test.json file
        if inputt == 'test':
            filepath = os.getcwd()+'/config/' + 'test' + '.json'
            with open(filepath) as f:
                params = json.load(f)
        # otherwise use default parameters in bus_params.json file
        else:
            filepath = os.getcwd()+'/config/' + 'bus_params' + '.json'
            with open(filepath) as f:
                params = json.load(f)
        # return parameters back to run.py
        return params