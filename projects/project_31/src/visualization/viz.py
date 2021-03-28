
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
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os.path
import json


class Visualize:

    def __init__(self):
        a = 2

    # get the data and make the figure for sick and healthy students
    def make_viz(self,result,params):
        filename = 'vis_params.json'
    	# reading default data folder from config
        filepath = os.getcwd()+'/config/'+filename
        with open(filepath) as f:
            file = json.load(f)
        path = file['path']
        student_num = params['student_num']
        result['step'] = result.index
        df = result[['step','sick?','present']].groupby(['step']).sum().reset_index()
        df.index = df.step
        df = df[['step', 'sick?','present']]
        df['healthy'] = df['present'] - df['sick?']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.step, y=df['healthy'],mode='lines',name='healthy'))
        fig.add_trace(go.Scatter(x=df.step, y=df['sick?'],mode='lines',name='sick'))
        fig.add_trace(go.Scatter(x=df.step, y=df['healthy']+df['sick?'],name='total',line = dict(color='black', width=1, dash='dot')))
        fig.update_layout(title="Passangers conditions through bus ride", xaxis_title="Minutes from the start of the ride", yaxis_title="# of passangers")
        fig.write_image(path+"results.png")
