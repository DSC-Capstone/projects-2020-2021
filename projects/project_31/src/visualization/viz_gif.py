# Importing required libraries
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import matplotlib as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os.path
import json

from mesa_geo import GeoAgent, GeoSpace
from mesa.time import BaseScheduler
from mesa.time import SimultaneousActivation
from mesa import datacollection
from mesa.datacollection import DataCollector
from mesa import Model
from shapely.geometry import Polygon, Point, LineString
from PIL import Image, ImageDraw, ImageFont


class GifMaker:

    def __init__(self):
        a = 0

    def make_gif(self,agent_data,params):
        filename = 'vis_params.json'
    	# reading default data folder from config
        filepath = os.getcwd()+'/config/'+filename
        with open(filepath) as f:
            file = json.load(f)
        path = file['path']
        seat_width = params['seat_width']
        seat_length = params['seat_length']
        seating_array = params['seating_array']
        bus_cols = int(params['bus_cols'])
        bus_rows = int(params['bus_rows'])
        bus_stop_student_count = params['bus_stop_student_count']
        width = (bus_cols + len(params['seating_array']) -1 + 2) * seat_width
        length = (bus_rows + 2) * seat_length
        ww = params['window_width']
        images = []
        steps = int(params['steps'])

        student_num = int(params['student_num'])
        sick_num = int(params['sick_num'])
        center = width // 2
        gray = (150, 150, 150)
        white = (250, 250, 250)
        red = (250,0,0)
        darkred = (100,0,0)
        blue = (50,50,200)
        green = (0,200,0)
        yellow = (200,200,0)
        # make the initial bus
        im = Image.new('RGB', (width, length), gray)
        draw = ImageDraw.Draw(im)

        #make the empty bus
        for r in range(bus_rows):
            # bus windows
            x1 = seat_width * 0.8
            x2 = width - x1
            y1 = seat_length*0.7 + r*seat_length
            y2 = seat_length*1.5 + r*seat_length
            draw.rectangle([(x1, y1), (x1+ww, y2)], fill=blue)
            draw.rectangle([(x2, y1), (x2-ww, y2)], fill=blue)
            # A/C panels
            y1 = seat_length*0.7 + r*seat_length
            y2 = seat_length*1.5 + r*seat_length
            draw.rectangle([(x1+ww+1, y2-ww), (x1+2*ww+1, y2-2*ww)], fill=yellow)
            draw.rectangle([(x2-ww, y2-ww), (x2-2*ww, y2-2*ww)], fill=yellow)

            # make empty seats
            for c in range(bus_cols):
                if c >= seating_array[0]:
                    c2 = c + 1
                else:
                    c2 = c
                x1 = seat_width + c2*seat_width
                x2 = seat_width*1.9 + c2*seat_width
                y1 = seat_length + r*seat_length
                y2 = seat_length*1.5 + r*seat_length
                draw.rectangle([(x1, y1), (x2, y2)], fill=white)
        imbase = im.copy()
        # add the initial empty bus for a longer period of time
        for i in range(5):
            images.append(imbase)
        
        
        draw.text((10,10), "Minute: 0", fill = (100,100,100))

        # make a photo of each step and appand it
        for i in range(1+steps):
            im2 = imbase.copy()
            draw = ImageDraw.Draw(im2)
            draw.text((10,10), "Minute: " + str(i), fill = (100,100,100))
            for c in range(bus_cols):
                for r in range(bus_rows):
                    cell = agent_data[agent_data.Step == i][agent_data.x == c][agent_data.y == r]
                    
                    cell_color = (0,0,0)
                    if len(cell) == 0 or cell.iloc[0,4] == False:
                        continue

                    # healthy student draw green
                    elif cell.iloc[0,5] == False:
                        cell_color = green
                    # original sick draw dark red
                    elif cell.iloc[0,5] == True and cell.iloc[0,6]== True:
                        cell_color = darkred
                    # new sick draw red
                    elif cell.iloc[0,5] == True:
                        cell_color = red

                    c = cell.iloc[0,2]
                    r = cell.iloc[0,3]
                    if c >= seating_array[0]:
                        c2 = c + 1
                    else:
                        c2 = c
                    x1 = seat_width + c2*seat_width
                    x2 = seat_width*1.9 + c2*seat_width
                    y1 = seat_length + r*seat_length
                    y2 = seat_length*1.5 + r*seat_length
                    draw.rectangle([(x1, y1), (x2, y2)], fill=cell_color)



            images.append(im2)


        images[0].save('figures/results.gif',
                       save_all=True, append_images=images[1:], optimize=False, duration=800, loop=0)
