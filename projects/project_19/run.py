from src.models.graphsage import *
from src.data.data import *
from src.utils.utils import *

import warnings
import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import csv
from itertools import permutations
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import numpy as np
import io
warnings.filterwarnings('ignore')


def main():
    players_data = pd.read_csv('data/NBA_players.csv')
    teams_data = pd.read_csv('data/NBA_teams.csv')
    clean_players(players_data)
    players_data['Combined'] = players_data['name'] + '_' +  players_data['team'] +  '_' + players_data['year'].astype(str)
    teams_data['Combined'] = teams_data['Abbrev'] + '_' +  teams_data['Year'].astype(str)

   
    id_df = pd.read_csv('data/playerlist.csv')
    id_df = id_df[['DISPLAY_FIRST_LAST', 'PERSON_ID']]
    player_map = pd.Series(id_df.DISPLAY_FIRST_LAST.values,index=id_df.PERSON_ID).to_dict()

    teams = {'CLE':'Cavaliers', 'TOR':'Raptors', 'MIA':'Heat', 'ATL':'Hawks', 'BOS': 'Celtics', 'CHO':'Hornets',
        'IND':'Pacers', 'DET': 'Pistons', 'CHI':'Bulls', 'WAS':'Wizards','ORL':'Magic', 'NYK':'Knicks',
        'BRK':'Nets', 'GSW':'Warriors', 'SAS':'Spurs', 'OKC':'Thunder', 'LAC':'Clippers', 'POR':'Blazers',
        'DAL':'Mavericks', 'MEM':'Grizzlies', 'HOU':'Rockets', 'UTA':'Jazz', 'SAC':'Kings', 'DEN':'Nuggets', 
        'NOP':'Pelicans', 'MIN':'Timberwolves', 'PHO':'Suns', 'LAL':'Lakers', 'MIL':'Bucks', 'PHI':'76ers'}

    mascots = {value:key for key, value in teams.items()}

    edges = []
    for x in range(2016, 2020):
        year = x
        previous_year = x-1
        
        URL = 'https://eightthirtyfour.com/nba/pbp/events_{}-{}_pbp.csv'
        URL = URL.format(str(previous_year), str(year))
        print("Currently reading: " + URL)
        r = requests.get(URL, verify=False).content
        
        df = pd.read_csv(io.StringIO(r.decode('utf-8')))
        players = players_data.loc[players_data.year == year]
        clean_players(players)
        map_id(df, player_map)
        for i in teams.values():
            team_edge = create_network(i, df, players,teams, mascots,year)
            edges.append(team_edge)

    print("Creating Player Edges")
    G = nx.Graph()
    for team in range(len(edges)):
        for player_edge in range(len(edges[team])):
            G.add_edge(edges[team][player_edge][0], edges[team][player_edge][1], weight=edges[team][player_edge][2])
    edge,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    print("Embedding Player Statistics")
    embedd_stats(G,players_data)

    matrix = nx.adjacency_matrix(G).A
    nodes = list(G.nodes())
    # labels = [teams_data.loc[teams_data.Combined == x[-8:]].Standing.item() for x in nodes]
    labels = [teams_data.loc[teams_data.Combined == x[-8:]].Playoff.item() for x in nodes]
    rows = len(matrix)
    columns = len(matrix[0])
    num_classes = 2
    A = build_adj(nodes,list(edge))
    train_idx, val_idx, test_idx = limit_data(labels)

    train_mask = np.zeros((rows,),dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros((rows,),dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros((rows,),dtype=bool)
    test_mask[test_idx] = True

    labels_encoded, classes = encode_label(labels)

    GraphSage(A, columns, rows, matrix, train_mask, val_mask, labels_encoded, num_classes, 'mean', 10, 0.5, 5e-4, 0.001,200, 200)

    
if __name__ == '__main__':
    main()
