from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from collections import defaultdict
import pandas as pd
import networkx as nx


def clean_players(players):
    players['name'] = players['name'].replace('Goran Dragić', 'Goran Dragic')
    players['name'] = players['name'].replace('Dennis Schröder', 'Dennis Schroder')
    players['name'] = players['name'].replace('Sasha Vujačić', 'Sasha Vujacic')
    players['name'] = players['name'].replace('José Calderón', 'Jose Calderon')
    players['name'] = players['name'].replace('Ömer Aşık', 'Omer Asik')
    players['name'] = players['name'].replace('Alexis Ajinça', 'Alexis Ajinca')
    players['name'] = players['name'].replace('James Ennis', 'James Ennis III')
    players['name'] = players['name'].replace('Bojan Bogdanović', 'Bojan Bogdanovic')
    players['name'] = players['name'].replace('J.J. Redick', 'JJ Redick')
    players['name'] = players['name'].replace('P.J. Hairston', 'PJ Hairston')
    players['name'] = players['name'].replace('J.R. Smith', 'JR Smith')
    players['name'] = players['name'].replace('Kevin Séraphin', 'Kevin Seraphin')
    players['name'] = players['name'].replace('Tomáš Satoranský', 'Tomas Santoransky')
    players['name'] = players['name'].replace('Greivis Vásquez', 'Greivis Vasquez')
    players['name'] = players['name'].replace('Anderson Varejão', 'Anderson Varejao')
    players['name'] = players['name'].replace('Manu Ginóbili', 'Manu Ginobili')
    players['name'] = players['name'].replace('Dāvis Bertāns', 'Davis Bertans')
    players['name'] = players['name'].replace('Nicolás Laprovíttola', 'Nicolas Laprovittola')
    players['name'] = players['name'].replace('Álex Abrines', 'Alex Abrines')
    players['name'] = players['name'].replace('Dario Šarić', 'Dario Saric')
    players['name'] = players['name'].replace('Sergio Rodríguez', 'Sergio Rodriguez')
    players['name'] = players['name'].replace('Tim Hardaway', 'Tim Hardaway Jr.')
    players['name'] = players['name'].replace('Glenn Robinson', 'Glenn Robinson III')
    players['name'] = players['name'].replace('C.J. Miles', 'CJ Miles')
    players['name'] = players['name'].replace('Timothé Luwawu-Cabarrot','Timothe Luwawu-Cabarrot')
    players['name'] = players['name'].replace('Derrick Jones', 'Derrick Jones Jr.')
    players['name'] = players['name'].replace('Tomas Santoransky', 'Tomas Satoransky')
    players['name'] = players['name'].replace('Wesley Iwundu', 'Wes Iwundu')
    players['name'] = players['name'].replace('Miloš Teodosić', 'Milos Teodosic')
    players['name'] = players['name'].replace('Dennis Smith', 'Dennis Smith Jr.')
    players['name'] = players['name'].replace('Bogdan Bogdanović', 'Bogdan Bogdanovic')
    players['name'] = players['name'].replace('Danuel House', 'Danuel House Jr.')
    players['name'] = players['name'].replace('Gary Payton', 'Gary Payton II')
    players['name'] = players['name'].replace('Walt Lemon', 'Walter Lemon Jr.')
    
def map_id(df, player_map):
    df['HOME_PLAYER_ID_1'] = df['HOME_PLAYER_ID_1'].map(player_map)
    df['HOME_PLAYER_ID_2'] = df['HOME_PLAYER_ID_2'].map(player_map)
    df['HOME_PLAYER_ID_3'] = df['HOME_PLAYER_ID_3'].map(player_map)
    df['HOME_PLAYER_ID_4'] = df['HOME_PLAYER_ID_4'].map(player_map)
    df['HOME_PLAYER_ID_5'] = df['HOME_PLAYER_ID_5'].map(player_map)
    df['AWAY_PLAYER_ID_1'] = df['AWAY_PLAYER_ID_1'].map(player_map)
    df['AWAY_PLAYER_ID_2'] = df['AWAY_PLAYER_ID_2'].map(player_map)
    df['AWAY_PLAYER_ID_3'] = df['AWAY_PLAYER_ID_3'].map(player_map)
    df['AWAY_PLAYER_ID_4'] = df['AWAY_PLAYER_ID_4'].map(player_map)
    df['AWAY_PLAYER_ID_5'] = df['AWAY_PLAYER_ID_5'].map(player_map)

def create_network(mascot,df,players,teams,mascots,year):
    subset = df.loc[df.HOME_TEAM.str.contains(mascot) | df.AWAY_TEAM.str.contains(mascot)]
    subset = subset[['GAME_ID','HOMEDESCRIPTION', 'VISITORDESCRIPTION','PCTIMESTRING', 'PERIOD', 'HOME_TEAM', 'AWAY_TEAM',
                'HOME_PLAYER_ID_1','HOME_PLAYER_ID_2', 'HOME_PLAYER_ID_3','HOME_PLAYER_ID_4','HOME_PLAYER_ID_5',
                'AWAY_PLAYER_ID_1','AWAY_PLAYER_ID_2', 'AWAY_PLAYER_ID_3','AWAY_PLAYER_ID_4','AWAY_PLAYER_ID_5']]
    edge_list = []
    nums = 0
    for game_id in subset['GAME_ID'].unique():
        boolean = False
        dic = defaultdict(int)
        subs = subset.loc[df.GAME_ID == game_id]
        names = subs.iloc[0][['HOME_PLAYER_ID_1','HOME_PLAYER_ID_2','HOME_PLAYER_ID_3','HOME_PLAYER_ID_4','HOME_PLAYER_ID_5']].values
        for name in names: 
            for team in list(players.loc[players['name'] == name].team.values):
                dic[team] += 1
        boolean = teams[max(dic, key=dic.get)] == mascot

        if boolean:        
            new = subs[['HOMEDESCRIPTION', 'PCTIMESTRING', 'PERIOD', 'HOME_TEAM', 'HOME_PLAYER_ID_1',
                             'HOME_PLAYER_ID_2', 'HOME_PLAYER_ID_3', 'HOME_PLAYER_ID_4', 'HOME_PLAYER_ID_5']]
            new['PCTIMESTRING'] = new['PCTIMESTRING'].str.split(":").apply(lambda x: int(x[0])*60 + int(x[1]))
            new = new[new['HOMEDESCRIPTION'].shift(-1).str.contains('SUB', na=False) |
               new['HOMEDESCRIPTION'].str.contains('SUB', na=False)]
            new = new[~new['HOMEDESCRIPTION'].str.contains('SUB', na=False)]

            curr_seconds = 720
            curr_period = 1
            for row in new.itertuples():
                if row.PERIOD > curr_period:
                    curr_period = row.PERIOD 
                    curr_seconds = 720 
                mutual_time = curr_seconds - row.PCTIMESTRING
                for p1 in range(5,9):
                    for p2 in range(p1+1, 9):
                        mutual_mins = round((curr_seconds - row.PCTIMESTRING)/60, 2)
                        edge_list.append([sorted([row[p1], row[p2]])[0], sorted([row[p1], row[p2]])[1] ,mutual_mins])
                curr_seconds = row.PCTIMESTRING

        else:
            new = subs[['VISITORDESCRIPTION', 'PCTIMESTRING', 'PERIOD', 'AWAY_TEAM', 'AWAY_PLAYER_ID_1',
                             'AWAY_PLAYER_ID_2', 'AWAY_PLAYER_ID_3', 'AWAY_PLAYER_ID_4', 'AWAY_PLAYER_ID_5']]
            new['PCTIMESTRING'] = new['PCTIMESTRING'].str.split(":").apply(lambda x: int(x[0])*60 + int(x[1]))
            new = new[new['VISITORDESCRIPTION'].shift(-1).str.contains('SUB', na=False) |
                   new['VISITORDESCRIPTION'].str.contains('SUB', na=False)]
            new = new[~new['VISITORDESCRIPTION'].str.contains('SUB', na=False)]

            curr_seconds = 720
            curr_period = 1
            for row in new.itertuples():
                if row.PERIOD > curr_period:
                    curr_period = row.PERIOD 
                    curr_seconds = 720 
                mutual_time = curr_seconds - row.PCTIMESTRING
                for p1 in range(5,9):
                    for p2 in range(p1+1, 9):
                        mutual_mins = round((curr_seconds - row.PCTIMESTRING)/60, 2)
                        edge_list.append([sorted([row[p1], row[p2]])[0], sorted([row[p1], row[p2]])[1] ,mutual_mins])
                curr_seconds = row.PCTIMESTRING
    edges = pd.DataFrame(edge_list)
    edges.columns =['P1', 'P2', 'MutualTime'] 
    edges = edges.groupby(['P1','P2']).sum().reset_index()
    
    team = mascots.get(mascot)
    edges['P1'] = edges['P1'] + '_' + team + '_' + str(year)
    edges['P2'] = edges['P2'] + '_' + team + '_' + str(year)
    edgeList = edges.values.tolist()
    return edgeList

def embedd_stats(G,players_data):
    ppg_dict = {}
    apg_dict = {}
    rpg_dict = {}
    three_point_percentage_dict = {}
    two_point_percentage_dict = {}
    mp_dict = {}
    gp_dict = {}

    for idx,node in players_data.iterrows():
        ppg_dict[node[10]] = node[3]
        apg_dict[node[10]] = node[4]
        rpg_dict[node[10]] = node[5]
        three_point_percentage_dict[node[10]] = node[6]
        two_point_percentage_dict[node[10]] = node[7]
        mp_dict[node[10]] = node[8]
        gp_dict[node[10]] = node[9]

    nx.set_node_attributes(G, ppg_dict, 'ppg')
    nx.set_node_attributes(G, apg_dict, 'apg')
    nx.set_node_attributes(G, rpg_dict, 'rpg')
    nx.set_node_attributes(G, three_point_percentage_dict, '3P%')
    nx.set_node_attributes(G, two_point_percentage_dict, '2P%')
    nx.set_node_attributes(G, mp_dict, 'mp')
    nx.set_node_attributes(G, gp_dict, 'gp')

def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1

        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break

    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    #get the first val_num
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx, test_idx

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, len(label_encoder.classes_)

def build_adj(nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    #obtain the adjacency matrix (A)
    A = nx.adjacency_matrix(G)
    return A