import pandas as pd
import numpy as np

abbr = [
    'PHO', 'DAL', 'POR', 'OKC', 'DEN', 'MEM', 'WAS', 'MIA', 'BRK', 'CLE', 'TOR',
    'NOP', 'HOU', 'IND', 'LAC', 'PHI', 'SAC', 'UTA', 'LAL', 'BOS', 'ORL', 'MIL',
    'SAS', 'ATL', 'GSW', 'CHI', 'NYK', 'DET', 'MIN', 'CHO'
]

def make_data():
    years = [11, 12, 13, 14, 15, 16, 17, 18, 19]

    data = pd.DataFrame()
    edges = pd.DataFrame()

    id_r = np.array(list(range(0, 30, 1)))

    for yr in years:
        fp1 = 'data/features/feat20' + str(yr) + '.csv'
        fp2 = 'data/schedule/sch20' + str(yr)

        curr = pd.read_csv(fp1)

        curr['id'] = curr['Tm'].map(dict(zip(abbr, id_r)))

        data = data.append(curr)

        curr_edge = pd.read_csv(fp2)

        curr_edge['Home'] = curr_edge['Home'].map(dict(zip(abbr, id_r)))
        curr_edge['Away'] = curr_edge['Away'].map(dict(zip(abbr, id_r)))

        edges = edges.append(curr_edge)

        id_r = id_r + 30

    return data, edges
