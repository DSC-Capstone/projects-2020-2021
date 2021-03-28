from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import display, HTML
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from functools import partial
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score


def make_groundtruth_figures(data_folder, update_figs=False, no_labels=False):
    vectors = pd.read_csv(os.path.join(data_folder, 'features.csv'), index_col='app')
    
    if no_labels: # mostly for testing
        all_apps = vectors.assign(
            label=['app', 'app'],
            category=['app', 'app']
        )
    else:
        all_apps = pd.read_csv("data/out/all-apps/all_apps.csv", index_col='app')
        all_apps['label'] = all_apps[all_apps.category=='malware'].app_dir.str.split('/').apply(lambda list: list[5])
        top_9_malware = all_apps.label.value_counts().sort_values(ascending=False)[:9]
        top_9_min = top_9_malware.min()
        other_mal_map = {key: "Other malware" for key, value in all_apps.label.value_counts().items() if value <= top_9_min}
#         other_mal_map = {key: key for key, value in all_apps.label.value_counts().items() if value <= 200}
        all_apps.label = all_apps.label.map(other_mal_map).fillna(all_apps.label)
        all_apps.label.fillna(all_apps.category, inplace=True)
    
    vectors = vectors.assign(
        label=all_apps.label,
        category=all_apps.category
    )
    labels = vectors.label
    
    # Retrieve node embeddings and corresponding subjects
    node_ids = list(vectors.uid)  # list of node IDs
    node_embeddings = vectors.drop(columns=['uid', 'category', 'label'])
    node_targets = labels

    transform = TSNE  # Dimensionality reduction transformer

    # 2D plot -- matplotlib
    print('Making 2D plot...')
    plt.rcParams.update({'font.size': 14})
    
    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings)
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(10, 8))
    plt.axes().set(aspect="equal")
    scatter = plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap='tab20',
        alpha=1,
        s=5
    )
    plt.title("2D {} visualization of node embeddings".format(transform.__name__))
    legend1 = plt.legend(scatter.legend_elements()[0], pd.Series(label_map.keys()).str.replace('-', ' ').str.title(),
                        loc='center left', bbox_to_anchor=(1, 0.5), title="App Type", markerscale=1.5)
    
    # order labels (https://stackoverflow.com/a/46160465/13710014)
#     handles, g_labels = plt.gca().get_legend_handles_labels()
#     print(handles, labels)
#     if not no_labels:
#         order = ['Popular Apps', 'Random Apps']
#         order +=  list(top_9_malware.index)
#         plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.savefig(os.path.join(data_folder, '2D-plot.png'), bbox_inches='tight')
    
    # 3D plot - using plotly
    print('Making 3D plot...')
    trans3d = transform(n_components=3)
    node_embeddings_3d = trans3d.fit_transform(node_embeddings)
    data_3d = pd.DataFrame(node_embeddings_3d, index=vectors.index)
    data_3d['malware'] = vectors['category']=='malware'
    data_3d['type'] = vectors.label
    type_chart = data_3d[['malware', 'type']].drop_duplicates()
    type_chart['num'] = type_chart.type.map(label_map)

    layout = go.Layout(
        title="Interactive 3D TNSE representation of node embeddings",
        margin={'l': 0, 'r': 0, 'b': 0, 't': 30},
        legend=dict(y=0.5, itemsizing='constant'),
        scene={
            'xaxis': {
                'showspikes': False,
                'showgrid': False, 
                'zeroline': False, 
                'visible': False
            },
            'yaxis': {
                'showspikes': False,
                'showgrid': False, 
                'zeroline': False, 
                'visible': False
            },
            'zaxis': {
                'showspikes': False,
                'showgrid': False, 
                'zeroline': False, 
                'visible': False
            }
        }
    )

    fig = go.Figure(layout=layout)

    # add invisible bounding trace to keep axes' scale constant
    fig.add_trace(
        go.Scatter3d(
            x=[data_3d[0].min(), data_3d[0].max()],
            y=[data_3d[1].min(), data_3d[1].max()],
            z=[data_3d[2].min(), data_3d[2].max()],
            mode='markers',
            marker={
                'color':'rgba(0,0,0,0)',
                'opacity': 0,
            },
            showlegend=False
        )
    )

    for index, row in type_chart.sort_values('num', ascending=False).iterrows():
        if row['malware']:
            symbol = 'circle'
            group='Malware'
            size = 2
        else:
            symbol = 'x'
            group='Unlabeled'
            size = 1.5

        name = f"{group}, {row['type'].replace('-', ' ').title()}"

        if row['type']=='Other malware':
            name=row['type']

        df = data_3d[data_3d.type==row['type']]
        rbg = tuple([255*val for val in cm.tab20(row['num'])[:3]])
        color = f"rgb{rbg}"
        trace  = go.Scatter3d(
            name=name,
            x=df[0],
            y=df[1],
            z=df[2],
            customdata=list(df.index),
            hovertemplate=
            "<b>%{customdata}</b><br>" +
            f"{name}" +
            "<extra></extra>",
            mode='markers',
            marker={
                'size': size,
                'opacity': 1,
                'color': color,
                'symbol': symbol,
            },
        )

        fig.add_trace(trace)

    # Save the plot.
    pio.write_html(fig, file=os.path.join(data_folder, '3D-plot.html'), auto_open=True)
    
    if update_figs:
        pio.write_html(fig, file=os.path.join('docs', '_includes', '3D-plot.html'), auto_open=True)
    
def compute_model_performance_statistics(pred, true):
    '''
    Returns a series with the f1-score, accuracy, recall, and confusion counts (TP, TN, FP, FN).
    '''
    TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
    return pd.Series({
        'ACC': accuracy_score(true, pred),
        'TPR': recall_score(true, pred),
        'F1': f1_score(true, pred),
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    })
    
    
def create_performance_table(m2v_results_path, hindroid_results_path, outpath=None):
    results = pd.read_csv(m2v_results_path, index_col='app', usecols=['app', 'm2vDroid', 'true'])
    if 'true' in results.columns:
        results = results.drop(columns=['true'])
    results = results.join(pd.read_csv(hindroid_results_path, index_col='app'))
    y_true = results.true
    table = results.drop(columns=['true']).apply(partial(compute_model_performance_statistics, true=y_true)).T
    table = table.astype({col: int for col in ['TP', 'TN', 'FP', 'FN']})
    if outpath is not None:
        table.to_csv(outpath)
    return table

    
def generate_analysis(data_path, jobs={}):
    "Generates plots, aggregates, and statistical analysis on app data located in `data_path`"

    #  load data
#     app_data_path = os.path.join(data_path, 'app_data.csv')
#     app_data = dd.read_csv(app_data_path)
#     os.makedirs(out_folder, exist_ok=True)

    
    if "plots" in jobs:
        make_groundtruth_figures(data_path, **jobs['plots'])
