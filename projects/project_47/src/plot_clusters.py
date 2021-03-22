import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def plot_2d_cluster(df, embs, col, min_cnt=500, subplot_size=7):
    """Plot 2D embeddings against existing features."""
    print(f"[clustering] plotting 2d clusters vs. {col}...")
    mlb = MultiLabelBinarizer()
    bin_col = mlb.fit_transform(df[col])
    bin_col = pd.DataFrame(bin_col, columns=mlb.classes_)
    val_tp = (
        (-bin_col.sum(axis=0)[bin_col.sum(axis=0).ge(min_cnt)])
        .sort_values()
        .index.values
    )
    bin_col_tp = bin_col[val_tp]

    ncol = 3
    if bin_col_tp.shape[1] < ncol:
        print("[clustering] warning: ignoring min_cnt since it's too large.")
        bin_col_tp = bin_col

    nrow = int(np.ceil(bin_col_tp.shape[1] / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * subplot_size, nrow * subplot_size)
    )

    for val, ax in zip(bin_col_tp.columns, axes.flatten()):
        ax.set_title(val)
        embs_neg = embs[bin_col_tp[val][bin_col_tp[val] == 0].index]
        embs_pos = embs[bin_col_tp[val][bin_col_tp[val] == 1].index]
        ax.scatter(embs_neg[:, 0], embs_neg[:, 1], c="lightgrey", s=60)
        ax.scatter(embs_pos[:, 0], embs_pos[:, 1], c="C0", s=60, alpha=0.15)
    fig.savefig(f"data/figures/2d_cluster_{col}.png")


def plot_2d_cluster_by_year(df, embs, subplot_size=7):
    """Plot 2D embeddings against years."""
    print("[clustering] plotting 2d clusters vs. years...")
    onehot = OneHotEncoder()
    bin_year = onehot.fit_transform(
        df.date.apply(lambda dt: dt.year).to_frame()
    ).toarray()
    bin_year = pd.DataFrame(
        bin_year,
        columns=list(map(lambda fn: fn[3:-2], onehot.get_feature_names())),
    ).astype(int)
    bin_year = bin_year.rename(columns={"n": "N/A"})
    ncol = 11
    nrow = int(np.ceil(bin_year.shape[1] / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * subplot_size, nrow * subplot_size)
    )

    for val, ax in zip(bin_year.columns, axes.flatten()):
        ax.set_title(val)
        embs_neg = embs[bin_year[val][bin_year[val] == 0].index]
        embs_pos = embs[bin_year[val][bin_year[val] == 1].index]
        ax.scatter(embs_neg[:, 0], embs_neg[:, 1], c="lightgrey", s=60)
        ax.scatter(embs_pos[:, 0], embs_pos[:, 1], c="C0", s=60, alpha=0.35)
    fig.savefig(f"data/figures/2d_cluster_years.png")


def plot_all_clusters(df, embs):
    """Run plot functions."""
    os.makedirs("data/figures", exist_ok=True)
    for feature in ["languages", "countries", "genres"]:
        plot_2d_cluster(df, embs, feature)
    plot_2d_cluster_by_year(df, embs)
