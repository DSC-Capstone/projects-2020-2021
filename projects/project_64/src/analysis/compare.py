import sys
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, '../data')
sys.path.insert(0, 'src/visualizations')

from etl import get_features_labels, clean_array
from visualize import visualize
from visualize import visualize_hist
from visualize import visualize_roc_compare



def compare(file_name, features, spectators, labels, nlabels, remove_mass_pt_window, entrystop,
           jet_features, track_features, sv_features, namedecode):
    
    data = get_features_labels(file_name, features, spectators, labels, nlabels, remove_mass_pt_window=True, entrystop=None)
    features, labels, specs, tree = data

    label_QCD = labels[:,0]
    label_Hbb = labels[:,1]

    print(sum(label_QCD+label_Hbb)/len(label_QCD+label_Hbb))


    jet_feat = jet_features
    track_feat = track_features
    sv_feat = sv_features

    track_features = tree.arrays(branches=track_feat,
                          entrystop=entrystop,
                          namedecode=namedecode)

    jet_features = tree.arrays(branches=jet_feat,
                          entrystop=entrystop,
                          namedecode=namedecode)

    sv_features = tree.arrays(branches=sv_feat,
                          entrystop=entrystop,
                          namedecode=namedecode)

    #jet_features = np.stack([jet_features[feat] for feat in ['fj_pt','fj_sdmass']],axis=1)
    #jet_features = clean_array(jet_features, specs, data_cfg['remove_mass_pt_window'])




    vis_path = 'data/visualizations/'

    # TRACK FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
    # number of tracks
    data = track_features['track_pt'].counts
    bin_vars = np.linspace(0,80,81)
    x_label = 'Number of tracks'
    y_label = 'Fraction of jets'
    file_name = 'trackcounts_hist.png'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('trackcounts_hist.png')

    # max. relative track pt
    data = track_features['track_pt'].max()/jet_features['fj_pt']
    bin_vars = np.linspace(0,0.5,51)
    x_label = r'Maximum relative track $p_{T}$'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('trackmaxrelpt_hist.png')


    # maximum signed 3D impact paramter value
    data = track_features['trackBTag_Sip3dVal'].max()
    bin_vars = np.linspace(-2,40,51)
    x_label = 'Maximum signed 3D impact parameter value'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('tracksip3val_hist.png')


    # maximum signed 3D impact paramter significance
    data = track_features['trackBTag_Sip3dSig'].max()
    bin_vars = np.linspace(-2,40,51)
    x_label = 'Maximum signed 3D impact parameter value'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('tracksip3sig_hist.png')


    # JET FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
    data = jet_features['fj_pt']
    bin_vars = np.linspace(0,4000,101)
    x_label = r'Jet $p_{T}$ [GeV]'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('fj_pt_hist.png')


    data = jet_features['fj_sdmass']
    bin_vars = np.linspace(0,300,101)
    x_label = r'Jet $m_{SD}$ [GeV]'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('fj_sdmass_hist.png')


    # SV FEATURES HISTOGRAMS DEPICTING DISCRIMINATORY EFFECT
    data = sv_features['sv_pt'].counts
    bin_vars = np.linspace(-2,40,51)
    x_label = 'SV pt Count'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('svptcounts_hist.png')


    data = sv_features['sv_mass'].counts
    bin_vars = np.linspace(-2,40,51)
    x_label = 'SV mass Count'
    y_label = 'Fraction of jets'
    visualize_hist(data, label_QCD, label_Hbb, bin_vars, x_label, y_label)
    visualize('svmasscounts_hist.png')


    # ROC CURVES

    disc = np.nan_to_num(sv_features['nsv'],nan=0)

    fpr, tpr, threshold = roc_curve(label_Hbb, disc)
    # plot ROC curve
    visualize_roc_compare(fpr, tpr)
    visualize('svcount_roc.png')


    disc = np.nan_to_num(sv_features['sv_pt'].max()/jet_features['fj_pt'],nan=0)

    fpr, tpr, threshold = roc_curve(label_Hbb, disc)
    # plot ROC curve
    visualize_roc_compare(fpr, tpr)
    visualize('maxsvpt-fjpt_roc.png')