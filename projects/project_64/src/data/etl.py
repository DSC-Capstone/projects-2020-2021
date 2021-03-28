import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

def get_features_labels(file_name, features, spectators, labels, nlabels, remove_mass_pt_window=True, entrystop=None):
    
    nfeatures = len(features)
    nspectators = len(spectators)
    nlabels = 2
    
    # load file
    root_file = uproot.open(file_name)
    try:
        tree = root_file['deepntuplizer/tree']
    except:
        tree = root_file['deepntuplizertree']
        
    feature_array = tree.arrays(branches=features, 
                                entrystop=entrystop,
                                namedecode='utf-8')
    spec_array = tree.arrays(branches=spectators, 
                             entrystop=entrystop,
                             namedecode='utf-8')
    label_array_all = tree.arrays(branches=labels, 
                                  entrystop=entrystop,
                                  namedecode='utf-8')

    feature_array = np.stack([feature_array[feat] for feat in features],axis=1)
    spec_array = np.stack([spec_array[spec] for spec in spectators],axis=1)
    
    njets = feature_array.shape[0]
    
    label_array = np.zeros((njets,nlabels))
    label_array[:,0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] + \
                                                          label_array_all['label_QCD_bb'] + \
                                                          label_array_all['label_QCD_c'] + \
                                                          label_array_all['label_QCD_cc'] + \
                                                          label_array_all['label_QCD_others'])
    label_array[:,1] = label_array_all['label_H_bb']

    
    #feature_array = clean_array(feature_array, spec_array, remove_mass_pt_window)
    #label_array = clean_array(label_array, spec_array, remove_mass_pt_window)
    # spec_array = clean_array(spec_array, spec_array, remove_mass_pt_window)
    
    return feature_array, label_array, spec_array, tree

def clean_array(arr, spec_array,remove_mass_pt_window):
    # remove samples outside mass/pT window
    if remove_mass_pt_window:
        arr = arr[(spec_array[:,0] > 40) & (spec_array[:,0] < 200) & (spec_array[:,1] > 300) & (spec_array[:,1] < 2000)]
    
    # remove unlabeled data
    return arr#[np.sum(arr,axis=1)==1]
