import sys
import json
import numpy as np

import tensorflow.keras as keras
from tensorflow.random import set_seed
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot


import tensorflow.keras.backend as K


import yaml

from DataGenerator import DataGenerator

sys.path.insert(0, '../data')
sys.path.insert(0, 'src/visualizations')

from generator import generator
from visualize import visualize
from visualize import visualize_loss
from visualize import visualize_roc

#setting seeds for consistent results
np.random.seed(2)
set_seed(3)

#GNN Additions
from GraphDataset import GraphDataset
from gnn_classes import *


def create_models(features, spectators, labels, nfeatures, nspectators, nlabels, ntracks, train_files, test_files, val_files, batch_size, remove_mass_pt_window, remove_unlabeled, max_entry):

    #imports
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, Flatten, Lambda
    
    # DATA GENERATORS FOR USE IN MODEL TRAINING AND TESTING
    train_generator = DataGenerator(train_files, features, labels, spectators, batch_size=batch_size, n_dim=ntracks, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
    
    val_generator = DataGenerator(val_files, features, labels, spectators, batch_size=batch_size, n_dim=ntracks, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
    
    test_generator = DataGenerator(test_files, features, labels, spectators, batch_size=batch_size, n_dim=ntracks, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
 

    #weights for training
    training_weights = {0:3.479, 1:4.002, 2:3.246, 3:2.173, 4:0.253, 5:1.360}
    
    # FULLY CONNECTED NEURAL NET CLASSIFIER
    

    # define dense keras model
    inputs = Input(shape=(ntracks,nfeatures,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Flatten(name='flatten_1')(x)
    x = Dense(64, name = 'dense_1', activation='relu')(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    outputs = Dense(nlabels, name = 'output', activation='softmax')(x)
    keras_model_dense = Model(inputs=inputs, outputs=outputs)
    keras_model_dense.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(keras_model_dense.summary())

    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=35)
    reduce_lr = ReduceLROnPlateau(patience=5,factor=0.5)
    model_checkpoint = ModelCheckpoint('keras_model_dense_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # fit keras model
    history_dense = keras_model_dense.fit_generator(train_generator, 
                                                    validation_data = val_generator, 
                                                    steps_per_epoch=len(train_generator), 
                                                    validation_steps=len(val_generator),
                                                    max_queue_size=5,
                                                    epochs=50,
                                                    class_weight=training_weights, 
                                                    shuffle=False,
                                                    callbacks = callbacks, 
                                                    verbose=0)
    # reload best weights
    keras_model_dense.load_weights('keras_model_dense_best.h5')

    visualize_loss(history_dense)
    visualize('fcnn_loss.png')


    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, Flatten, Lambda, GlobalAveragePooling1D
    import tensorflow.keras.backend as K
    

    # define Deep Sets model with Conv1D Keras layer
    inputs = Input(shape=(ntracks,nfeatures,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Conv1D(64, 1, strides=1, padding='same', name = 'conv1d_1', activation='relu')(x)
    x = Conv1D(32, 1, strides=1, padding='same', name = 'conv1d_2', activation='relu')(x)
    x = Conv1D(32, 1, strides=1, padding='same', name = 'conv1d_3', activation='relu')(x)
    
    # sum over tracks
    x = GlobalAveragePooling1D(name='pool_1')(x)
    x = Dense(100, name = 'dense_1', activation='relu')(x)
    outputs = Dense(nlabels, name = 'output', activation='softmax')(x)
    
    
    keras_model_conv1d = Model(inputs=inputs, outputs=outputs)
    keras_model_conv1d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(keras_model_conv1d.summary())

    # define callbacks
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    early_stopping = EarlyStopping(monitor='val_loss', patience=35)
    
    #defining learningrate decay model
    num_epochs = 100
    initial_learning_rate = 0.01
    decay = initial_learning_rate / num_epochs
    learn_rate_decay = lambda epoch, lr: lr * 1 / (1 + decay * epoch)
    reduce_lr2 = ReduceLROnPlateau(patience=5,factor=0.5)
    
    #reduce_lr = ReduceLROnPlateau(patience=5,factor=0.5)
    reduce_lr = LearningRateScheduler(learn_rate_decay)
    model_checkpoint = ModelCheckpoint('keras_model_conv1d_best.h5', monitor='val_loss', save_best_only=True)
    #callbacks = [early_stopping, model_checkpoint, reduce_lr2]
    callbacks = [early_stopping, model_checkpoint, reduce_lr2]
    
    #weights for training
    training_weights = {0:3.479, 1:4.002, 2:3.246, 3:2.173, 4:0.253, 5:1.360}
    
    # fit keras model
    history_conv1d = keras_model_conv1d.fit_generator(train_generator, 
                                                      validation_data = val_generator, 
                                                      steps_per_epoch=len(train_generator), 
                                                      validation_steps=len(val_generator),
                                                      max_queue_size=5,
                                                      epochs=num_epochs,
                                                      class_weight=training_weights,
                                                      shuffle=False,
                                                      callbacks = callbacks, 
                                                      verbose=0)
    # reload best weights
    keras_model_conv1d.load_weights('keras_model_conv1d_best.h5')

    visualize_loss(history_conv1d)
    visualize('conv1d_loss.png')

    #GNN START
    
    #load data
    graph_dataset = GraphDataset('gdata_train', features, labels, spectators, n_events=1000, n_events_merge=1, 
                             file_names=train_files)
    graph_dataset.process()
    
    #understand data
    from torch_geometric.data import Data, DataListLoader, Batch
    from torch.utils.data import random_split
    
    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(graph_dataset)
    valid_num = int(valid_frac*full_length)
    batch_size = 32

    train_dataset, valid_dataset = random_split(graph_dataset, [full_length-valid_num,valid_num])

    train_loader = DataListLoader(graph_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    
    #create gnn model
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric.transforms as T
    from torch_geometric.nn import EdgeConv, global_mean_pool
    from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
    from torch_scatter import scatter_mean
    from torch_geometric.nn import MetaLayer

    model = InteractionNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    import os.path as osp
    
    n_epochs = 20
    stale_epochs = 0
    best_valid_loss = 99999
    patience = 5
    t = tqdm(range(0, n_epochs))
    
    
    # calculate weights    
    s = 0
    for data in graph_dataset:
        d = data[0].y[0]
        if s is 0:
            s = d
        else:

            s += d
    weights = []
    for w in s:
        # wi = (# jets)/(# classes * # jets in class)
        den = w.item() * 6
        num = sum(s).item()
        weights += [num/den] 
            
    
    for epoch in t:
        loss = train(model, optimizer, train_loader, train_samples, batch_size,leave=bool(epoch==n_epochs-1), weights=weights)
        valid_loss = test(model, valid_loader, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('           Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join('interactionnetwork_best.pth')
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break
    
    #load test data
    test_dataset = GraphDataset('data', features, labels, spectators, n_events=1000, n_events_merge=1, 
                             file_names=test_files)
    test_dataset.process()
    
    test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate

    test_samples = len(test_dataset)
    
    #model evaluation
    model.eval()
    t = tqdm(enumerate(test_loader),total=test_samples/batch_size)
    y_test = []
    y_predict = []

    for i,data in t:
        data = data.to(device)    
        batch_output = model(data.x, data.edge_index, data.batch)    
        y_predict.append(batch_output.detach().cpu().numpy())
        y_test.append(data.y.cpu().numpy())
    y_test = np.concatenate(y_test)
    y_predict = np.concatenate(y_predict)
    
    #GNN END

    # COMPARING MODELS
    predict_array_dnn = []
    predict_array_cnn = []
    label_array_test = []

    for t in test_generator:
        label_array_test.append(t[1])
        predict_array_dnn.append(keras_model_dense.predict(t[0]))
        predict_array_cnn.append(keras_model_conv1d.predict(t[0]))


    predict_array_dnn = np.concatenate(predict_array_dnn,axis=0)
    predict_array_cnn = np.concatenate(predict_array_cnn,axis=0)
    label_array_test = np.concatenate(label_array_test,axis=0)

    fpr_dnn = []
    tpr_dnn = []
    fpr_cnn = []
    tpr_cnn = []
    fpr_gnn = []
    tpr_gnn = []
    # create ROC curves for each class
    for i in range(nlabels):
        t_fpr_d, t_tpr_d, thresh_d = roc_curve(label_array_test[:,i], predict_array_dnn[:,i])
        t_fpr_c, t_tpr_c, thresh_c = roc_curve(label_array_test[:,i], predict_array_cnn[:,i])
        t_fpr_g, t_tpr_g, thresh_g = roc_curve(y_test[:,i], y_predict[:,i])
        
        #appending
        fpr_dnn.append(t_fpr_d)
        tpr_dnn.append(t_tpr_d)
        fpr_cnn.append(t_fpr_c)
        tpr_cnn.append(t_tpr_c)
        fpr_gnn.append(t_fpr_g)
        tpr_gnn.append(t_tpr_g)

    # plot ROC curves
    visualize_roc(fpr_cnn, tpr_cnn, fpr_dnn, tpr_dnn, fpr_gnn, tpr_gnn)
    visualize('fnn_vs_conv1d.pdf')
    
    

