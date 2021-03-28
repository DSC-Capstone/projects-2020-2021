import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam

##THIS IS IMPORTANT!
tf.compat.v1.disable_eager_execution()

def training(model, train_gen, valid_gen, lr, epochs, save_path, log_path):
    optimizer = Nadam(lr=lr) 
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    model.compile(optimizer=optimizer, 
                  loss= 'categorical_crossentropy', 
                  metrics= ["accuracy"])
    
    callbacks = [
        ModelCheckpoint(
                        filepath = save_path,
                        save_weights_only = False, 
                        save_best_only = True,
                        monitor='val_loss',
                        mode = "min"),

        EarlyStopping(monitor = "val_loss",
                      patience = 30,
                      mode = "min"),
        
        CSVLogger(log_path,
                  separator = ",",
                  append = False),
        
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience= 10,
                          mode = "min")
        ]

    #print("curdir:", os.path.abspath(os.curdir))
    #print("save_path:", save_path)

    history = model.fit(
        x = train_gen,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = valid_gen,
        use_multiprocessing = True,
        workers = 8
    )

    return history
    
