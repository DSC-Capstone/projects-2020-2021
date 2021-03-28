from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.applications import ResNet50V2

def build_model(num_classes, size = 224):
    
    model = Sequential()
    
    tr = ResNet50V2(include_top = False,
                       weights = "imagenet",
                       input_shape = (size, size, 3))
    
    #Transfer Learning
    for layer in tr.layers[:14]:
        layer.trainable = False
        model.add(layer)
    
    #Convolution layers
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    

    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(256, (3, 3), padding = "same"))
    model.add(Conv2D(256, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    #Dense layers
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.7))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(128, activation = "relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(num_classes, activation = "softmax"))
    
    return model

def build_model_light(num_classes, size = 224):
    
    model = Sequential()
    
    tr = ResNet50V2(include_top = False,
                       weights = "imagenet",
                       input_shape = (size, size, 3))
    
    #Transfer Learning
    for layer in tr.layers[:14]:
        layer.trainable = False
        model.add(layer)
    
    #Convolution layers
    model.add(Conv2D(32, (3, 3), padding = "same"))
    model.add(Conv2D(32, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    #Dense layers
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.7))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(128, activation = "relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(num_classes, activation = "softmax"))
    
    return model

