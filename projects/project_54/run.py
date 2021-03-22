import sys
sys.path.append("./src")
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from model_trans import *
from util import *
from training import *
import json
from tensorflow.keras.applications import resnet_v2
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
import PIL.Image as Image
tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    
    targets = sys.argv[1:]
   
    with open("./config/parameters.json") as param:
        data = json.load(param)
    param.close()
    
    model_param = data["model_param"]
    load_data = data["load_data"]
    generate_stats = data["generate_stats"]
    run_test = data["run_test"]
    run_custom_img = data["run_custom_img"]

    #Train model
    if "train_model" in targets:

        lr, epochs, batch_size, mapping_path, save_directory, save_path,log_directory, log_path = model_param.values()
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

        train_label_path, train_image_path, valid_label_path, valid_image_path, target, size = load_data.values()

        num_classes = pd.read_csv(valid_label_path)[target].nunique()


        train_gen = create_generator(train_label_path,
                                     train_image_path,
                                     target,
                                     size,
                                     batch_size,
                                     mapping_path,
                                     resnet_v2.preprocess_input, 
                                     is_training = True)

        valid_gen = create_generator(valid_label_path,
                                     valid_image_path,
                                     target,
                                     size,
                                     batch_size,
                                     mapping_path,
                                     resnet_v2.preprocess_input, 
                                     is_training = False)

        model = build_model(num_classes = num_classes)

        print(model.summary())

        training(model, train_gen, valid_gen, lr, epochs, save_path, log_path)
    
    #generate statistics
    if "generate_stats" in targets:
        log_path, label_path, image_path, target, save_path, model_path, mapping_path = generate_stats.values()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        model = load_model(model_path)
        
        print("model loaded")
        generator = create_generator(label_path,
                                         image_path,
                                         target,
                                         224,
                                         128,
                                         mapping_path,
                                         resnet_v2.preprocess_input, 
                                         is_training = False)


        generate_curves(log_path, save_path)
        create_stats(model, generator, target, label_path, mapping_path, save_path)
    
    #run on test sample
    if "run_test" in targets:
        model_path, mapping, image_path, to_save = run_test.values()
        face_img = Image.open(image_path)
        ig = integrated_grad_PIL(model_path, mapping, face_img, to_save = to_save)
        gradcam, guided = grad_cam(model_path, mapping, face_img, to_save = to_save)
        
    #test your own image
    if "run_custom_img" in targets:
        model_path, mapping, image_path, to_save = run_custom_img.values()
        to_save = bool(to_save)
        face_img = detect_face(image_path, to_save = to_save)
        ig = integrated_grad_PIL(model_path, mapping, face_img, to_save = to_save)
        gradcam, guided = grad_cam(model_path, mapping, face_img, to_save = to_save)
        


    
