import sys
import os

def test_model_param(train_label_path, train_image_path, valid_label_path, valid_image_path, target, size):
    error = False

    if not os.path.exists(train_label_path):
        error = True
        print("The train_label_path: {} does not exist, please modify the path".format(train_label_path))
        return error

    if not os.path.exists(train_image_path):
        error = True
        print("The train_image_path: {} does not exist, please modify the path".format(train_image_path))
        return error

    if not os.path.exists(valid_label_path):
        error = True
        print("The valid_label_path: {} does not exist, please modify the path".format(valid_label_path))
        return error

    if not os.path.exists(valid_image_path):
        error = True
        print("The valid_image_path: {} does not exist, please modify the path".format(valid_image_path))
        return error

    if target not in ["age", "race", "gender"]:
        error = True
        print("{}  is not a valid target, please specify age, race, or gender as your target".format(train_label_path))
        return error
    
    if (size > 224):
        error = True
        print("Current size: {}. It is not recommended to be greater than 224. Please change.".format(train_label_path))
        return error
    elif(size < 100):
        error = True
        print("Current size: {}. It is not recommended to be less than than 100. Please change.".format(train_label_path))
        return error
    else:
        pass
    
    return error    

def test_integrated_param(model_param_path, image_path, label_path, save_path, target, mapping, size, img_idx_lst):
    error = False

    if not os.path.exists(model_param_path):
        error = True
        print("The model_param_Path: {} does not exist, please modify the path".format(train_label_path))
        return error

    if not os.path.exists(image_path):
        error = True
        print("The image_path: {} does not exist, please modify the path".format(train_image_path))
        return error

    if not os.path.exists(label_path):
        error = True
        print("The label_path: {} does not exist, please modify the path".format(valid_label_path))
        return error

    if not os.path.exists(save_path):
        error = True
        print("The save_path: {} does not exist, please modify the path".format(valid_image_path))
        return error

    if target not in ["age", "race", "gender"]:
        error = True
        print("{} is not a valid target, please specify age, race, or gender as your target".format(train_label_path))
        return error
    
    if not os.path.exists(mapping):
        error = True
        print("The mapping: {} does not exist, please modify the path".format(mapping))
        return error
    
    if len(img_idx_lst) == 0:
        error = True
        print("you should have more than 1 image included!")
        return error
        
    
    return error  
    