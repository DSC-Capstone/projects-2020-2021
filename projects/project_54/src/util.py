import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense
from tensorflow.keras import Sequential
import tensorflow as tf
from IntegratedGradients import *
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import resnet_v2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import dlib
import io
from model_trans import *
from grad_cam import *
tf.compat.v1.disable_eager_execution()

"""
Function to create generator from the csv file
input
    csv_path: path to the csv file
    image_path: path to the image directory
    target: the class of interest(age, gender, or race)
    size: the size of the image
    batch_size: the batch size
    preprocess_input: The preprocess function to apply based on different transfer learning model. Make sure to change
    the import statement above if wants to apply different transfer learning model
    mapping_path: a directionary objects indicating how each category is being mapped to the respective integer representation
    is_training: whether or not the generator is used as training

output
    a generator object ready to be trained
"""
def create_generator(csv_path, image_path, target, size, batch_size, mapping_path, preprocess_input, is_training):
    
    if is_training:
        rotation_range = 30
        horizontal_flip = True
        vertical_flip = True
        shuffle = True
    else:
        rotation_range = 0
        horizontal_flip = False
        vertical_flip = False
        shuffle = False
    
    df = pd.read_csv(csv_path)
    df["file"] = df["file"].apply(lambda x: os.path.join(image_path, x.split("/")[1]))
    
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip, 
        vertical_flip = vertical_flip,
        #rescale = 1.0 / 255
    )
    
    data_generator = imgdatagen.flow_from_dataframe(
        dataframe = df,
        directory = None,
        x_col = "file",
        y_col = target,
        target_size = (size, size),
        batch_size = batch_size,
        save_format = "jpg",
        shuffle = shuffle
    )
    
    with open(mapping_path, "w") as f:
        json.dump(data_generator.class_indices, f)
    f.close()
    
    return data_generator
    

"""
Function to re-organize the dataset
input
    save_path: The new directory to save all the dataset
    train_csv_path, valid_csv_path, train_image_path, valid_image_path are self-explanatory
    target: the category to reorganized, such as age, gender, or raace
    
output will look similar to this(e.g. using gender)
    save_path
        train
            male
                images
                ...
            female
                images
                ...
        validation
            male
                images
                ...
            female
                images
                ...
"""
def create_dataset(save_path, train_csv_path, valid_csv_path, train_image_path, valid_image_path, target):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("created dircetory: {}".format(save_path))
        
    else:
        print("dataset {} already exist!".format(save_path))
        return
        #shutil.rmtree(save_path)

        
    csv_path = [train_csv_path, valid_csv_path]
    image_path = [train_image_path, valid_image_path]
    names = ["train", "validation"]
    
    for i in range(len(csv_path)):
        df = pd.read_csv(csv_path[i])
        df["file"] = df["file"].apply(lambda x: os.path.join(image_path[i], x.split("/")[1]))
        grp_df = df.groupby(target)
        grps = grp_df.groups.keys()
        
        sub_dir = os.path.join(save_path, names[i])
        os.mkdir(sub_dir)
        print("created sub-directory: {}".format(sub_dir))
        
        for grp in grps:
            grp_dir = os.path.join(sub_dir, grp)
            os.mkdir(grp_dir)
            original_file_path = grp_df.get_group(grp)["file"]
            func = lambda x: os.path.join(grp_dir, x.split("/")[-1])
            new_file_path = original_file_path.apply(func).values
            original_file_path = original_file_path.values
            print("created category-directory: {}".format(grp_dir))
            
            for j in range(len(new_file_path)):
                img = PIL.Image.open(original_file_path[j])
                img.save(new_file_path[j])
    
    print("Finished!")

"""
function to visualize the training progress
input
    log_path: The csv file that logged the training progress
    target: the name of the class (e.g. age, race, gender)
    
output
    the accuray and loss curve for both the training and validation
"""
def generate_curves(log_path, save_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
       
    df = pd.read_csv(log_path)
    path_to_viz = save_path
    acc_name = os.path.join(path_to_viz, "acc_curve")
    loss_name = os.path.join(path_to_viz, "loss_curve")
    
    ax = plt.gca()
    plt.plot(df["accuracy"])
    plt.plot(df["val_accuracy"])
    plt.title("Training Accuracy vs. Validation Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    ax.legend(['Train','Validation'],loc='lower right')
    plt.savefig(acc_name)
    plt.close()
    
    ax = plt.gca()
    plt.plot(df["loss"])
    plt.plot(df["val_loss"])
    plt.title("Training loss vs. Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    ax.legend(['Train','Validation'],loc='upper right')
    plt.savefig(loss_name)
    

"""
Function to evaluate the model by calculating category-specific statistics
input
    model: a loaded model
    generator: a generator that contains data
    label_df: the csv file that contains the information of the data
    target: the name of the class (e.g. age, race, gender)
    target_map: The mapping of the class
    save_path: where the plot should be saved
    
output
    The class-specific barplot for precision, recall, f1-score, accuracy, and support
"""
def create_stats(model, generator, target, label_path, mapping_path, save_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    label_df = pd.read_csv(label_path)
    with open(mapping_path) as f:
        target_map = json.load(f)
    f.close()
    
    pred = model.predict(generator).argmax(axis = 1)
    ground_truth = label_df[target].replace(target_map).values
    cr = classification_report(ground_truth, pred, target_names = target_map.keys())
    
    with open(os.path.join(save_path, "class_report.txt"), "w") as f:
        f.write(cr)
    f.close()
    
    cr = classification_report(ground_truth, pred, target_names = target_map.keys(), output_dict = True)
    
    result_df = pd.DataFrame(cr).T.iloc[:len(target_map), :]
    result_df = result_df.reset_index().rename(columns= {"index": "category"})

    cm = confusion_matrix(ground_truth, pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = cm.diagonal()
    result_df["accuracy"] = acc
    result_df.to_csv(os.path.join(save_path, "result_df.csv"), index = False)
    
    stat_names = ["precision", "recall", "f1-score", "accuracy", "support"]
    
    for name in stat_names:
        save_dir = os.path.join(save_path, name + "_barplot")
        plt.figure(figsize = (12,8))
        sns.barplot(x = "category", y= name, data= result_df,linewidth=2.5, 
                    facecolor=(1, 1, 1, 0), edgecolor="0")
        plt.title("{} across {}".format(name, target), fontsize = 20)
        plt.xlabel(target, fontsize = 16)
        plt.ylabel(name, fontsize= 16)
        plt.savefig(save_dir)  

"""
function to create a bounding  box for the detected faces of an image
"""
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

"""
Function to detect face from an image
input
    image_path: The path to the image. The image should include ONLY 1 face to align with the purpose of our web-app
    im_size: The size that the image should be resized
    
output
    an numpy array of an image that has been processed using the resnetv2.preprocess_input
"""
def detect_face(image_path, im_size = 224, default_max_size=800,size = 300, padding = 0.25, to_save = False):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/dlib_mod/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('./models/dlib_mod/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height

    img = dlib.load_rgb_image(image_path)
    old_height, old_width, _ = img.shape
    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)
    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(image_path))
        return
    elif num_faces > 1:
        print("Multiple face in '{}'. A random face will be returned".format(image_path))
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
    image = dlib.get_face_chips(img, faces, size=size, padding = padding)[0]

    image = Image.fromarray(image, 'RGB')
    image = image.resize((im_size, im_size))

    if to_save:
        image.save("./face.png")
    #image = np.array(image) / 255.0
    #ori_img = np.array(image)
    #processed_img = resnet_v2.preprocess_input(np.array(image))
    #processed_img = processed_img[None,:]
    return image

"""
function to get the prediction from the model
input
    img_path: The path to the image
    model_path: The path to the model
    mapping_path: The mapping
    
out
    The prediction made by the model
"""

"""
function to make a single prediction of an image
input
    img_path: The path to the image
    model_path: The path to the model
    mapping_path: The mapping between labels(in number) and categories
    result_df_path: The aggregate results
    
output
    out: The prediction
    pred_prob: The accuracy of making the out prediction
    aggregate_acc: The accuracy of the aggregate category

***NOTE: result_df_path can be found in(assuming race):
    "./visualization/race/stats/result_df.csv"
"""
def get_prediction(img_path, model_path, mapping_path, result_df_path):
    img_arr = detect_face(img_path)
    if img_arr.shape != (1, 224,224,3):
        print("Wrong input size")
        return
    else:
        model = keras.models.load_model(model_path)
        
        with open(mapping_path) as f:
            mapping = json.load(f)
        f.close()
        mapping = {val:key for key, val in mapping.items()}
        pred = model.predict(img_arr).squeeze()
        out = mapping[pred.argmax()]
        pred_prob = np.round(pred[pred.argmax()] * 100, 4)
        
        result_df = pd.read_csv(result_df_path)
        aggregate_acc = np.round(result_df[result_df["category"] == out]["accuracy"].values[0] * 100, 4)
    
    return out, pred_prob, aggregate_acc



""""
Function to create a biased dataset based on the population stats manually added according to the 2020 US Census. The Asian populations are estimated based on 2016 population data.

input
    csv_path: path to the csv file
    save_path: path to save the biased dataset
    
output
    a biased dataset saved in the save_path
"""
def create_biased_dataset(csv_path, save_path):
    df = pd.read_csv(csv_path)
    races = df["race"].unique()
    population = {"White": 0.601, 
                 "Black": 0.134,
                 "Latino_Hispanic": 0.185,
                 "East Asian": 0.022,
                 "Southeast Asian": 0.022,
                 "Indian": 0.012,
                  "Middle Eastern": 0.024
                 }

    total = int(len(df.loc[df.race == 'White'])/population['White'])

    biased_df = pd.DataFrame()
    for race in races:
        num_rows = int(total*population[race])
        # Randomly sample rows based on population
        single_race = df.loc[df.race == race].sample(num_rows)
        if biased_df.empty:
            biased_df = single_race
        else:
            biased_df = pd.concat([biased_df, single_race])
    biased_df = biased_df.sample(frac=1).reset_index(drop=True)
    save_path.to_csv(save_path, index = False)
    


"""
Function to use the integrated_gradient to visualize the image
input 
    model_param_path: saved model in .hdf5 format
    image_path: The image_path
    label_path: The label_path in .csv format
    save_path: path to save the image
    img_idx: The index of the image

output
    original pictures
    annotation heatmaps
"""

def integrated_grad_pic(model_param_path, image_path, label_path, save_path, target, mapping, size, img_idx_lst):
    
    model = keras.models.load_model(model_param_path)
    ig = integrated_gradients(model)
    
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {val:key for key, val in mapping_dict.items()}
    
    max_iter_range = len(mapping_dict)
    if target == "age" or target == "race":
        subplot_row = 3
        subplot_col = 3
    else:
        subplot_row = 1
        subplot_col = 2
    
    for img_idx in img_idx_lst:
        img_name = "{}.jpg".format(img_idx)
        sample_path = os.path.join(image_path, img_name)
        sample_label_df = pd.read_csv(label_path)
        sample_label = sample_label_df[sample_label_df["file"].str.contains(img_name)][target].values[0]

        sample_image = Image.open(sample_path)
        sample_image.save(os.path.join(save_path, "Original_") + str(img_idx)+".png")

        processed_image = resnet_v2.preprocess_input(plt.imread(sample_path)).reshape(-1, size, size, 3)

        exs = []
        output_prob = model.predict(processed_image).squeeze()
        for i in range(1, max_iter_range + 1):
            exs.append(ig.explain(processed_image.squeeze(), outc=i-1))
        exs = np.array(exs)

        # Plot them
        th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

        fig = plt.subplots(subplot_row, subplot_col,figsize=(15,15))
        for i in range(max_iter_range):
            ex = exs[i]
            plt.subplot(subplot_row,subplot_col,i+1)
            plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict[i],output_prob[i]), 
                      fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,"integrated-viz_") + str(img_idx)+".png")
        plt.close()
        print("Ground Truth for {}:".format(img_idx), sample_label)
        print("Predicted for {}:".format(img_idx), mapping_dict[np.argmax(output_prob)])
        
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

"""
functions to load the model with weights

input
    weight_name: weight_name of the checkpoint
output
    the model loaded with weights
"""
def load_model_with_weights(weight_name):
    if "age" in weight_name:
        num_classes = 9
    elif "race" in weight_name:
        num_classes = 7
    else:
        num_classes = 2
    
    model = build_model(num_classes = num_classes)
    model.load_weights(weight_name)
    
    return model
    
"""
Function similar to integrated_grad_pic but just do it on one image

modification: the returned output is an image. This function no longer save the image into jpg.

input 
    model_param_path: saved model in .hdf5 format
    mapping: The dictionary object of the mapping between labels and category
    target: The target(e.g. race, age, gender)
    image_path: The path to the image
    
output
    a single image of object PIL.PngImage
"""
def integrated_grad_pic_single(model_param_path,mapping, target, image_path):
    
    model = keras.models.load_model(model_param_path)
    ig = integrated_gradients(model)
    
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {val:key for key, val in mapping_dict.items()}
    
    max_iter_range = len(mapping_dict)
    if target == "age" or target == "race":
        subplot_row = 3
        subplot_col = 3
    else:
        subplot_row = 1
        subplot_col = 2
    
    
    processed_image = resnet_v2.preprocess_input(plt.imread(image_path)).reshape(-1, size, size, 3)

    exs = []
    output_prob = model.predict(processed_image).squeeze()
    for i in range(1, max_iter_range + 1):
        exs.append(ig.explain(processed_image.squeeze(), outc=i-1))
    exs = np.array(exs)

    # Plot them
    th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

    fig = plt.subplots(subplot_row, subplot_col,figsize=(15,15))
    for i in range(max_iter_range):
        ex = exs[i]
        plt.subplot(subplot_row,subplot_col,i+1)
        plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict[i],output_prob[i]), 
                  fontsize=10)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    img = fig2img(fig)
    return img


"""
Another version of integrated_grad implementation that just shows the heatmap with the highest
predictive accuracy

NOTE: Before running this, Make sure you:
    1. Called Detect_face to crop the image only(WITHOUT USING Resnet Preprocessing)
    2. You should call resnet preprocessing unit INSIDE this function because
       the PIL.fromarray CANNOT take in float32 data type
       
   ALSO: Make sure you'd changed the model path and mapping path so that the function can run.

input
    model_path: The path to the model
    
    mapping: The dictionary that maps the class labels to numeric
    
    PIL_img: a PIL_img object PIL.Image.Image
    
    target: the target(e.g. race, age, gender)
    
    lookup: The particular category to lookup. For instance, given target = race, lookup = None
            would display the heatmap with the highest probability. But if lookup = "white",
            the function would display the heatmap with "white" category even if the category
            does have have the highest probability.
   
output
    a single image of object PIL.PngImagePlugin.PngImageFile
"""
def integrated_grad_PIL(model_path, mapping, PIL_img, lookup = None, to_save = False):    
    
    model = keras.models.load_model(model_path)
    ig = integrated_gradients(model)
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {key.lower():val for key, val in mapping_dict.items()}
    mapping_dict_rev = {val:key for key, val in mapping_dict.items()}
    
    ############################THIS LINE IS IMPORTANT!!!!#################################
    PIL_img = resnet_v2.preprocess_input(np.array(PIL_img)[None, :]) ##IMPORTANT!!!
    output_prob = model.predict(PIL_img).squeeze()
    pred_idx = output_prob.argmax()
    
    if lookup == None:
        pass
    else:
        lookup = lookup.lower()
        pred_idx = mapping_dict[lookup]

    ex = ig.explain(PIL_img.squeeze(), outc=pred_idx)

    th = max(np.abs(np.min(ex)), np.abs(np.max(ex)))

    plt.figure(figsize = (6, 6))
    plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
    plt.title("heatmap for {} with probability {:.2f}".format(mapping_dict_rev[pred_idx],
                                                                 output_prob[pred_idx]), fontsize=12)
    
    if to_save:
        plt.savefig("./ig.png")
    
    fig = plt.gcf()
    im = fig2img(fig)

    return im

"""
Another version of integrated_grad implementation that shows the heatmap with the highest
predictive accuracy(according to the fair model) for BOTH fair and biased models

NOTE: Before running this, Make sure you:
    1. Called Detect_face to crop the image only(WITHOUT USING Resnet Preprocessing)
    2. You should call resnet preprocessing unit INSIDE this function because
       the PIL.fromarray CANNOT take in float32 data type
       
   ALSO: Make sure you'd changed the model path and mapping path so that the function can run.

input
    model_path: The path to the fair model
    
    biased_model_path: The path to the biased model
    
    mapping: The dictionary that maps the class labels to numeric
    
    PIL_img: a PIL_img object PIL.Image.Image
    
    target: the target(e.g. race, age, gender)
    
    lookup: The particular category to lookup. For instance, given target = race, lookup = None
            would display the heatmap with the highest probability. But if lookup = "white",
            the function would display the heatmap with "white" category even if the category
            does have have the highest probability.
   
output
    a single image of object PIL.PngImagePlugin.PngImageFile
"""
def integrated_grad_PIL_v2(model_path, biased_model_path, mapping, PIL_img, target, lookup = None, to_save = False):

    model = keras.models.load_model(model_path)
    model_biased = keras.models.load_model(biased_model_path)
    ig = integrated_gradients(model)
    ig_biased = integrated_gradients(model_biased)

    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()

    mapping_dict = {key.lower():val for key, val in mapping_dict.items()}
    mapping_dict_rev = {val:key for key, val in mapping_dict.items()}
    
    ############################THIS LINE IS IMPORTANT!!!!#################################
    PIL_img = resnet_v2.preprocess_input(np.array(PIL_img)[None, :]) ##IMPORTANT!!!
    
    output_prob = model.predict(PIL_img).squeeze()
    output_prob_biased = model_biased.predict(PIL_img).squeeze()
    pred_idx = output_prob.argmax()
    
    if lookup == None:
        pass
    else:
        lookup = lookup.lower()
        pred_idx = mapping_dict[lookup]

    ex = ig.explain(PIL_img.squeeze(), outc=pred_idx)
    ex_biased = ig_biased.explain(PIL_img.squeeze(), outc=pred_idx)
    
    th = max(np.abs(np.min(np.concatenate([ex, ex_biased]))), np.abs(np.max(np.concatenate([ex, ex_biased]))))

    plt.subplots(1, 2, figsize = (10, 10))
    plt.subplot(1,2,1)
    plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
    plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict_rev[pred_idx],
                                                                 output_prob[pred_idx]), fontsize=12)
    plt.subplot(1,2,2)
    plt.imshow(ex_biased[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
    plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict_rev[pred_idx],
                                                                 output_prob_biased[pred_idx]), fontsize=12)
    
    if to_save:
        plt.savefig("./ig.png")
    
    fig = plt.gcf()
    im = fig2img(fig)

    return im

"""
grad_cam function that converts a PIL image to grad_cam heatmap

input 
    model_path: The path to the model
    
    mapping: The dictionary that maps the class labels to numeric
    
    PIL_img: a PIL_img object PIL.Image.Image
    
    target: the target(e.g. race, age, gender)
    
    lookup: The particular category to lookup. For instance, given target = race, lookup = None
            would display the heatmap with the highest probability. But if lookup = "white",
            the function would display the heatmap with "white" category even if the category
            does have have the highest probability.
    
    to_save: Whether or not to save the heatmap. If true, the heatmaps wil be saved in the current directory.
   
output
     two images of object PIL.PngImagePlugin.PngImageFile: grad_cam and guided_grad_cam
"""
def grad_cam(model_path, mapping, PIL_img, lookup = None, to_save = False):
    #loading models and params
    model = load_model(model_path)
    nb_classes = model.output.shape[1]
    
    #read the mapping
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {key.lower():val for key, val in mapping_dict.items()}
    mapping_dict_rev = {val:key for key, val in mapping_dict.items()}
    
    
    #preprocess image
    PIL_img = np.array(PIL_img).astype("float32")[None, :]
    image = resnet_v2.preprocess_input(PIL_img)
    preprocessed_input = image
    
    #inference 
    if lookup == None:
        output_prob = model.predict(image).squeeze()
        pred_idx = output_prob.argmax()
    else:
        lookup = lookup.lower()
        pred_idx = mapping_dict[lookup]
    
    #grad_cam
    target_layer = lambda x: target_category_loss(x, pred_idx, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers if l.name == "conv2d_7"][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam / 255) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)
    
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', model_path)
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    guided = deprocess_image(gradcam)
 
    if to_save:
        cv2.imwrite("grad_cam" + ".jpg", cam)
        cv2.imwrite("guided_cam" + ".jpg", guided)
        
    cam = Image.fromarray(cam)
    guided = Image.fromarray(guided)
        
    return cam, guided


"""
grad_cam function that converts a PIL image to NORMALIZED grad_cam heatmap

input 
    model_path: The path to the model
    
    biased_model_path: The path to the biased model
    
    mapping: The dictionary that maps the class labels to numeric
    
    PIL_img: a PIL_img object PIL.Image.Image
    
    target: the target(e.g. race, age, gender)
    
    lookup: The particular category to lookup. For instance, given target = race, lookup = None
            would display the heatmap with the highest probability. But if lookup = "white",
            the function would display the heatmap with "white" category even if the category
            does have have the highest probability.
    
    to_save: Whether or not to save the heatmap. If true, the heatmaps will be saved in the current directory.
   
output
     two image of object PIL.PngImagePlugin.PngImageFile: normalized grad_cam from unbiased and biased model
"""
def grad_cam_normalized(model_path, biased_model_path, mapping, PIL_img, target, lookup = None, to_save = False):
     
    model = load_model(model_path)
    biased_model = load_model(model_path_biased)

    nb_classes = model.output.shape[1]
    
    #read the mapping
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {key.lower():val for key, val in mapping_dict.items()}
    mapping_dict_rev = {val:key for key, val in mapping_dict.items()}
    
    
    #preprocess image
    PIL_img = np.array(PIL_img).astype("float32")[None, :]
    image = resnet_v2.preprocess_input(PIL_img)
    preprocessed_input = image
    
    #inference 
    if lookup == None:
        output_prob = model.predict(image).squeeze()
        output_prob_biased = biased_model.predict(image).squeeze()
        pred_idx = output_prob.argmax()
        pred_idx_biased = output_prob_biased.argmax()
    else:
        lookup = lookup.lower()
        pred_idx = mapping_dict[lookup]
        pred_idx_biased = mapping_dict[lookup]
        
    
    #grad_cam
    target_layer = lambda x: target_category_loss(x, pred_idx, nb_classes)
    biased_target_layer = lambda x: target_category_loss(x, pred_idx_biased, nb_classes)
    
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    
    biased_model.add(Lambda(biased_target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    biased_loss = K.sum(biased_model.layers[-1].output)
    
    conv_output =  [l for l in model.layers if l.name == "conv2d_7"][0].output
    biased_conv_output =  [l for l in biased_model.layers if l.name == "conv2d_7"][0].output
    
    
    grads = normalize(K.gradients(loss, conv_output)[0])
    biased_grads = normalize(K.gradients(biased_loss, biased_conv_output)[0])
    
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    biased_gradient_function = K.function([biased_model.layers[0].input], [biased_conv_output, biased_grads])
    

    output, grads_val = gradient_function([image])
    biased_output, biased_grads_val = biased_gradient_function([image])
    
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    biased_output, biased_grads_val = biased_output[0, :], biased_grads_val[0, :, :, :]
    

    weights = np.mean(grads_val, axis = (0, 1))
    biased_weights = np.mean(biased_grads_val, axis = (0, 1))
    
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    biased_cam = np.ones(biased_output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    for i, w in enumerate(biased_weights):
        biased_cam += w * biased_output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    biased_cam = cv2.resize(biased_cam, (224, 224))
    cam = np.maximum(cam, 0)
    biased_cam = np.maximum(biased_cam, 0)

    max_arr = np.concatenate([cam,biased_cam])
    max_norm = np.max(max_arr)

    heatmap = cam / max_norm
    biased_heatmap = biased_cam/ max_norm
    
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    biased_cam = cv2.applyColorMap(np.uint8(255*biased_heatmap), cv2.COLORMAP_JET)

    cam = np.float32(cam / 255) + np.float32(image)
    biased_cam = np.float32(biased_cam / 255) + np.float32(image)
    
    cam = 255 * cam / np.max(cam)
    biased_cam = 255 * biased_cam / np.max(biased_cam)
    
    cam = np.uint8(cam)
    biased_cam = np.uint8(biased_cam)
    
    cam = Image.fromarray(cam)
    biased_cam = Image.fromarray(biased_cam)
    
    return cam, biased_cam

             