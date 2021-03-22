from PIL import Image
import os
import pandas as pd
import seaborn as sns
import glob

def load_color_palette(color_list):
    # Set your custom color palette
    sns.set_palette(sns.color_palette(color_list))
    
def load_paths():
    tpc = '/datasets/MaskedFace-Net/train/covered' #train path covered
    tpic = '/datasets/MaskedFace-Net/train/incorrect' #train path incorrect
    tpu = '/datasets/MaskedFace-Net/train/uncovered' #train path uncovered

    hopc = '/datasets/MaskedFace-Net/holdout/covered' #holdout path covered
    hopic = '/datasets/MaskedFace-Net/holdout/incorrect' #holdout path incorrect
    hopu = '/datasets/MaskedFace-Net/holdout/uncovered' #holdout path uncovered

    vpc = '/datasets/MaskedFace-Net/validation/covered' #validation path covered
    vpic = '/datasets/MaskedFace-Net/validation/incorrect' #validation path incorrect
    vpu = '/datasets/MaskedFace-Net/validation/uncovered' #validation path uncovered
    return [tpc, tpic, tpu, hopc, hopic, hopu, vpc, vpic, vpu]

def get_incorrect_folders():
    tpic = '/datasets/MaskedFace-Net/train/incorrect'
    hopic = '/datasets/MaskedFace-Net/holdout/incorrect'
    vpic = '/datasets/MaskedFace-Net/validation/incorrect'
    return [tpic, hopic, vpic]

def count_images_print(d):
    image_desc = []
    image_count = []
    count = 0
    for path in os.listdir(d):
        if os.path.isfile(os.path.join(d, path)):
            count += 1
        image_count.append(count)
    split = d.split("/")
    print("There are " + str(count) + " images of " + split[4]+ " faces in the " + split[3] + " folder.")
    image_desc.append(count)
    image_desc.append(split[4])
    image_desc.append(split[3])
    return image_desc

def get_incorrect_print(file_type,d):
    img_desc = []
    tifCounter = len(glob.glob1(file_type,d))
    split = file_type.split("/")
   # print(split)
    print("There are " + str(tifCounter) + " images of " + d + " in the " + split[3] + " folder" )
    img_desc.append(tifCounter)
    img_desc.append(d[6:-4].replace("_", " "))
    img_desc.append(split[3])
    return img_desc

def get_incorrect(file_type,d):
    img_desc = []
    tifCounter = len(glob.glob1(file_type,d))
    split = file_type.split("/")
    img_desc.append(tifCounter)
    img_desc.append(d[6:-4].replace("_", " "))
    img_desc.append(split[3])
    return img_desc

def visualize_image_split(file_path):
    image_data = []
    for d in file_path:
        x = count_images_print(d)
        image_data.append(x)
    image_data = pd.DataFrame(image_data)
    image_data.columns = ['count', 'mask category', 'dataset']
    sns.barplot(x = 'mask category', y = 'count', hue = 'dataset', data = image_data).set_title("Mask Usage Count split via dataset")
    
    
def count_images(d):
    image_desc = []
    image_count = []
    count = 0
    for path in os.listdir(d):
        if os.path.isfile(os.path.join(d, path)):
            count += 1
        image_count.append(count)
    split = d.split("/")
    image_desc.append(count)
    image_desc.append(split[4])
    image_desc.append(split[3])
    return image_desc

def visualize_image(file_path):
    image_data = []
    for d in file_path:
        x = count_images(d)
        image_data.append(x)
    image_data = pd.DataFrame(image_data)
        
    image_data.columns = ['count', 'mask category', 'dataset']
    image_data_gb = image_data.groupby(['mask category']).sum()
    
    for i in range(len(image_data_gb['count'])):
        print("There are " + str(image_data_gb['count'][i]) + " images of " + image_data_gb.index[i] + " faces in the folder.")
    sns.barplot(x = 'mask category', y = 'count', data = image_data).set_title("Mask Usage Count")
    
    
def visualize_stats_split(incorrect_folders):
    all_types = ["*Mask_Mouth_Chin.jpg", "*Mask_Nose_Mouth.jpg", "*Mask_Chin.jpg"]
    image_stats = []
    for d in all_types:
        for i in incorrect_folders:
            x = get_incorrect_print(i,d)
            image_stats.append(x)
    image_stats = pd.DataFrame(image_stats)
    image_stats.columns = ["count", "Reason", "dataset"]
    sns.barplot(x = 'Reason', y = 'count', hue = 'dataset', data = image_stats).set_title("Count of Improper Face Mask Usage split via dataset")
    
    
def visualize_stats(incorrect_folders):
    all_types = ["*Mask_Mouth_Chin.jpg", "*Mask_Nose_Mouth.jpg", "*Mask_Chin.jpg"]
    image_stats = []
    for d in all_types:
        for i in incorrect_folders:
            x = get_incorrect(i,d)
            image_stats.append(x)
    image_stats = pd.DataFrame(image_stats)
    image_stats.columns = ["count", "Reason", "dataset"]
    isg = image_stats.groupby(['Reason']).sum()
    
    for i in range(len(isg['count'])):
        print("There are " + str(isg['count'][i]) + " images of " + isg.index[i] + " faces in the folder.")
    sns.barplot(x = 'Reason', y = 'count', data = image_stats).set_title("Count of Improper Face Mask Usage")