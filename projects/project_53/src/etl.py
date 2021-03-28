from src import gradcam
from src import input_paths as data
from PIL import Image 

covered_path = data.covered_path


def get_id(path):
    data_path = '/datasets/MaskedFace-Net/train'
    categories = os.listdir(data_path)
    labels = [i for i in range(len(categories))]
    label_dict = dict(zip(categories,labels))
    print(label_dict)
    print(categories)
    print(labels)
    return None

def print_image(id_path):
    pil_im = Image.open(id_path)
    print(pil_im)
    
def gcam(model):
    gradcam.run_grad(model)
    return

def gcam_train(model):
    gradcam.run_grad_train(model)
    return


import glob
def stats():
        print("Printing information about the data")
        def get_incorrect(file_type,d):
            img_desc = []
            tifCounter = len(glob.glob1(file_type,d))
            split = file_type.split("/")
           # print(split)
            print("There are " + str(tifCounter) + " images of " + d + " in the " + split[3] + " folder" )
            img_desc.append(tifCounter)
            img_desc.append(d[6:-4].replace("_", " "))
            img_desc.append(split[3])
            return img_desc
    
        
        
        all_types = ["*Mask_Mouth_Chin.jpg", "*Mask_Nose_Mouth.jpg", "*Mask_Chin.jpg", "*Mask_Nose_Mouth.jpg"]
        incorrect_folders = [data.tpic,data.hopic,data.vpic]
        image_stats = []
        for d in all_types:
            for i in incorrect_folders:
                x = get_incorrect(i,d)
                image_stats.append(x)
        print(image_stats)
    
       


    
    
    
