from PIL import Image

import sys
import os
import json
import numpy as np
import cv2
import torch


from src import input_paths as data
from src import etl as e

data_path = '/datasets/MaskedFace-Net/holdout'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories,labels))

test_params = 'test/testdata/test-params.json'


print(label_dict)
print(categories)

#Function to deal with image output path
def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param

sys.path.insert(0, 'src') # add library code to path

def main(targets):
    if 'test' in targets:
        model_path = 'Config/inceptionResnetV1.pth'
        model = torch.load('Config/inceptionResnetV1.pth',map_location=torch.device('cpu'))
        #Looks at etl file for information
        e.stats()
        p = load_params(test_params)
        #perform etl
        e.gcam(model)
        print("Done")
        
    if 'run_grad' in targets:
        #Function to deal with input image paths
        path = data.covered_path
        model_path = 'Config/inceptionResnetV1.pth'
        model = torch.load('Config/inceptionResnetV1.pth',map_location=torch.device('cpu'))
        e.stats()
        p = load_params(test_params)
        #perform etl
        e.gcam_train(model)
        print("Done")

        
#first call to start data pipeline
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
