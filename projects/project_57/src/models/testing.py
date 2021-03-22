from __future__ import print_function
from __future__ import division
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from torchvision.utils import save_image


from training import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--custom-image-path', default=None,
                        help='Use custom image')
    parser.add_argument('--test-size', type=int, default=1,
                        help='Size of the dataset that you want to test your model on')
    parser.add_argument('--model-name', type=str, default="resnet",
                        help='Model name')
    parser.add_argument('--model-path', type=str, default=os.path.join('../DSC180B-Face-Mask-Detection/models', 'model_resnet_best_val_acc_0.955.pt'),
                        help='Load model path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    args = get_args()
    # Top level data directory. Here we assume the format of the directory conforms
    # to the ImageFolder structure
    data_dir = "/datasets" + "/MaskedFace-Net"
    # Image path
    custom_image_path = args.custom_image_path
    # Model path
    model_path = args.model_path
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model_name
    # Number of classes in the dataset
    num_classes = 3
    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch_size
    # Size of val and test dataset
    test_size = args.test_size
  
    #New test case
    #Go to the correct file path

    #Accuracy on the validation set with a batch size of 4
    #Load in the saved training model

    #Change filepath back to where the dataset is
    #os.chdir("/datasets" + "/MaskedFace-Net")
    input_size = 0
    if model_name == "inception":
        input_size = 299
    else:
        input_size = 224

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'holdout': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'custom': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation', 'holdout']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation', 'holdout']}
        
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    
    
    # Use custom image
    if custom_image_path is not None:
        img = cv2.imread(custom_image_path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("invalid custom image path, please check your parameter")
            sys.exit(0)
        im_pil = Image.fromarray(img)
        transformed_im_pil = data_transforms['custom'](im_pil)
        print(transformed_im_pil.shape)
        custom_input = transformed_im_pil.unsqueeze(0)
        try:
            model = torch.load(model_path)
        except:
            print("invalid model path, please check your parameter again")
            sys.exit(0)
        model = model.to(device)
        custom_input = custom_input.to(device)
        classes = ('correctly masked', 'incorrectly masked', 'not masked')
        custom_output = model(custom_input)
        _, custom_predicted = torch.max(custom_output, 1)
        custom_results = ' | '.join('%5s' % classes[custom_predicted[j]] for j in range(1))
        print('Model Prediction on custom set: ', ' | '.join('%5s' % classes[custom_predicted[j]] for j in range(1)))
        save_image(custom_input[0].cpu(),'results/model_prediction/' + "custom_img_{0}_prediction_{1}.jpg".format(random.randint(1,10000), custom_results))
        sys.exit(0)
        
    # Use MaskedFace-Net 
    maskedface_net_val = image_datasets['validation']
    val_sub = torch.utils.data.Subset(maskedface_net_val, np.random.choice(len(maskedface_net_val), 1, replace=False))
    data_loader_val_sub = torch.utils.data.DataLoader(val_sub,
                                                      batch_size=batch_size, 
                                                      shuffle=True)
    
    maskedface_net_test = image_datasets['holdout']   
    test_sub = torch.utils.data.Subset(maskedface_net_test, np.random.choice(len(maskedface_net_test), 1, replace=False))
    data_loader_test_sub = torch.utils.data.DataLoader(test_sub,
                                                      batch_size=batch_size, 
                                                      shuffle=True)
    

    
    try:
        if args.use_cuda:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location="cpu")
    except:
        print("invalid model path, please check your parameter again")
        sys.exit(0)
    
    # Send the model to GPU
    model = model.to(device)
    #Accuracy section
    val_correct = 0
    val_total = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in data_loader_val_sub:
            val_inputs = val_inputs.to(device)
            val_labels = torch.tensor(np.array(val_labels).astype(int)) # to delete
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            _, predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (predicted == val_labels).sum().item()
        for test_inputs, test_labels in data_loader_test_sub:
            test_inputs = test_inputs.to(device)
            test_labels = torch.tensor(np.array(test_labels).astype(int)) # to delete
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (predicted == test_labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * val_correct / val_total))
    print('Accuracy of the network on the test set: %d %%' % (100 * test_correct / test_total))
    
    classes = ('correctly masked', 'incorrectly masked', 'not masked')
    
    _, val_predicted = torch.max(val_outputs, 1)
    _, test_predicted = torch.max(test_outputs, 1)
    
    val_results = ' | '.join('%5s' % classes[val_predicted[j]] for j in range(1))
    test_results = ' | '.join('%5s' % classes[test_predicted[j]] for j in range(1))
    
    print('Model Prediction on validation set: ', ' | '.join('%5s' % classes[val_predicted[j]] for j in range(1)))
    print('Model Prediction on test set: ', ' | '.join('%5s' % classes[test_predicted[j]] for j in range(1)))
    
    print('Please visit /results/model_prediction for the image')
    
    save_image(val_inputs[0].cpu(),'results/model_prediction/' + "val_img_{0}_prediction_{1}.jpg".format(random.randint(1,len(maskedface_net_val)), val_results))
    
    save_image(test_inputs[0].cpu(),'results/model_prediction/' + "test_img_{0}_prediction_{1}.jpg".format(random.randint(1,len(maskedface_net_test)), test_results))
    
   