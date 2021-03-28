# import custom modules
import sys
sys.path.insert(0, "src/util")
sys.path.insert(0, "src/data_util")

# imports
import torch
import torchvision

import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
import os

from nbdt.model import SoftNBDT
from nbdt.model import HardNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.loss import HardTreeSupLoss

from write_to_json import *

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    '''
    trains model by using train and validation sets
    '''
    # define lists
    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    
    loss_train_evo=[]
    acc_train_evo=[]
    fs_train_evo=[]
    
    loss_val_evo=[]
    acc_val_evo=[]
    fs_val_evo=[] 
    
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('---> Begin model training...')
    for epoch in range(num_epochs):
        i = 0
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # determine if in train or validation phase
        for phase in ['train_snakes_r1', 'valid_snakes_r1']:
            if phase == 'train_snakes_r1':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            fscore = []

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients before beginning backprop
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train_snakes_r1'):
                    # calculate loss from model outputs
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train_snakes_r1':
                        loss.backward()
                        optimizer.step()

                # statistics
                labels_cpu = labels.cpu().numpy()
                predictions_cpu = preds.cpu().numpy()
                Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
                fscore.append(Fscore)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_fscore = np.average(np.array(fscore))
            
            print('{} Loss: {:.4f} Acc: {:.4f} F: {:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_fscore))
            
            if phase == 'train_snakes_r1':
                loss_train_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_train_evo.append(epoch_acc)
                fs_train_evo.append(epoch_fscore)                
            else:
                loss_val_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_val_evo.append(epoch_acc)
                fs_val_evo.append(epoch_fscore) 
                
            # deep copy the model
            if phase == 'valid_snakes_r1' and epoch_fscore > best_fscore:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    print("---> Finished model training.")
    
    return model, loss_train_evo, acc_train_evo, fs_train_evo, loss_val_evo, acc_val_evo, fs_val_evo

def set_parameter_requires_grad(model, feature_extracting):
    '''
    sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    '''
    initializes pretrained vgg model
    '''
    print("---> Begin model initialization...")
    ft_extract = False
    if feature_extract == "True":
        ft_extract=True

    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, ft_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    print("---> Finished model initialization.")
    
    return model_ft, input_size

def create_dataloaders(DATA_DIR, batch_size, input_size):
    '''
    return model transformations for training and validation sets
    '''
    print("---> Begin dataloader creation...")
    
    data_transforms = {
            'train_snakes_r1': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([0, 90]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid_snakes_r1': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([0, 90]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }   
    
    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in [
            'train_snakes_r1',
            'valid_snakes_r1']   
    }
    
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        ) for x in [
            'train_snakes_r1',
            'valid_snakes_r1'
        ]
    }
    print("---> Finished creating dataloaders.")
    
    return dataloaders_dict, len(image_datasets['train_snakes_r1'].classes)

def params_to_update(model_ft, feature_extract):
    '''
    defines params to update for optimizer, based on feature extract
    '''
    ft_extract = False
    if feature_extract == "True":
        ft_extract=True
        
    params_to_update = model_ft.parameters()
    if ft_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                a=1 # print("\t",name)
                
    return params_to_update

def run_model(data_cfg, model_cfg, criterion):
    '''
    Runs model based on parameters from data_cfg and model_cfg. Additionally, writes best model's weights to a path in config
    '''
    
    # create dataloaders
    dataloaders_dict, num_classes = create_dataloaders(
        data_cfg['dataDir'],
        model_cfg['batchSize'],
        model_cfg['inputSize']
    )
    
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_cfg['modelName'],
        num_classes,
        feature_extract = model_cfg['featureExtract'],
        use_pretrained=True
    )

    model_ft = model_ft.to(device) # make model use GPU

    params_update = params_to_update(model_ft, model_cfg['featureExtract'])

    # Optimizer
    optimizer_ft = optim.Adam(params_update, lr=model_cfg['lr'])
    
    # train model
    model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer = optimizer_ft,
        num_epochs = model_cfg['nEpochs'])
    
    # save model
    save_model(model_ft, data_cfg, model_cfg)
    
    return model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val 
    
def save_model(model_ft, data_cfg, model_cfg):
    '''
    saves weights of a passed model
    '''
    # save model to model states in params
    now = datetime.now().strftime("%d%m%Y_%H:%M")
    model_path = os.path.join(data_cfg['dataDir'], "model_states")
    model_name = os.path.join(
        model_path, 
        "{}_{}_{}.pth".format(
            now,
            model_cfg['nEpochs'],
            model_cfg['modelName']
        )
    )
    if not os.path.isdir(model_path): # make sure model path is made
        print("---> Creating {}".format(model_path))
        os.mkdir(model_path)
        
    # saves model
    print('---> saving model at {}/{}'.format(model_path, model_name))
    torch.save(model_ft.state_dict(), model_name)
    
    
def run_nbdt(data_cfg, model_cfg, loss_type):
    '''
    Runs nbdt 
    '''
    # check to make sure loss_type is specified
    assert (
        loss_type in ["SoftTreeSupLoss", "HardTreeSupLoss"]
    ), "Please specify SoftTreeSupLoss or HardTreeSupLoss"
    
    # create dataloaders
    dataloaders_dict, num_classes = create_dataloaders(
        data_cfg['dataDir'],
        model_cfg['batchSize'],
        model_cfg['inputSize']
    )
    
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model for this run
    model, input_size = initialize_model(
        model_cfg['modelName'],
        num_classes,
        feature_extract = model_cfg['featureExtract'],
        use_pretrained=True
    ) 
    
    # load model weights
    if loss_type == "SoftTreeSupLoss":
        model_weights = torch.load(data_cfg['hierarchyModelPath'])
    elif loss_type == "HardTreeSupLoss":
        model_weights = torch.load(data_cfg['hierarchyModelPath'])
        
    model = model.to(device) # make model use GPU
    model.load_state_dict(model_weights)
    
    # create NBDT
    print('---> Creating NBDT...')
    if loss_type == "SoftTreeSupLoss":
        model = SoftNBDT(
            model = model,
            dataset = 'snakes', 
            hierarchy='induced-densenet121',
            path_graph = os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
            path_wnids =  data_cfg['wnidPath']
        )
    else:
        model = HardNBDT(
            model = model,
            dataset = 'snakes', 
            hierarchy='induced-densenet121',
            path_graph = os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
            path_wnids =  data_cfg['wnidPath']
        )
    print('---> Finished creating NBDT.')
    
    model.eval()

    running_corrects = 0
    fscore = []

    print('---> Running inference...')
    # iterate over data
    for inputs, labels in dataloaders_dict['valid_snakes_r1']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        labels_cpu = labels.cpu().numpy()
        predictions_cpu = preds.cpu().numpy()
        Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
        fscore.append(Fscore)
        running_corrects += torch.sum(preds == labels.data)
    print('---> Finished running inference...')
    
    epoch_acc = running_corrects.double() / len(dataloaders_dict['valid_snakes_r1'].dataset)
    epoch_fscore = np.average(np.array(fscore))
    
    print(" ")
    print('{} Acc: {:.4f} F1: {:.4f}'.format('NBDT Test', epoch_acc, epoch_fscore))
    print(" ")