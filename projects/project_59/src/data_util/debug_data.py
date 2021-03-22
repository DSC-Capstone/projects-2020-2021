# imports
import os
import random

from PIL import Image
from PIL import UnidentifiedImageError
import PIL

def create_validation_set(DATA_DIR, TRAIN_DIR, VALID_DIR, train):
    '''
    Create validation set from training data.
    
    params:
    train = proportion of desired training data
    '''
    # check if dir exists
    if not os.path.isdir(os.path.join(DATA_DIR, 'valid_snakes_r1')):
        os.mkdir(os.path.join(DATA_DIR, 'valid_snakes_r1'))
        
    # check if dir already has validation data
    if len(os.listdir(VALID_DIR)) > 0:
        print("Already created validation set")
        return
        
    print("Creating validation set at {}".format(VALID_DIR))
    for c in os.listdir(TRAIN_DIR): # get all class folder from train
        # build paths for sampling
        train_dir = os.path.join(TRAIN_DIR, c)
        valid_dir = os.path.join(VALID_DIR, c)

        # make validation dir for each class
        if not os.path.isdir(valid_dir):
            os.mkdir(valid_dir)

        # begin sampling from train for validation set
        data_list = os.listdir(train_dir)
        total_samples = len(data_list)

        random_idx = random.sample(range(total_samples), total_samples)
        train_idx = random_idx[0:int(total_samples*train)]
        valid_idx = random_idx[int(total_samples*train) + 1 : total_samples]

        # move files
        for idx in valid_idx:
            os.rename(os.path.join(train_dir, data_list[idx]), os.path.join(valid_dir, data_list[idx]))
    
    print("Finished creating validation set")
            
def delete_corrupted(DIR):
    '''
    Solves problem of corrupted files crashing model during training, iterates through DIR to find
    any corrupted files and ipynb checkpoints.
    
    Ex file structure:
    data_dir/snake_train_r1/class-X
    
    Would input data_dir/snake_train_r1 to scan within all class-X folders within r1 for unwanted files/dirs
    
    params:
    DIR = directory to scan and delete files from
    '''
    print("---> beginning corrupt file deletion....")
    # keep track of deleted files
    deleted_file_count = 0
    deleted_ipynb = 0
    
    # for each subclass
    for c in os.listdir(DIR):
        cur_dir = os.path.join(DIR, c) # join paths

        for file in os.listdir(cur_dir):
            fp = os.path.join(cur_dir, file)
            try:
                if fp == fp.endswith('.ipynb_checkpoints'):
                    os.rmdir(fp)
                    
                    deleted_ipynb += 1
                    continue
                img = Image.open(fp)
            except PIL.UnidentifiedImageError: # if unidentified file, delete
                os.remove(fp)
                deleted_file_count += 1
            except IsADirectoryError:
                print('ipynb checkpoint found, skipping...')
                
    print('deleted {} corrupted images'.format(deleted_file_count))
    print('deleted {} ipynb checkpoints'.format(deleted_ipynb))