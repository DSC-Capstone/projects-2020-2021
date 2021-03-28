import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from . import transforms as transforms_custom
from torch.utils.data import Dataset
from pathlib import Path
import zipfile
import urllib.request
import shutil
import time

# add snakes to __all__
__all__ = names = (
    "snakes",
)

class snakes(Dataset):
    '''
    Snake dataset
    '''
    
    dataset = None
    
    def __init__(self, root="../teams/DSC180A_FA20_A00/a01group09/", *args, train=True, download=False, **kwargs):
        super().__init__()
        
        dataset = _snakesTrain if train else _snakesVal
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)} # convert to idx
        
    @staticmethod
    def transform_train(input_size = 224):
        '''
        training transformations
        '''
        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ]
        )
    
    @staticmethod
    def transform_val(input_size = 224):
        '''
        validation transformations (same as training)
        '''
        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ]
        )
    
    def __getitem__(self, i):
        return self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)

# establish separate directories for training vs validation sets
class _snakesTrain(datasets.ImageFolder):
    def __init__(self, root="../teams/DSC180A_FA20_A00/a01group09/", *args, **kwargs):
        super().__init__(os.path.join(root, "train_snakes_r1"), *args, **kwargs)
        
class _snakesTrain(datasets.ImageFolder):
    def __init__(self, root="../teams/DSC180A_FA20_A00/a01group09/", *args, **kwargs):
        super().__init__(os.path.join(root, "train_snakes_r1"), *args, **kwargs)
        