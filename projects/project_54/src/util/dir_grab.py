'''
Misc functions 
'''
import os
from pathlib import Path

def get_directory(dataset, root="./data/hierarchies"):
    return os.path.join(root, dataset)

def makeparentdirs(path):
    dir = Path(path).parent
    os.makedirs(dir, exist_ok=True)