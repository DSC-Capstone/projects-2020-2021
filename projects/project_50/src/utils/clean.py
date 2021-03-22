import os
import shutil

def remove_data():
    """
    Removes files specified in locations variable
    """
    locations = ['data']
    for l in locations:
        if os.path.exists(l):
            shutil.rmtree(l)
