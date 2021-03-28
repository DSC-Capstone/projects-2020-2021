import os
import subprocess
import glob
import numpy as np

def move_data(rawdir, posedir):
    """
    >>> posedir = "data/poses/"
    >>> os.path.isfile(posedir+'param1_gt.txt')
    True
    >>> os.path.isfile(posedir+'param2_gt.txt')
    True
    >>> os.path.isfile(posedir+'param1_slam.txt')
    True
    >>> os.path.isfile(posedir+'param1_odom.txt')
    True
    """
    for file in list(filter(lambda x: x.endswith('.yaml'), os.listdir(rawdir))):
        os.makedirs(posedir, exist_ok = True)
        file = file[:-5]
        mv_dir_gt = subprocess.call(["cp", os.path.join(rawdir + '/' + file + '_gt.txt'), posedir])
        mv_dir_slam = subprocess.call(["cp", os.path.join(rawdir + '/' + file + '_slam.txt'), posedir])
        mv_dir_odom = subprocess.call(["cp", os.path.join(rawdir + '/' + file + '_odom.txt'), posedir])
        mv_dir_yaml = subprocess.call(["cp", os.path.join(rawdir + '/' + file + '.yaml'), posedir])
