import os
import shutil
import bagpy
import rosbag
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import math

from .conversion import convert
from .viz_analysis import plot_all

def test(conv_cfg, plot_cfg):
    
    convert(**conv_cfg)
    plot_all(**plot_cfg)

    
if __name__ == '__main__':
    main()
