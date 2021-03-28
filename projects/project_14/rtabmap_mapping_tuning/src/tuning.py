import numpy as np
import os
import subprocess

def find_metrics(datadir, plotsdir, metricdir, metricfile, evaldir):
    """
    >>> os.path.isfile("results/metrics/metrics.txt")
    True
    """
    os.makedirs(plotsdir, exist_ok = True)
    os.makedirs(metricdir, exist_ok = True)
    if os.path.isfile(metricdir+metricfile):
        subprocess.call(["rm", metricdir + metricfile])
    subprocess.call(["touch", metricdir + metricfile])
    
    for file in list(filter(lambda x: x.endswith('.yaml'), os.listdir(datadir))):
        file = file[:-5]
        ate = subprocess.run(["python", evaldir + "evaluate_ate.py", datadir + file + '_slam.txt', datadir + file + '_gt.txt', "--plot", plotsdir + file + "_ate.png"], stdout=subprocess.PIPE, text = True) 
        rpe = subprocess.run(["python", evaldir + "evaluate_rpe.py", datadir + file + '_odom.txt', datadir + file + '_gt.txt', "--plot", plotsdir + file + "_rpe.png", "--fixed_delta"], stdout=subprocess.PIPE, text = True) 
        file1 = open(metricdir + metricfile, "a")
        file1.write("{}, {}, {}\n".format(file, ate.stdout.strip(), rpe.stdout.strip())) 
        file1.close() 
