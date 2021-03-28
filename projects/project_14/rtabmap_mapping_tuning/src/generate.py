import pandas as pd
import subprocess
import os

def create_launch_files(datafile, paramdir, launchfile, outparamdir, outlaunchdir):
    """
    >>> os.path.isfile("results/mapping/params/best_params.yaml")
    True
    >>> os.path.isfile("results/mapping/launch/mapping.launch")
    True
    """
    data = pd.read_csv(datafile, header = None, names = ['param_file', 'ate', 'rpe']).set_index('param_file')
    best = data.sort_values(by=['ate', 'rpe']).index[0]
    os.makedirs(outparamdir[:-16], exist_ok = True)
    os.makedirs(outlaunchdir[:-14], exist_ok = True)
    subprocess.call(["cp", paramdir + "/" + best + ".yaml", outparamdir])
    subprocess.call(["cp", launchfile, outlaunchdir])
