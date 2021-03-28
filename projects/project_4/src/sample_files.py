import os
import subprocess
import numpy as np
import requests
import sys
import datetime
import re

PROP = 0.01

if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    for file in os.listdir(data_dir):
        # randomly sample each file in data dir with proportion
        in_file = os.path.join(data_dir, file)
        out_file = os.path.join(out_dir, file)
        
        cmd = "shuf -n 30000 {infile} --output {out}".format(infile=in_file, out=out_file)
        print('sampling', in_file)
        subprocess.run(cmd, shell=True)
        
print('\n Done! \n')
