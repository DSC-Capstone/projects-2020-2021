import os
import subprocess
import numpy as np
import requests
import sys
import datetime
import re

# specifying date ranges
months = [7, 8, 9, 10, 11]
days = np.arange(1,32)
nums = np.arange(0, 24)
# only choosing year 2020
file_format = '2020-{month}-{day}_{num}_ids.txt'

if __name__ == "__main__":
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    for month in months:
        for day in days:
            try:
                num = np.random.choice(nums)
                txt_file = file_format.format(month=month, day=day, num=num)
                txt_file = os.path.join(data_dir, txt_file)
                cmd = "mv {file} {outdir}".format(file=txt_file, outdir=out_dir)
                subprocess.run(cmd, shell=True)
            except:
                pass
print('done moving all files that match criteria')
