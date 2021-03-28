import os
import subprocess
import numpy as np
import requests
import sys
from datetime import timedelta, date

"""

Script to download data from https://github.com/echen102/us-pres-elections-2020

Run this script in the command line via the following:
python dl_election_data.py <data_folder>

For example:
python dl_election_data.py ../data/election_data_2020

"""


years = [2020, 2021]
months = np.arange(1, 13)
days = np.arange(1, 32)
file_nums = np.arange(0, 24) # there are 23 txt files per day in the dataset

def make_url(year, month, day, file_num):
    month = "{:02d}".format(month)
    day = "{:02d}".format(day)
    file_num = "{:02d}".format(file_num)
    url = 'https://raw.githubusercontent.com/echen102/us-pres-elections-2020/master/{year}-{month}/us-presidential-tweet-id-{year}-{month}-{day}-{file_num}.txt'.format(year=year, month=month, day=day, file_num=file_num)
    
    return url

def get_response(url):
    requests.get(url)

if __name__ == "__main__":
    data_folder = sys.argv[1]
    for year in years:
        for month in months:
            for day in days:
                for file_num in file_nums:
                    url = make_url(year, month, day, file_num)
                    out_file = '{y}-{m}-{d}_{f}_ids.txt'.format(y=year, m=month, d=day, f=file_num)
                    out_path = os.path.join(data_folder, out_file)
                    
                    curl_cmd = "curl -L {url} -o {out_path}".format(url=url, out_path=out_path)

                    try:
                        subprocess.run(curl_cmd, shell=True)
                        print('ran successfully')
                    except:
                        pass


    print('done!')
