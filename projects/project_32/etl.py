
import numpy as np
import pandas as pd
import math
from scipy import stats
import os
import json


def get_data(input_dir, output_dir):
    data_files = os.listdir(input_dir)

    if ('default.json' in data_files):
        # setup variables

        intake_params = pd.read_json(input_dir)# + '\params.json')
