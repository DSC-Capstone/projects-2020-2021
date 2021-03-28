import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import warnings
import json
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0,'src\lib')

from time import time
from src.webscraper import * 
from src.model import *
from src.FCN import *
from src.featureSpaceDay import *
from src.generateCorr import *

seed = 234
np.random.seed(seed)

dow = "./data/dowJonesData";
spy = "./data/SP500Data";
parser = argparse.ArgumentParser()
parser.add_argument('flag',nargs = '+')

t = time()
args = parser.parse_args()
with open('./config/model-params.json') as f:
    p = json.loads(f.read())
    dataset = p['dataset']
    threshold = p['thresh']
    timeframe = p['timeframe']

if 'test' in args.flag:
    print('Running the Model');
    modelRun()

elif 'all' in args.flag:
    print('Downloading the Data for The Dow Jones and The S&P500')
    getData(timeframe)
    print('Generating the Correlation graphs');
    buildCorrGraph(dow,threshold);
    buildCorrGraph(spy,threshold);
    print('Running the Model');
    modelRun()

elif 'fcn' in args.flag:
    print('Running the Model');
    fcnRun()

elif 'build' in args.flag:
    print('Downloading the Data for The Dow Jones and The S&P500')
    getData(timeframe)
    print('Generating the Correlation graphs');
    buildCorrGraph(dow,threshold);
    buildCorrGraph(spy,threshold);

else:
    print('Downloading the Data for The Dow Jones and The S&P500')
    getData(timeframe)
    print('Generating the Correlation graphs');
    buildCorrGraph(dow,threshold);
    buildCorrGraph(spy,threshold);
    print('Running the Model');
    modelRun()


    #then run the data through the model

print('time used: %d s' % (time() - t))
