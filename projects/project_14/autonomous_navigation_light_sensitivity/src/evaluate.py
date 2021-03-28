import numpy as np
import os
import json
import math
from PIL import Image, ImageStat

def runtime_performance_eval(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    default = runtime_performance(indir[0])
    tuned = runtime_performance(indir[1])
    similarity = np.round(1 - abs(default-tuned)/default , 3) * 100
    f = open(outdir + "runtime_evaluation_result.txt", "w+")
    f.write("Runtime performance evaluation (consistency of luminescence of recorded images over the duration of one lap run) \n\n")
    f.write("Evaluated by the standard deviation of the perceived brightness formula across every images recorded over time. \n\n")
    f.write("Runtime performance for default, non-bright conditions: %s" % default)
    f.write("\n\n")
    f.write("Runtime performance for tuned configuration under bright conditions: %s" % tuned)
    f.write("\n\n")
    f.write("Similarity in runtime performance between default and best-tuned configurations: %s" % similarity)
    f.write(" %")
    f.close()
    
    return
    

# measures how consistent the luminescence of mobile image data is across one track lap's time, using standard deviation
# as a metric of variability
def runtime_performance(indir):
    
    store_perceived = []
    for i in range(len(os.listdir(indir))):
        store_perceived.append(perceived_bn(indir, i))
    
    runtime_performance = np.std(store_perceived)
    rounded = np.round(runtime_performance, 3)
    
    return rounded

# function to calculate perceived brightness which effectively measures luminescence of images using rgb color codes
def perceived_bn(indir, jpg):
    
    converted_num = "% s" % jpg
    if (jpg < 10):
        jpg_string = indir + '/frame000' + converted_num + '.jpg'
    else:
        jpg_string = indir + '/frame00' + converted_num + '.jpg'
    img = Image.open(jpg_string)
    stat = ImageStat.Stat(img)
    r,g,b = stat.rms
    
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
