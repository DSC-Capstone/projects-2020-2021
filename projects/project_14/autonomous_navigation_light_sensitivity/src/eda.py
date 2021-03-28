# import the necessary packages
import os
import numpy as np
from PIL import Image

def main_eda(datadir, outdir, **kwargs):
    
    os.makedirs(outdir, exist_ok = True)
    # access all .jpg files in the directory.
    allfiles = os.listdir(datadir)
    img_lst = [datadir + file for file in allfiles if file[-4:] in ['.jpg']]
    
    # Given that all images are the same size, get the dimensions of the first image.
    width, height = Image.open(img_lst[0]).size
    N = len(img_lst)
    
    # Create an array of floats to store the average.
    arr = np.zeros((height, width, 3), np.float)
    
    # Calculate average pixel intensities, casting each image as an array of floats.
    for i in img_lst:
        img_arr = np.array(Image.open(i), dtype=np.float)
        arr = arr+img_arr/N
    
    # Round values in an array and cast as 8-bit integer.
    arr = np.array(np.round(arr), dtype=np.uint8)
    
    # Generate and save the final merged image (average pixel image of all images in a track lap data).
    merged = Image.fromarray(arr, mode="RGB")
    merged.save(outdir+"average_pixel_picture.jpg")
    
    return
 
