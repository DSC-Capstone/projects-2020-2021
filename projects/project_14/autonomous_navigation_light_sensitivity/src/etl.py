import os
import shutil

def load_data(single_source, lap_source, single_data_fp, lap_data_fp):
    #Creates the directory to copy data into
    os.makedirs(single_data_fp, exist_ok = True)
    #os.makedirs(lap_data_fp, exist_ok = True)
    
    print("Data in my single image data folder:")
    print(os.listdir(single_source))
    
    print("Data in my collective lap data folder:")
    print(os.listdir(lap_source))
    
    for i in os.listdir(single_source):
        shutil.copy(single_source + i, single_data_fp)
    
    for j in os.listdir(lap_source):
        os.makedirs(lap_data_fp + j + '/', exist_ok = True)
        for k in os.listdir(lap_source + j):
            shutil.copy(lap_source + j + '/' + k, lap_data_fp + j + '/')
            
    print('After loading single image data: ' + str(os.listdir(single_data_fp)))
    print('After loading collective lap data: ' + str(os.listdir(lap_data_fp)))
