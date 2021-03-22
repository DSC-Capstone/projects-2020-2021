import os
import pandas as pd
import shutil
import cv2
import time

def create_data():
    '''
    Creates data directory if not already existing
    '''
    # first create the data directory
    try:
        directory = "data"
        parent_dir = "./"
        path = os.path.join(parent_dir, directory)
        
        #remove data dir if one already exists
        if (os.path.exists(path) and os.path.isdir(path)):
            shutil.rmtree(path)

        os.mkdir(path)
        
        # create a convenient hierarchical structure of folders inside /data
        directory1 = "viz_steps"
        directory2 = "viz_final"
        parent_dir = "./data/"
        data_link_1=os.path.join(parent_dir, directory1)
        data_link_2=os.path.join(parent_dir, directory2)
        os.mkdir(data_link_1)
        os.mkdir(data_link_2)
        print('Successfully created data folder in the main directory.')
        return 
    except:
        print('Data folder creation failed. Please check parameters and folder directories and re-run python command.')
        return

def resize(file, param1, param2):
    '''
    Given the input parameter (usually images), uses the CV2 package to resize data accordingly to enhance performance
    '''
    try:
        image = cv2.imread(file) # should note that the image gets inverted sometimes through cv2
        image = cv2.resize(image, (param1,param2))
        success_resize = "Image has been successfully resized to dimensions found in clean_file.JSON"
        print(success_resize)
        return success_resize
    except:
        print('Resize function failed. Please check parameters inputted and try again.')
        return
    
def analyze(map_,stepsize,step_dir, out_dir, start_x,start_y,end_x,end_y):
    '''
    Given the algorithm parameter, will run the algorithms on base sample map and compare runtime of each algorithm to determine perfomance.
    Note for the future is to implement feature to compare ground truth difference for analysis to compare performance.
    '''
    try:
        start_alg1 = time.time()
        #print('hi world') # example execution
        os.system("python ./src/generate_rrt_vis.py -start {} {} -stop {} {} -p {} -s {} -step_dir {} -out_dir {}".format(start_x,start_y,end_x,end_y,map_,stepsize,step_dir,out_dir))
        end_alg1 = time.time()
        total_time_alg1 = end_alg1 - start_alg1
        print('The total runtime for algorithm 1 is:', total_time_alg1)
        
        start_alg2 = time.time()
        #print('hello world') # example execution
        os.system("python ./src/generate_a_star_vis.py") # settings default in python file (can change withing)
        end_alg2 = time.time()
        total_time_alg2 = end_alg2 - start_alg2
        print('The total runtime for algorithm 2 is:', total_time_alg2)
        
        perf_dict = {}
        perf_dict['algorithm_1'] = total_time_alg1
        perf_dict['algorithm_2'] = total_time_alg2
        
        better_performer = min(perf_dict, key=perf_dict.get)
        print('Given navigaion algorithms have been successfully tested for performance comparison. Results can be found in the main data directory.')
        return better_performer
    except:
        print('Analyze function could not be completed. Check the algorithm base code to see if there are errors with parameters or other bugs in the code.')
        return

def run_viz(map_,stepsize,step_dir, out_dir, start_x,start_y,end_x,end_y):
    '''
    Creates data directory and run visualization code to be generated
    '''
    # first create the data directory
    directory = "data"
    parent_dir = "./"
    path = os.path.join(parent_dir, directory)
    
    #remove data dir if one already exists
    if (os.path.exists(path) and os.path.isdir(path)):
        shutil.rmtree(path)

    os.mkdir(path)
    
    # create a convenient hierarchical structure of folders inside /data
    directory1 = "viz_steps"
    directory2 = "viz_final"
    parent_dir = "./data/"
    data_link_1=os.path.join(parent_dir, directory1)
    data_link_2=os.path.join(parent_dir, directory2)
    os.mkdir(data_link_1)
    os.mkdir(data_link_2)
    
    try:
        os.system("python ./src/generate_rrt_vis.py -start {} {} -stop {} {} -p {} -s {} -step_dir {} -out_dir {}".format(start_x,start_y,end_x,end_y,map_,stepsize,step_dir,out_dir))
        print('Successfully ran RRT visualization, optimal navigation path can be viewed at: {}'.format(out_dir))
        return
    except:
        print('Error has occured. This occasionally happens with the navigation algorithm node branching process. Please re-run specific python run.py command again.')
        return 

if __name__ == '__main__':
    main()