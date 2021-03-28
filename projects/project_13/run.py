import sys
import json
import os

sys.path.insert(0, 'src')

from src.generate_rrt_vis import *
from src.util import *


points_cfg_test = json.load(open('config/rrt_vis_points_test.json'))
points_cfg_all = json.load(open('config/rrt_vis_points_all.json'))
mapping_cfg_test = json.load(open('config/rrt_vis_mapping_test.json'))
mapping_cfg_all = json.load(open('config/rrt_vis_mapping_all.json'))
clean_file = json.load(open('config/clean_file.json'))



def main(targets):
    if 'test' in targets: 
        run_viz(**mapping_cfg_test,**points_cfg_test)
        #print('Successfully ran RRT visualization on maze.png, optimal navigation path can be viewed {}'.format(mapping_cfg_test['out_dir']))
        
    if 'data' in targets:
        create_data()
        #print('Successfully created data folder in the main directory')
        
    if 'clean' in targets:
        resize(**clean_file)
        #print('Files inputted sucessfully cleaned. Should now be ready for analysis, visualization to be loaded on interactive interface')
        
    if 'analyze' in targets:
        analyze(**mapping_cfg_all,**points_cfg_all)
        #print('Given navigaion algorithms have been successfully tested for performance comparison. Results can be found in the main data directory')
        
    if 'all' in targets:
        run_viz(**mapping_cfg_all,**points_cfg_all)
        #print('Successfully ran RRT visualization on thunderhill_cropped.png, optimal navigation path can be viewed {}'.format(mapping_cfg_all['out_dir']))



if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)