import pandas as pd
import numpy as np
import os
import nbformat
from nbconvert import HTMLExporter

def list_flatten(nested_list):
    """
    flatten nested list 
    """
    result = []
    for l in nested_list:
        if not isinstance(l, list):
            result.append(l)
        else:
            result.extend(list_flatten(l))
    return result 

def cantor_pairing(values):
    """
    Using cantor_paring to uniquely encode two/multiple ids into one id 

    values: input numpy array must has length larger than two 
    return: return a numpy array with unique ids 
    """
    
    # base case 
    if values.shape[0] == 2:
        a, b = values[0], values[1]
        return 1/2 * (a+b) * (a+b+1) + b
    
    if values.shape[0] > 2:
        return cantor_pairing(np.vstack((cantor_pairing(values[:2,:]), values[2:,:])))

def reset_id(series):
    """
    Assign and map ids to input series 
    """
    unique_arr = np.unique(series)
    unique_dict = {val:ind for ind, val in enumerate(unique_arr)}
    return series.map(unique_dict)


def convert_notebook(report_in_path, report_out_path, **kwargs):

    curdir = os.path.abspath(os.getcwd())
    indir, _ = os.path.split(report_in_path)
    outdir, _ = os.path.split(report_out_path)
    os.makedirs(outdir, exist_ok=True)

    config = {
        "ExecutePreprocessor": {"enabled": True},
        "TemplateExporter": {"exclude_output_prompt": True, "exclude_input": True, "exclude_input_prompt": True},
    }

    nb = nbformat.read(open(report_in_path), as_version=4)
    html_exporter = HTMLExporter(config=config)

    # change dir to notebook dir, to execute notebook
    os.chdir(indir)
    body, resources = (
        html_exporter
        .from_notebook_node(nb)
    )

    # change back to original directory
    os.chdir(curdir)

    with open(report_out_path, 'w') as fh:
        fh.write(body)
        

class UniqueIdGenerator:
    """
    Generate and Save Unique Id for each item 
    """
    
    def __init__(self, output_path, name=''):
        """
        Initialize the UniqueIdGenerator by 
            next_id : the id for next unique object start from 0
            output_path : output path 
            name : the name of the object 
            id_dict : dictionary to save both values of names 
        """
        
        self.next_id = 0
        self.output_path = output_path
        self.name = name
        self.id_dict = {}
            
    def get_id_dict(self):
        """
        Get id dictionary 
        """
        return self.id_dict
    
    def add(self, value):
        """
        Add new item for block_id 
        """

        # if key exist
        if value in self.id_dict.keys():
            return self.id_dict[value]

        else:
            current_id = self.next_id
            self.id_dict[value] = current_id
            self.next_id += 1
            return current_id
        

    def save_to_csv(self):
        """
        Save dictionary into .csv file 
        Only used for when content = True 
        """
        df = pd.DataFrame(
            data={f'{self.name}': self.id_dict.keys(), 
                    f'{self.name}_id':self.id_dict.values()}
        )
        df.to_csv(f'{self.output_path}/{self.name}.csv')
        



   
