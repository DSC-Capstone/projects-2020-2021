from stackapi import StackAPI
import pandas as pd
import numpy as np
import time
import datetime
import json
import os

def get_id(x):
    """ Transforms owner column into just the user ID """
    if type(x) == dict:
        return x.get('user_id')
    else:
        return x
    
def api_get(configs):
    """ Get new data from API """
    # Apply config file paramaters to create SITE
    last_request = configs['last_request']
    SITE = StackAPI(configs['site'], key=configs['key'])
    SITE.page_size = configs['page_size']
    SITE.max_pages = configs['max_pages']
    questions_filter = configs['questions_filter']
    answers_filter = configs['answers_filter']

    # Get request time (inside parameters for test data only)
    if configs.get("request_time"):
        request_time = configs.get("request_time")
        print('test')
    else:
        request_time = int(time.time())
        print('didn\'t test')
        
    print(request_time - last_request)

        
    # Get the questions  
    questions = SITE.fetch('questions', filter=questions_filter, fromdate=last_request, todate=request_time)

    # Transform the question data
    full_questions = pd.DataFrame(questions)
    question_items = pd.DataFrame(full_questions['items'].tolist())
    question_items['owner'] = question_items['owner'].apply(get_id)
    question_items['creation_date'] = pd.to_datetime(question_items['creation_date'], unit='s')
    print(question_items.shape[0])

    # Get the answers
    answers = SITE.fetch('answers', filter=answers_filter, fromdate=last_request, todate= request_time)

    # Transform the answer data
    full_answers = pd.DataFrame(answers)
    answer_items = pd.DataFrame(full_answers['items'].tolist())
    answer_items['owner'] = answer_items['owner'].apply(get_id)
    answer_items['creation_date'] = pd.to_datetime(answer_items['creation_date'], unit='s')
    print(question_items.shape[0])

    # Merge both datasets and create post type column
    combo = pd.concat([question_items, answer_items])
    combo.reset_index(drop=True, inplace=True)
    combo['post_type'] = combo['answer_id'].apply(lambda x: int(pd.notna(x) + 1))

    # Filter out answered questions
    answered_questions = combo[(combo['post_type'] == 2) & (combo['score'] > 2)].question_id
    combo = combo[~combo['question_id'].isin(answered_questions)].reset_index(drop=True)
    
    # Return dataset
    return combo 

def update_configs(configs, filepath):
    """ Update configs file with new information """
    
    # Does not update if test data
    if configs.get("request_time"):
        return None
    else:
        new_file = {
            "page_size": configs["page_size"],
            "max_pages": configs["max_pages"],
            "last_request": last_request,
            "key": configs["key"],
            "site": configs["site"],
            "questions_filter": configs["questions_filter"],
            "answers_filter": configs["answers_filter"],
            "api_data": configs["api_data"]
        }

        with open(filepath, 'w') as conv: 
             conv.write(json.dumps(new_file))

def main(filepath):

    # Open configs file  
    with open(filepath) as conf:
        configs = json.load(conf)
    
    # Get Data
    api_data = api_get(configs)
    
    # Add data to the api csv file, create if there is no data
    if os.path.isfile(configs["api_data"]):
        api_data.to_csv(configs["api_data"], mode='a', header=False)
    else:
        api_data.to_csv(configs["api_data"])

    # Update configs file
    update_configs(configs, filepath)

    print('########### New API Data Generated ###########')
    
    
if __name__ == "__main__":
    main(sys.argv)
    
