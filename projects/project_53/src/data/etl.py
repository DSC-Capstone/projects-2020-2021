import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET


def get_df(file_path):
    """Parses xml file and converts data to pandas file."""
    parsed = ET.parse(file_path)
    root = parsed.getroot()
    
    data = []

    for i, child in enumerate(root):
        data.append(child.attrib)
        
    # Turn into pandas DataFrame and set index to Id
    dfItem = pd.DataFrame.from_records(data).set_index('Id')
    return dfItem


def create_datasets(file, split_date, threshold):
    """" Create usable datasets from raw data """
    # Get data from file
    posts = get_df(file)
    split_date = eval(split_date)
    
    # Use only needed data and convert to right format
    relevant = posts[['PostTypeId', 'CreationDate', 'Score', 'Title', 'Body', 'Tags', 'OwnerUserId', 'AnswerCount', 'ParentId']]
    relevant['CreationDate'] = pd.to_datetime(relevant['CreationDate'])
    
    # Split the data to include only answered data after a certain date
    useful_data = relevant[(relevant['AnswerCount'] != '0') & (relevant['CreationDate'] >= pd.datetime(2018, 1, 1))]
    
    # Ensure answers are only for questions in the dataset and split into questions and answers
    useful_data_index = np.array(useful_data[useful_data['PostTypeId'] == '1'].index)
    useful_questions = useful_data.loc[useful_data_index]
    useful_answers = useful_data[useful_data['ParentId'].isin(useful_data_index)]
    
    # Drop irrelevant columns
    data_q = useful_questions.drop(['PostTypeId', 'ParentId', 'AnswerCount'], axis=1)
    data_a = useful_answers.drop(['PostTypeId', 'AnswerCount', 'Tags'], axis=1)
    
    # Set index
    data_a['Id'] = data_a.index
    data_q['Id'] = data_q.index
    
    # Sort Values
    data_a = data_a.sort_values('Score', ascending=False)
    
    # Filtering out low answering user
    grouped_a = data_a.groupby('OwnerUserId').count()
    print('Grouped Users Count: ', grouped_a.shape)
    
    grouped_a = grouped_a[grouped_a.Id > threshold]
    grouped_a['OwnerUserId'] = grouped_a.index
    users = list(grouped_a.OwnerUserId)
    filtered_a = data_a[data_a.OwnerUserId.isin(users)]
    filtered_q = data_q[data_q.Id.isin(filtered_a.ParentId)]
    
    # Print Sizes
    print('Filtered Grouped Users Count: ', grouped_a.shape)
    print('Number of Users: ', len(users))
    print('Original Answers Data Shape: ', data_a.shape)
    print('Filtered Answers Data Shape: ', filtered_a.shape)
    print('Original Questions Data Shape: ', data_q.shape)
    print('Filtered Questions Data Shape: ', filtered_q.shape)

    
    # Regex on Tag and Body to clean text
    clean = re.compile('<.*?>')
    clean_text = lambda x: re.sub(clean, '', x)
    clean_tags = lambda x: x.replace('<', ' ').replace('>', ' ')
    filtered_q['Tags'] = filtered_q['Tags'].fillna('').apply(clean_tags)
    filtered_q['Body'] = filtered_q['Body'].fillna('').apply(clean_text)
    
    # Create user and post indices starting at 0
    filtered_a.index = range(len(filtered_a.Id))
    filtered_q.index = range(len(filtered_q.Id))
    user_indices = pd.Series(range(len(filtered_a['OwnerUserId'].unique())), index=filtered_a['OwnerUserId'].unique()).drop_duplicates()
    
    # Create post_mappings
    user_dict = dict(user_indices)
    user_id = list(user_indices.values)
    post_indices = pd.Series(range(len(filtered_a['ParentId'].unique())), index=filtered_a['ParentId'].unique()).drop_duplicates()

    # Add post and user indices columns
    user_ind = lambda x: user_indices.loc[x]
    post_ind = lambda x: post_indices.loc[x]
    filtered_a['user_indicies'] = filtered_a['OwnerUserId'].apply(user_ind)
    filtered_a['post_indicies'] = filtered_a['ParentId'].apply(post_ind)
    filtered_q['post_indicies'] = filtered_q['Id'].apply(post_ind)
    filtered_q = filtered_q.sort_values('post_indicies', ascending=True)
    
    # Return train, validation, and test datasets
    return filtered_q, filtered_a, post_indices

def main(configs):
    FILE = configs["file"]
    SPLIT_DATE = configs["split_date"]
    THRESHOLD = configs["num_answers_threshold"]
    UNFINISHED_QUESTIONS = configs["unfinished_questions_file"]
    ANSWERS = configs["answers_file"]
    POST_MAPPINGS = configs["post_mappings"]
    df_q, df_a, post_indices = create_datasets(FILE, SPLIT_DATE, THRESHOLD)
    df_q.to_csv(UNFINISHED_QUESTIONS)
    df_a.to_csv(ANSWERS)
    post_indices.to_csv(POST_MAPPINGS)
    print('########### Data Created ###########')
    
    
if __name__ == "__main__":
    main(sys.argv)
    

    
