import pandas as pd
import os
import numpy as np

from sklearn.datasets import fetch_20newsgroups
import src.utils  as utils


def bbc_preprocessing(path, save_p):

    # set Data Path and type of news list
    data_folder = path + 'News Articles/'
    summary_folder = path + 'Summaries/'
    entries = os.listdir(data_folder)

    # Collect all data from each txt and ready to add to CSV file.
    all_data = {}
    for i in entries:
        temp = []
        folder_path = data_folder + i +'/'
        file_lst = os.listdir(folder_path)
        for j in file_lst:
            if j != '.ipynb_checkpoints':
                with open(folder_path +j, 'r', errors='ignore') as file:
                    temp.append(file.read().replace('\n', ''))
        all_data[i] = temp


    # Collect summary of news and ready to add to CSV file.
    all_sum = {}
    for i in entries:
        temp = []
        folder_path = summary_folder + i +'/'
        file_lst = os.listdir(folder_path)
        for j in file_lst:
            if j != '.ipynb_checkpoints':
                with open(folder_path +j, 'r', errors='ignore') as file:
                    temp.append(file.read().replace('\n', ''))
        all_sum[i] = temp


    # Create DataFrame object for cleaning.
    # Create test.csv for test target. Which select first 10 news from each group.
    total_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in np.arange(len(entries)):
        if i == 0:
            total_df = pd.DataFrame.from_dict(all_data['business'])
            total_df['type'] = 'business'
            total_df['summary'] = all_sum['business']
            total_df['type_code'] = i+1
            
            
            index = np.random.choice(total_df.shape[0],10)
            test_df = total_df.loc[index]
        else:
            temp_df = pd.DataFrame.from_dict(all_data[entries[i]])
            temp_df['type'] = entries[i]
            temp_df['summary'] = all_sum[entries[i]]
            temp_df['type_code'] = i+1
            total_df =pd.concat([total_df, temp_df], axis=0)
            
            index = np.random.choice(temp_df.shape[0], 10)
            temp_test_df = temp_df.loc[index]
            test_df =pd.concat([test_df,temp_test_df] , axis=0) 
            
    # Save the complete data csv.
    total_df.columns = ['text','type','summary','type_code']
    total_df = total_df.reset_index(drop = True)
    total_df.to_csv(os.path.join(save_p+'bbc_data.csv'), index=False)


    # Save the test.csv.
    test_df.columns = ['text','type','summary','type_code']
    test_df = test_df.reset_index(drop = True)
    test_df.head()
    test_df.to_csv(os.path.join(save_p +'test.csv'), index=False)

    print('Done')

def news_preprocessing(save_p):
    """
    Preprocessing function for 20 news group. It will generate a text file for Autophrase tool to extract quality phrases.
    """

    newsgroups_train = fetch_20newsgroups(subset='train')
    news_df = pd.DataFrame.from_dict(newsgroups_train,'index').T
    with open(save_p +"/20news.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(news_df.data.apply(lambda x: utils.clean_text(x)+ '.')))
    
    
    print('=========================================================')  
    print('20 News Group Dataset Ready for Running AutoPhrase..')
    print('=========================================================')
    
    
    