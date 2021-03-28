import pandas as pd
import numpy as np
from attrdict import AttrDict 
import pageviewapi
import os
def page_view_test(read_link, to_link):

    directory = "output"
        
    # Parent Directory path  
    parent_dir = "./test/"
        
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path)  

    result = pd.read_csv(read_link)
    result=result.drop(columns=['Unnamed: 0'])
    result['title'] = result['title'].str.replace("_", " ")
    result.columns=['date','revert','edit','commentor','title','comment','Revision Time','M']
    page_view_df = result.groupby(['title']).agg({'date' : [np.min]})
    page_view_df.columns = ['min_date']
    page_view_df['title'] = page_view_df.index
    page_view_df['title'] = page_view_df['title'].str.replace("_", " ")
    page_view_df = page_view_df.reset_index(drop = True)
    page_view_df['min_date'] = pd.to_datetime(page_view_df['min_date'])
    page_view_df['min_date'] = page_view_df.min_date.map(lambda x: x.strftime('%Y%m%d'))

    dataframe = pd.DataFrame()
    dictionary_other = {}
    for i in np.arange(page_view_df.shape[0]):
        title =page_view_df.iloc[i]['title']
        start_date = page_view_df.iloc[i]['min_date']
        try:
            page_v_dict = pageviewapi.per_article('en.wikipedia',title, start_date, '20210101',
                                access='all-access', agent='all-agents', granularity='daily')
            new_dictionary = {}

            for key in page_v_dict: 
                for i in page_v_dict[key]:
                    new_dictionary['title'] = i['article'].replace('_',' ')
                    new_dictionary['timestamp'] = i['timestamp']
                    try:
                        new_dictionary['views']+=i['views']
                    except:
                        new_dictionary['views']=0


                new_dataf =  pd.DataFrame(new_dictionary,index=[0])
                dataframe = pd.concat([new_dataf,dataframe])
        except:
            dictionary_other[title] = np.nan
            continue
    page_view = dataframe.drop_duplicates()
    page_view.reset_index(drop=True,inplace=True)

    non = pd.DataFrame.from_dict(dictionary_other,orient='index')
    non['title'] = non.index
    non = non.rename(columns = {0:'views'})
    non = non.reset_index(drop = True)
    non['timestamp'] = '2021010100'

    frames = [page_view,non]
    page_view = pd.concat(frames)

    page_view = page_view[['title','views']]
    last_dataf = result.merge(page_view, how='left', on='title')

    result = last_dataf.to_csv(to_link, index = False)
    return result
