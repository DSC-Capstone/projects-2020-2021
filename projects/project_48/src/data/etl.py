import os
from os import listdir, path, makedirs

import pandas as pd
import random
import sys

def main(configs):
    folder = configs['original_loc']
    out_folder = configs['output_location']
    # create necessary folders:
    makedirs(folder, exist_ok=True)
    makedirs(out_folder, exist_ok=True)
    
    os.environ['KAGGLE_USERNAME'] = "anthonyfong123"
    os.environ['KAGGLE_KEY'] = "ba9a57520251afe7d52832c1456def56"
    import kaggle
    #!kaggle competitions download -d shuyangli94/food-com-recipes-and-user-interactions
    #kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
    #kaggle.datasets.download
    kaggle.api.authenticate()
    
    #Data from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions
    kaggle.api.dataset_download_files('shuyangli94/food-com-recipes-and-user-interactions', path=folder, unzip=True)
    
    #Data from https://www.kaggle.com/kaggle/recipe-ingredients-dataset/home
    kaggle.api.dataset_download_files('kaggle/recipe-ingredients-dataset', path=folder, unzip=True)
    
    raw_r = pd.read_csv(configs['recipes'])
    raw_i = pd.read_csv(configs['inter'])

    raw_r = raw_r[raw_r['minutes'] < 300000]
    combined = pd.merge(raw_r,raw_i,how='inner',left_on='id',right_on='recipe_id')
    combined.to_csv(out_folder+'/combined.csv',index=False)
    
    final_r = raw_r.astype({'id': 'object','contributor_id': 'object'})
    final_r = final_r.rename(columns={'id':'recipe_id'}).set_index('recipe_id')
    final_i = raw_i[['rating','recipe_id']].groupby('recipe_id')['rating'].agg(['mean','count'])
    final_i = final_i.rename(columns = {'mean':'mean_rating','count':'review_count'})
    final_data = final_r.merge(final_i, on = 'recipe_id')
    final_data.to_csv(out_folder+'/final_data.csv',index=False)

if __name__ == "__main__":
    main(sys.argv)
