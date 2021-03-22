import os
import pandas as pd
import numpy as np

def calc_user_polarity(science_path, myth_path, politics_path, output_path, output_file): 
    myth_df = pd.read_csv(myth_path)
    politics_df = pd.read_csv(politics_path)
    science_df = pd.read_csv(science_path)
    
    myth_count = myth_df.groupby('author').count().iloc[:, 0].rename('myth (%)')
    politics_count = politics_df.groupby('author').count().iloc[:, 0].rename('politics (%)')
    science_count = science_df.groupby('author').count().iloc[:, 0].rename('science (%)')

    science_myth = pd.concat([science_count, myth_count], axis=1)
    all_three = pd.concat([science_myth, politics_count], axis=1)
    all_three = all_three.fillna(0)
    all_three['total'] = all_three.sum(axis=1)
    
    user_polarity = all_three.copy()
    user_polarity['science (%)'] = all_three['science (%)'] / all_three['total'] * 100
    user_polarity['politics (%)'] = all_three['politics (%)'] / all_three['total'] * 100
    user_polarity['myth (%)'] = all_three['myth (%)'] / all_three['total'] * 100
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    user_polarity.to_csv(output_path + '/' + output_file)