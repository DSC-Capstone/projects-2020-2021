import os
import json
import numpy as np 
from mlxtend.evaluate import permutation_test

def normalize_likes(likes_over_years):
    '''
    normalize likes over current year using (curr_year_likes - mean_lastyear_likes) / mean_last_year_likes
    ''' 
    normal_likes_over_years = {}
    years = [str(year) for year in range(2008, 2021)]
    
    for i in range(1, len(years)):
        prev_mean = np.mean(likes_over_years[years[i-1]])
        normal_likes_over_years[years[i]] = list(map(lambda x: (x - prev_mean) / prev_mean, likes_over_years[years[i]]))
        
    return normal_likes_over_years

def run_permutation_between(dist1, dist2, outpath):
    '''
    run permutation test between scientific and misinformation 
    '''
    years = [str(year) for year in range(2009, 2021)]
    s = ""
    for year in years:
        p_value = permutation_test(dist1[year],
                                   dist2[year],
                                   method='approximate',
                                   num_rounds=10000,
                                   seed=42)
        s += 'Scientific vs Misinformation permutation test for year '+ year + ': the p-value is  ' + str(p_value) + '.\n'
    
    with open(outpath + '/permutation_between.txt', 'w+') as f:
        f.write(s)

def run_permutation_within(dist, category, outpath): 
    '''
    run permutation test within one group and compare year by year 
    '''
    years = [str(year) for year in range(2009, 2021)]
    s = ""
    for i in range(len(years)-1):
        p_value = permutation_test(dist[years[i]],
                                   dist[years[i+1]],
                                   method='approximate',
                                   num_rounds=10000,
                                   seed=42)
        s += '{0} {1} vs {0} {2} permutation test: the p-value is {3}. \n'.format(category, years[i], years[i+1], p_value)
    
    with open(outpath + '/' + category + '_permutation_within.txt', 'w+') as f:
        f.write(s)

def run_permutations(sci_inpath, misinfo_inpath, outpath):
    '''
    runs all permutation tests 
    '''
    with open(sci_inpath) as f:
        sci_data = json.load(f)
    with open(misinfo_inpath) as f:
        mis_data = json.load(f)
    
    sci_normal = normalize_likes(sci_data)
    mis_normal = normalize_likes(mis_data)
    
    run_permutation_within(sci_normal, 'Scientific', outpath)
    run_permutation_within(mis_normal, 'Misinformation', outpath)
    
    run_permutation_between(sci_normal, mis_normal, outpath)