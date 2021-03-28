import sys
import os
import json

sys.path.insert(0, 'src/lib')

from data import get_data, get_csvs
from ratio import get_ratio_csv
from metrics import * 
from metrics_dataviz import *
from ratio_dataviz import *
from sort_tweets import sort_files
from permutation_tests import run_permutations

def main(targets):
    '''
    driver method for running through targets to recreate our project
    '''
    all_flag = False

    with open('config/data-params.json') as fh:
        data_cfg = json.load(fh)
    
    if targets == []:
        all_flag = True

    if 'data' in targets or all_flag:
        get_data(data_cfg['scientific_path'], data_cfg['misinformation_path'])
        get_csvs(data_cfg['scientific_path'], data_cfg['misinformation_path'], 
                    data_cfg['scientific_politicians'], data_cfg['misinformation_politicians'])

    if 'ratio' in targets or all_flag:
        get_ratio_csv(data_cfg['scientific_path'], data_cfg['misinformation_path'], 
                    data_cfg['scientific_politicians'], data_cfg['misinformation_politicians'])

    if 'metrics' in targets or all_flag:
        sort_files(data_cfg['scientific_path'])
        sort_files(data_cfg['misinformation_path'])

        outpath = data_cfg['output_path']
        sci_path = data_cfg['scientific_path']
        misinfo_path = data_cfg['misinformation_path']
        sci_sort_path = data_cfg['scientific_sorted_path']
        misinfo_sort_path = data_cfg['misinformation_sorted_path']
        x_months = 4
        x_tweets = 200

        count_likes_over_months(sci_path, outpath, 'scientific')
        count_likes_over_months(misinfo_path, outpath, 'misinfo')

        avg_likes_over_months(sci_path, outpath, 'scientific', x_months)
        avg_likes_over_months(misinfo_path, outpath, 'misinfo', x_months)

        max_likes_over_months(sci_path, outpath, 'scientific', x_months)
        max_likes_over_months(misinfo_path, outpath, 'misinfo', x_months)

        cumu_likes_over_months(sci_path, outpath, 'scientific')
        cumu_likes_over_months(misinfo_path, outpath, 'misinfo')

        count_likes_over_tweets(sci_sort_path, outpath, 'scientific')
        count_likes_over_tweets(misinfo_sort_path, outpath, 'misinfo')

        avg_likes_over_tweets(sci_sort_path, outpath, 'scientific', x_tweets)
        avg_likes_over_tweets(misinfo_sort_path, outpath, 'misinfo', x_tweets)

        max_likes_over_tweets(sci_sort_path, outpath, 'scientific', x_tweets)
        max_likes_over_tweets(misinfo_sort_path, outpath, 'misinfo', x_tweets)

        cumu_likes_over_tweets(sci_sort_path, outpath, 'scientific')
        cumu_likes_over_tweets(misinfo_path, outpath, 'misinfo')

        group_likes_over_years(sci_path, outpath, 'scientific')
        group_likes_over_years(misinfo_path, outpath, 'misinfo')

        group_likes_over_months(sci_path, outpath, 'scientific')
        group_likes_over_months(misinfo_path, outpath, 'misinfo')

    if 'visualization' in targets or all_flag:
        scientific_ratios_graph(data_cfg['scientific_path'], data_cfg['output_path'], data_cfg['scientific_politicians'])
        misinfo_ratios_graph(data_cfg['misinformation_path'], data_cfg['output_path'], data_cfg['misinformation_politicians'])
        outpath = data_cfg['output_path']
        sci_avg_likes_time_path = outpath + '/scientific_avg_likes_over_months.json'
        mis_avg_likes_time_path = outpath + '/misinfo_avg_likes_over_months.json'

        sci_largest_ratio(['Sen. Lisa Murkowski', 'Rep. Katie Porter'], sci_avg_likes_time_path, outpath)
        misinfo_largest_ratio(['Lindsey Graham', 'Tulsi Gabbard 🌺'], mis_avg_likes_time_path, outpath)
        both_largest_ratio(['Sen. Lisa Murkowski', 'Lindsey Graham'], sci_avg_likes_time_path, mis_avg_likes_time_path, outpath)

        most_likes_comparison(['Alexandria Ocasio-Cortez', 'Rep. Jim Jordan'], sci_avg_likes_time_path, mis_avg_likes_time_path, outpath)
        
        sci_avg_likes_tweets_path = outpath + '/scientific_avg_likes_over_tweets.json'
        mis_avg_likes_tweets_path = outpath + '/misinfo_avg_likes_over_tweets.json'

        most_tweets_comparison(['Alexandria Ocasio-Cortez', 'Rep. Jim Jordan'], sci_avg_likes_tweets_path, mis_avg_likes_tweets_path, outpath)
        
        sci_group_likes_year_path = outpath + '/scientific_group_likes_over_years.json'
        mis_group_likes_year_path = outpath + '/misinfo_group_likes_over_years.json'
        group_sum_over_year(sci_group_likes_year_path, mis_group_likes_year_path, outpath)

        normalized_group_sum_over_year(sci_group_likes_year_path, mis_group_likes_year_path, outpath)

        sci_group_likes_months_path = outpath + '/scientific_group_likes_over_months.json'
        mis_group_likes_months_path = outpath + '/misinfo_group_likes_over_months.json'
        group_median_over_month(sci_group_likes_months_path, mis_group_likes_months_path, outpath)

    if 'permute' in targets or all_flag:
        outpath = data_cfg['output_path']
        sci_path = outpath + '/scientific_group_likes_over_years.json'
        mis_path = outpath + '/misinfo_group_likes_over_years.json'

        run_permutations(sci_path, mis_path, outpath)


    if 'test' in targets: 
        test_output = data_cfg['test_output_path']

        avg_likes_over_months(data_cfg['test_scientific_path'], test_output, 'scientific', 5)
        avg_likes_over_months(data_cfg['test_misinformation_path'], test_output, 'misinfo', 5)
        max_likes_over_months(data_cfg['test_scientific_path'], test_output, 'scientific', 5)
        max_likes_over_months(data_cfg['test_misinformation_path'], test_output, 'misinfo', 5)

        test_sci_avg_likes_over_months_path = test_output + '/scientific_avg_likes_over_months.json'
        test_mis_avg_likes_over_months_path = test_output + '/misinfo_avg_likes_over_months.json'
        test_sci_max_likes_over_months_path = test_output + '/scientific_max_likes_over_months.json'
        test_mis_max_likes_over_months_path = test_output + '/misinfo_max_likes_over_months.json'

        sci_likes_over_months(["User4", "User5", "User6"], test_sci_avg_likes_over_months_path, test_output)
        misinfo_likes_over_months(["User1", "User2", "User3"], test_mis_avg_likes_over_months_path, test_output)
        compare_sci_misinfo(["User4", "User1"], test_sci_max_likes_over_months_path, test_mis_max_likes_over_months_path, test_output)
        max_all_sci(test_sci_max_likes_over_months_path, test_output)
        max_all_misinfo(test_mis_max_likes_over_months_path, test_output)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)