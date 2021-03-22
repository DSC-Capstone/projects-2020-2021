#! /usr/bin/env python

import sys
import json
import pandas as pd
from etl import *
from eda import *
from revision import *
from word import *

def main(targets):
    if 'data' in targets:
        
      with open('config/data-params.json') as fh:
        data_cfg=json.load(fh)
      top_1000()
      pageview_csv(data_cfg['articles'], data_cfg['pageview'])
      for i in data_cfg['essential']:
        get_revisions(i, "data/revisions")
    if 'eda' in targets:
         
      with open('config/data-params.json') as fh:
        data_cfg=json.load(fh)
      router=get_dfs(data_cfg['pageview'])
      top10er=get_top_10_average_daily_view(router, data_cfg['eda'])
      plot_top10(top10er, router, data_cfg['eda'])
    if 'revision' in targets:
      result=get_all_df('data/revisions')
      get_user_activities(result, 'data/result')
      LDA(result,'data/result')
    if 'word' in targets:
      gener('data/result')

    if 'test' in targets:
      with open('config/data-params.json') as fh:
        data_cfg=json.load(fh)
      pageview_csv("test/test.csv", "test/pageview")
      for i in data_cfg['essential']:
        get_revisions(i, "test/revisions")
      router=get_dfs('test/pageview')
      top10er=get_top_10_average_daily_view(router,'test/eda')
      plot_top10(top10er,router,'test/eda')
      result=get_all_df('test/revisions')
      get_user_activities(result, 'test/result')
      LDA(result, 'test/result')
      gener('test/result')

    return






if __name__ == "__main__":
    targets=sys.argv[1: ]
    main(targets)