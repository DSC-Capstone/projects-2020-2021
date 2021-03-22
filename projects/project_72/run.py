#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')

from data_reddit import get_data
from data_reddit import parse_reddit
from format_data_reddit import make_content_text
from format_data_reddit import make_content_search
from lsh import *
#from lsh import process_all
from lsh import update_df
from emotion import find_lexicon
from emotion import limbic_score
from save_data import save_csv

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
    if 'test' in targets:
        # load short version of data
        fp = 'test/textdata/drugs_test_year.json'
        QUERY="test"
        term1 = 'test/testdata/terms.csv'
        term2 = 'test/testdata/terms2.csv'
        terms,ontology,df,QUERY= get_data(term1,term2,fp,QUERY)
        #format data
        content_term = make_content_search(terms)
        
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        # finding drug terms
        lsh_ls_term = create_lsh_term(content_term,**analysis_cfg)
        ls_total = process_all(lsh_ls_term,df,QUERY, **analysis_cfg)

        update_df(ls_total,df,terms,QUERY) 
        
        # analysis emotion of sentence
        find_lexicon(df)
        df = limbic_score(df)

        #parse dependency for identified sentence
        save_csv(df,'test_data_result.csv')
        
    if 'data' in targets or 'analysis' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        # load data
        terms,ontology,df,QUERY= get_data(**data_cfg)
                
        #format data
        content_term = make_content_search(terms)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        # get list of matching ontologies
        lsh_ls_term = create_lsh_term(content_term,**analysis_cfg)
        ls_total = process_all(lsh_ls_term,df,QUERY, **analysis_cfg)
 
        #update data
        update_df(ls_total,df,terms,QUERY) 
        
        # analysis emotion of sentence
        find_lexicon(df)
        df = limbic_score(df)

        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        # make the data target
        save_csv(df,**model_cfg)


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
