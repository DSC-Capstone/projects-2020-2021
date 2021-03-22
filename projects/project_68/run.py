#!/usr/bin/env python
# coding: utf-8

import sys
import json
import os

sys.path.insert(0, 'src') #getting the directory of our etl file in order to import etl since it is in a different folder
from etl import get_file_names, shorten_data, create_path
from all_pipeline import fastqc,cutadapt,second_fastqc,kallisto
from test_pipeline import test_kallisto, test_fastqc
from utils import function_to_convert_notebook, function_to_convert_notebook_test


def main(targets):
    
    if 'all' in targets:
        # HERE is where we run the FULL PIPELINE
        text = input("You are about to run the FULL pipeline this will take DAYS to run.\nOnly run this if you know what you are doing. Input [Yes/No] to run the pipeline now:\n")
        
        if (text == 'Yes') | (text =='yes'):
            text2 = input("Running the full pipeline will recreate everything that is in the report, you do NOT need to run the full pipeline to see our results! Asking again, are you SURE you want to run the FULL pipeline? Input [Yes/No] now:\n")
            if (text2 == 'Yes') | (text2 =='yes'):
                print("Running FULL pipeline now! Goodluck, have fun:")
                
#                 print("")
                with open('config/data-params.json') as fh:
                    data_cfg = json.load(fh)
                    raw_files = get_file_names(**data_cfg)
#                 raw_files = get_file_names()
                print("Running FastQC:\n")
                pass_data, fail_data = fastqc(raw_files)
                print("Running Cutadapt:\n")
                cut_fastq_files = cutadapt(pass_data)
                print("Running Second FastQC:\n")
                pass_cut_data, fail_cut_data = second_fastqc()
                print("Running Kallisto:\n")
                kallisto(pass_cut_data)
                print("Running R:\n")
                os.system("Rscript notebooks/R/DESeq_Wilcox.R")
                #output the report and the other html docs
                with open('config/eda-params.json') as fh:
                    eda_cfg = json.load(fh)
                    function_to_convert_notebook(**eda_cfg)
                with open('config/analyze-params.json') as fh:
                    analyze_cfg = json.load(fh)
                    function_to_convert_notebook(**analyze_cfg)
                with open('config/viz-params.json') as fh:
                    viz_cfg = json.load(fh)
                    function_to_convert_notebook(**viz_cfg)
                with open('config/report-params.json') as fh:
                    report_cfg = json.load(fh)
                    function_to_convert_notebook_test(**report_cfg)
                
        
    if 'data' in targets:
        #doing our data retrieval
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            get_file_names(**data_cfg)
      
    if 'eda' in targets:
        with open('config/eda-params.json') as fh:
            eda_cfg = json.load(fh)
            function_to_convert_notebook(**eda_cfg)
            
    if 'analyze' in targets:
        with open('config/analyze-params.json') as fh:
            analyze_cfg = json.load(fh)
            function_to_convert_notebook(**analyze_cfg)
            
    if 'viz' in targets:
        with open('config/viz-params.json') as fh:
            viz_cfg = json.load(fh)
            function_to_convert_notebook(**viz_cfg)
        
    if 'test' in targets:
        
        print("\n\nTest is now running\n")
        
        with open('config/test-params.json') as fh:
            test_cfg = json.load(fh)
            print("Creating path from inside Main:")
            create_path(**test_cfg)
        
            print("shortening the data from inside Main:")
            fq_1, fq_2 = shorten_data(**test_cfg)
            
            #fastqc:
            print("running fastqc from main:")
            pass_data, cut_data = test_fastqc([fq_1,fq_2])
            print(pass_data)
            
            # kallisto:
            test_kallisto(pass_data)
            
            #generating the report here:
#         print(os.listdir())
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
            function_to_convert_notebook_test(**report_cfg)
            
            print("Success!")
        print("\n\nTest finished running\n\n")
    return

# In[ ]:


if __name__ ==  '__main__':
    targets = sys.argv[1:]
    main(targets)