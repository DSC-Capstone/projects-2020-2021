#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
def function_to_convert_notebook(outdir, report_in_path, report_out_path):
    os.system("mkdir report")
    #print(report_in_path)
    #print(os.listdir())
    os.system("jupyter nbconvert --to html "+ report_in_path)
    #print(os.listdir())
    os.system("mv "+report_out_path+" report")
    return

def function_to_convert_notebook_test(outdir, report_in_path, report_out_path):
    os.system("mkdir report")
    os.system("jupyter nbconvert --to html "+ report_in_path)
    #print(os.listdir())
    os.system("mv notebooks/Alzheimers-Biomarker-Analysis.html report")
    #os.rename('report/report.html', 'report/complete_report.html')
    return

# In[ ]: