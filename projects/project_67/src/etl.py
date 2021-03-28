#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

def get_file_names(file_path):
    raw_files = []
    os.chdir(file_path)
    print("\nGetting file names from DSMLP:\n")
    raw_files = os.listdir()
    print(raw_files)
    os.chdir("/datasets/home/40/840/r1cummin/alzheimers_gene_analysis")
    return raw_files

#for test
def shorten_data(test_1,test_2):
    print("Creating the shorten files for testing:")
    os.system("zcat test/testdata/"+test_1+" | head -70000 > "+test_1[0:9]+".fastq.gz")
    os.system("zcat test/testdata/"+test_2+" | head -70000 > "+test_2[0:9]+".fastq.gz")
    print("Files created successfully!")
    return test_1[0:9]+".fastq.gz", test_2[0:9]+".fastq.gz"



# In[ ]:

# for test
def create_path(test_1,test_2):
    print("attempting to create symlink")
    os.system("ln -s /teams/DSC180A_FA20_A00/b04genetics/group_1/raw_data/"+test_1+" test/testdata/"+test_1)
    os.system("ln -s /teams/DSC180A_FA20_A00/b04genetics/group_1/raw_data/"+test_2+" test/testdata/"+test_2)
    print("Symlink created!")
    return