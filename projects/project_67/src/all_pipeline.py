#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd

#fastqc
def fastqc(raw_files):
    pass_data = [] #creating array that will hold name of passing data
    fail_data = [] #creating array that will hold name of not passing data
    for i in raw_files:
#         os.chdir('..')
#         os.chdir("alzheimers_gene_analysis/")
        #os.chdir('/datasets/home/40/840/r1cummin/alzheimers_gene_analysis')
        print("making dir")
        os.system('mkdir fastq_out')
        print("running fastqc...")
        os.system('/opt/FastQC/fastqc /teams/DSC180A_FA20_A00/b04genetics/group_1/raw_data/'+i+' --extract --outdir=fastq_out/')
        print("Fastqc finished")
      
        print("Opening file")
        with open('fastq_out/'+i[0:-3].replace('.','_')+'c/fastqc_data.txt') as f:
            first_line = f.readlines()[1]
        print("Seeing if pass in file... then adding to array")
        if 'pass' in first_line:
            pass_data.append(i)
        else:
            fail_data.append(i)
        print("Now deleting dir in order in order to do the next iteration:")
        os.system('rm -rf fastq_out')
    return pass_data, fail_data

#cutadapt
def cutadapt(pass_data):
    cut_fastq_files = []
    print("making temp dir")
    os.system('mkdir cutadapt_tmp')
    for i in (pass_data):
        print(i)
        print("Starting cutadapt:")
        os.system('cutadapt -a AGATCGGAAGAGC -o cutadapt_tmp/'+i+' /teams/DSC180A_FA20_A00/b04genetics/group_1/raw_data/'+i+' --cores=32')
        cut_fastq_files.append(i)
        print("cutadapt finished on this file, starting on next file:")
    print("cutadapt finished in its entirety")
    os.system("mkdir data")
    os.system("mkdir out")
    os.system("mv out data")
    os.system("mv cutadapt_tmp data/out")
    return cut_fastq_files

#second_fastqc
def second_fastqc():
    # reRunning the cutadapt files through fastqc

    pass_cut_data = [] #creating array that will hold name of passing data
    fail_cut_data = [] #creating array that will hold name of not passing data

    print('Getting name of cutadapt files')
    os.chdir('data/out/cutadapt_tmp')
    cut_fastq_files = os.listdir() #all the trimmed fastq files
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    cut_fastq_files = [i for i in cut_fastq_files if '.fastq' in i]
    cut_fastq_files.sort()

    os.system('mkdir cut_fastq_out')
#     counter = 1
    for i in cut_fastq_files:
#         print(i)
#         print(counter)
#         counter += 1
#         os.chdir('..')
#         print("making dir...")
#         os.system('mkdir cut_fastq_out')
        print("running fastqc...")
        try:
            os.system('/opt/FastQC/fastqc cutadapt_tmp/'+i+' --extract --outdir=cut_fastq_out/')
        except:
            print("The file "+i+" encountered an error (Most likely too truncated)\nMoving to next file")
            fail_cut_data.append(i)
            continue
            
        print("Fastqc finished")
        
        print("Opening file")
#         print(' the file path before opeining: cut_fastq_out/'+i[0:-3][::-1].replace('.','_',1)[::-1]+'c/fastqc_data.txt')
        try:
            print(' the file path before opeining: cut_fastq_out/'+i[0:-3][::-1].replace('.','_',1)[::-1]+'c/fastqc_data.txt')
            with open('cut_fastq_out/'+i[0:-3][::-1].replace('.','_',1)[::-1]+'c/fastqc_data.txt') as f:
                cut_first_line = f.readlines()[1]
            print("Seeing if pass in file... then adding to array")
            if 'pass' in cut_first_line:
                pass_cut_data.append(i)
            else:
                fail_cut_data.append(i)
        except:
            print("The file "+i+" encountered an error (Most likely too truncated)\nMoving to next file")
            fail_cut_data.append(i)
            continue
        #print("Now deleting dir in order so that we can do the next iteration:")
        os.system('rm -rf cut_fastq_out')
    return pass_cut_data, fail_cut_data


#kallisto
def kallisto(pass_cut_data):
    
    print("Making dir for Kallisto Output:")
    os.system("mkdir kallisto_tmp")
    for i in pass_cut_data:
        print("Running Kallisto:")
        #print(i)
        command = f"/opt/kallisto_linux-v0.42.4/kallisto quant -i data/reference.idx -o kallisto_tmp/kallisto_output_"+i[0:9]+" --single -l 50 -s 10 -b 8 -t 8 cutadapt_tmp/"+i
        os.system(command)
    print("Kallisto successfully ran")
    print("Still need to combine all the CSVs though!")
    print("Moving the Kallisto_tmp data to data/out/")
#     os.system("mkdir out")
    os.system("mv out data")
    os.system("mv kallisto_tmp data/out")
    
    #combining the counts:
    df = pd.DataFrame()
    os.chdir("data/out/kallisto_tmp")
    kallisto_count_files = os.listdir()
    kallisto_count_files = [i for i in kallisto_count_files if 'kallisto_output_' in i]
    kallisto_count_files.sort()
    for i in kallisto_count_files:

        print("Kallisto ran now adding to the dataframe, with only the counts (Also renaming the column name)")
        #print(i)
        tmp_df = pd.read_csv(i+"/abundance.tsv",sep="\t")[['target_id','est_counts']]
        tmp_df[i[-9:]] = tmp_df['est_counts']
        tmp_df = tmp_df.drop(['est_counts'],axis=1)
        if 'target_id' in df.columns:
            df = pd.concat([df,tmp_df[i[-9:]]],axis=1)
        else:
            df = pd.concat([df,tmp_df],axis=1)

        print("Added to the dataframe")
        print("Continuing to next iteration:\n")
    df.to_csv('kallisto_counts.csv')
    os.system("mv kallisto_counts.csv data/out")
    return