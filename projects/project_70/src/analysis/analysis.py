#!/usr/bin/env python
# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020

# This step will take the preprocessed fastq data and analyze the RNA levels for the 4 groups and 
# determine correlations

import os
import re
import pandas as pd
import numpy as np
import logging
    


def process_biofluid_disorder_comparisons(deseq2, output_prefix, cleanup, verbose):
    '''
    Purpose : Analyze the disorder comparisons per brain region
    Arguments: 
    deseq2          : dictionary of normalize deseq2 config arguments
    output_prefix   : output prefix for all files (normalized matrix, LRT, etc)
    cleanup         : if set to 1 then remove merged bam input file
    verbose : set to 1 if you want verbose output
    '''
    tmp_files = []
    # Iterate for each brain region
    for biofluid_region in deseq2["biofluid_regions"]:
        # Create Directory
        biofluid_dir = output_prefix.replace("%biofluid_region%", biofluid_region) + "/"
        if not os.path.isdir(biofluid_dir):
            os.makedirs(biofluid_dir)
            if verbose:
                logging.info("mkdir " + biofluid_dir)
        # load full matrix and features
        biofluid_count_matrix_df = pd.read_csv(deseq2["input_counts"], sep="\t", index_col=0)
        biofluid_feature_df = pd.read_csv(deseq2["input_features"], sep="\t")
        # filter brain region
        biofluid_feature_df = biofluid_feature_df[biofluid_feature_df["Biofluid"] == biofluid_region]


        for disorder in deseq2["disorders"]:
            disorder_a =  disorder
            disorder_b =  deseq2["control"]
            disorder_dir = (biofluid_dir + disorder).replace(" ", "_") + "/"
            if not os.path.isdir(disorder_dir):
                os.makedirs(disorder_dir)
                if verbose:
                    logging.info("mkdir " + disorder_dir)


            feature_df = biofluid_feature_df.copy()
            count_matrix_df = biofluid_count_matrix_df.copy()

            logging.info(biofluid_region + " x " + disorder_a + " vs Control")
            # filter all a and b disorders
            feature_df = feature_df[ (feature_df["Disorder"] == disorder_a) | (feature_df["Disorder"] == disorder_b)] 
            feature_df["Disorder"] = feature_df["Disorder"].replace({disorder_a:0, disorder_b:1})

            # filter samples in count matrix
            count_matrix_df = count_matrix_df[ feature_df["Run"].tolist() ]
            # save the filtered count and feature tables
            input_count_filename = disorder_dir + "gene_matrix.tsv"
            tmp_files.append(input_count_filename)
            count_matrix_df.to_csv(input_count_filename, sep="\t")
            input_feature_filename = disorder_dir + "features.tsv"
            tmp_files.append(input_feature_filename)
            
            # remove biofluid_region from feature table
            feature_df = feature_df.drop(["Biofluid"], axis=1) # , "Run"
            feature_df.to_csv(input_feature_filename, sep="\t", index=False)

            # DEQSe2 Process
            if os.path.exists(deseq2["Rscript"]):
                command = deseq2["Rscript"] + " "
            else:
                command = "Rscript "
            command += deseq2["source"]
            command += " "
            command += input_count_filename
            command += " "
            command += input_feature_filename
            command += " "
            command += disorder_dir
            command += " full="
            command += str(deseq2["full"])
            command += " reduced="
            command += str(deseq2["reduced"])
            command += " charts=1"
            if deseq2["parallel"]==1:
                command += " parallel=1"
            else:
                command += " parallel=0"
            if verbose:
                logging.info(command)
            os.system(command)

            if cleanup == 1:
                for tmp_file in tmp_files:
                    if verbose:
                        logging.info("rm " + tmp_file)
                    os.remove(tmp_file)   
                tmp_files = []





def process_analysis(deseq2, output_prefix, cleanup, verbose):
    '''
    Purpose : normalize merged BAM script using R's DESeq2
    Arguments: 
    deseq2          : dictionary of normalize deseq2 config arguments
    output_prefix   : output prefix for all files (normalized matrix, LRT, etc)
    cleanup         : if set to 1 then remove merged bam input file
    verbose : set to 1 if you want verbose output
    '''
    
    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Analysis")
    
    # process each brain region x disorder - used in pvalue histogram plot
    process_biofluid_disorder_comparisons(deseq2, output_prefix, cleanup, verbose)


    if verbose:
        logging.info("# Finished")
        logging.info("# ---------------------------------------------------")


