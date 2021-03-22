#!/usr/bin/env python
# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020

import os
from shutil import rmtree
from shutil import copyfile
from shutil import move
import re
import pandas as pd
import numpy as np
from random import randint
from random import seed
import logging

def clean_data(raw_dir, tmp_dir, out_dir, verbose):
    '''
    Purpose : Clean the data folders
    Arguments: 
    raw_dir : directory where initial source data will be copied
    tmp_dir : directory where temporary files will be created
    out_dir : directory where final output files will be created
    verbose : set to 1 if you want verbose output
    '''

    #if input("Are you sure? (yes/no) ") != "yes":
    #    exit()

    if os.path.isdir(raw_dir):
        rmtree(raw_dir)
        if verbose:
            logging.info("rm " + raw_dir)
    
    if os.path.isdir(tmp_dir):
        rmtree(tmp_dir)
        if verbose:
            logging.info("rm " + tmp_dir)
    
    if os.path.isdir(out_dir):
        rmtree(out_dir)
        if verbose:
            logging.info("rm " + out_dir)

    if verbose:
        logging.info("# Data folders deleted and cleaned")
    return


def make_data_directories(raw_dir, tmp_dir, out_dir, verbose):
    # Create raw directory
    if not os.path.isdir(raw_dir):
        if verbose:
            logging.info("mkdir " + raw_dir)
        os.makedirs(raw_dir)
    elif verbose:
            logging.info(raw_dir + " exists already")

    # Create tmp directory
    path = os.path.join(os.getcwd(), tmp_dir)
    if not os.path.isdir(tmp_dir):
        if verbose:
            logging.info("mkdir " + tmp_dir)
        os.makedirs(tmp_dir)
    elif verbose:
            logging.info(tmp_dir + " exists already")

    # Create out directory
    path = os.path.join(os.getcwd(), out_dir)
    if not os.path.isdir(out_dir):
        if verbose:
            logging.info("mkdir " + out_dir)
        os.makedirs(out_dir)
    elif verbose:
            logging.info(out_dir + " exists already")

    if verbose:
        logging.info("Data directories created")
    return


def invoke_process(sample, process, tmp_dir, output_files, verbose):
    '''
    Purpose : finds and removes adapter sequences
    Arguments: 
    sample          : one patient sample 
    process         : dictionary of process config arguments (removes adapter sequences, e.g. cutadapt)
    tmp_dir         : directory where temporary files will be created
    output_files    : location of two output files
    verbose         : set to 1 if you want verbose output
    returns the path for the two fastq files
    '''
    fastq_1 = sample["FASTQ_1"]
    fastq_2 = sample["FASTQ_2"]
    success = False;
    if process["enable"] == 1:
        # update the input and output file
        command = process["tool"]
        command += " "
        command += process["arguments"]
        command += " "
        # Add adapters for trimmed pair
        for adapter in process["r1_adapters"]:
            command += " -a " + adapter
        for adapter in process["r2_adapters"]:
            command += " -A " + adapter    
        command += " -o " + output_files[0] + " -p " + output_files[1]
        command += " " + fastq_1 + " " + fastq_2
        if verbose:
            logging.info(command)
        if os.path.exists(process["tool"]):
            os.system(command)
            success = True
    else:
        # Just return the reference to original files
        return [fastq_1, fastq_2]

    if success == False:
        # create some dummy files just for debugging purposes
        open(output_files[0], 'a').close()
        open(output_files[1], 'a').close()

    return [output_files[0], output_files[1]]

def invoke_align(sample, aligncount, tmp_dir, src_files, cleanup, verbose):
    '''
    Purpose : finds and removes adapter sequences
    Arguments: 
    sample      : one patient sample 
    aligncount  : dictionary of aligncount config arguments
    tmp_dir     : directory where temporary files will be created
    cleanup     : if set to 1 then remove temporary files (recommended to reduce storage)
    verbose     : set to 1 if you want verbose output
    '''
    
    # update the input and output file

    fastq_1 = sample["FASTQ_1"]
    sample_name = re.search('(\w+)_1', fastq_1).group(1)

    command = aligncount["tool"]
    command += " "
    command += aligncount["arguments"]

    if "kallisto" in aligncount["tool"]:
        command += " -i " + aligncount["index_file"]
        command += " " + src_files[0] + " " + src_files[1]
        command += " -o " + tmp_dir + "/" + sample_name
    elif "STAR" in aligncount["tool"]:
        command += " --genomeDir " + aligncount["gene_path"]
        command += " --genomeFastaFiles " + src_files[0] + " " + src_files[1]
        command += " --outFileNamePrefix " + tmp_dir + "/" + sample_name + "_"

    if verbose:
        logging.info(command)
    

    if os.path.exists(aligncount["tool"]):
        os.system(command)
        
    # Kallisto copy abudnace file to standard gene count output
    if "kallisto" in aligncount["tool"]:
        kallisto_abundance = tmp_dir + "/" + sample_name + "/abundance.tsv"
        output_genecount = tmp_dir + "/" + sample_name + "_ReadsPerGene.out.tab"
        
        # copy the kallisto file to standard location
        if os.path.exists(kallisto_abundance):
            copyfile(kallisto_abundance, output_genecount)
        if verbose==1:
            logging.info("cp " + kallisto_abundance + output_genecount)
        
        # remove the kallisto temporary files
        if cleanup == 1:
            if os.path.isdir(tmp_dir + "/" + sample_name):
                rmtree(tmp_dir + "/" + sample_name)
            if verbose:
                logging.info("rm " + tmp_dir + "/" + sample_name)


    # Remove this code, just creates a dummy count file gene count output
    if not os.path.exists(aligncount["tool"]):
        if ("GeneCounts" in aligncount["arguments"]) or ("kallisto" in aligncount["tool"]):
            out_file_name = tmp_dir + "/" + sample_name + "_" + "ReadsPerGene.out.tab"
            fp = open(out_file_name, 'w')
            # STAR insert header
            if "kallisto" in aligncount["tool"]:
                fp.write("target_id\tlength\teff_length\test_counts\ttpm\n")
            elif "STAR" in aligncount["tool"]:
                fp.write("N_unmapped\t26098\t26098\t26098\n")
                fp.write("N_multimapping\t26098\t26098\t26098\n")
                fp.write("N_noFeature\t26098\t26098\t26098\n")
                fp.write("N_ambiguous\t26098\t26098\t26098\n")
            # Output a bunch of random genes
            for gene_num in range(1, 31):
                fp.write("NM_" + str(gene_num) + "\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\n")
            fp.close()
            if "STAR" in aligncount["tool"]:
                # Create Dummy Log File for STAR with PRUA
                out_file_name = tmp_dir + "/" + sample_name + "_" + "Log.final.out"
                fp = open(out_file_name, 'w')
                fp.write("Uniquely mapped reads % |       "+str(float(randint(0, 100)))+"%\n")
                fp.close()
                # Output optional BAM files
                if "TranscriptomeSAM" in aligncount["arguments"]:
                    out_file_name = tmp_dir + "/" + sample_name + "_" + "Aligned.bam"
                    open(out_file_name, 'a').close()
    return




def process_data(raw_dir, tmp_dir, out_dir, sra_runs, process, aligncount, cleanup, verbose):
    '''
    Purpose : Create data folders in working directory and copy source data files
    Arguments: 
    raw_dir     : directory where initial source data will be copied
    tmp_dir     : directory where temporary files will be created
    out_dir     : directory where final output files will be created
    sra_runs    : dictionary of sra_runs config arguments (sra database, filter)
    process     : dictionary of process config arguments (removes adapter sequences, e.g. cutadapt)
    aligncount  : dictionary of aligncount config arguments (finds and removes adapter sequences, e.g. STAR)
    cleanup     : if set to 1 then remove temporary fastq files (recommended to reduce storage)
    verbose : set to 1 if you want verbose output
    '''

    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Process + Align + Count")
    

    # Step 1: Create the data folders
    make_data_directories(raw_dir, tmp_dir, out_dir, verbose)

    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Process")


    # Step 2: Copy to Raw folder the SRA file
    if verbose == 1:
        logging.info("cp " + sra_runs["input_database"] + " " + sra_runs["output_database"])
    input_database = sra_runs["input_database"]
    if not os.path.exists(input_database):
        input_database = "../" +input_database 
    copyfile(input_database, sra_runs["output_database"])

    # Add Run column for download
    if ("curl" in aligncount["tool"]) and (sra_runs["input_database2"] != ""):
        df = pd.read_csv(sra_runs["output_database"])
        biosample_id1 = df["BIOSAMPLE NAME"].str[0:2]
        biosample_id2 = df["BIOSAMPLE NAME"].str[2:4]
        biosample_fluid = df["BIOSAMPLE NAME"].str[-3:]
        df["Sample Name"] = biosample_id1 + "_" + biosample_id2 + "_" + biosample_fluid

        # Open SRA database
        input_database2 = sra_runs["input_database2"]
        if not os.path.exists(input_database2):
            input_database2 = "../" +input_database2
        df_sra = pd.read_csv(input_database2)
        df_sra = df_sra[["Run", "sex", "Sample Name", "submitted_subject_id"]]
        
        # Open S1 database
        input_database3 = sra_runs["input_database3"]
        if not os.path.exists(input_database3):
            input_database3 = "../" +input_database3
        df_s1 = pd.read_csv(input_database3)
        df_s1["submitted_subject_id"] = df_s1["Subject ID"].str.replace("-", "_")

        df_s1 = pd.merge(df_s1, df_sra, on="submitted_subject_id")
        df_s1 = df_s1.drop(["submitted_subject_id", "Subject ID", "gender M=1", "ClinicalDXSummary", "Control", "AD", "PD"], axis=1)
        
        df = pd.merge(df, df_s1, on="Sample Name")
        df.to_csv(sra_runs["output_database"])


    # Step 3: Iterate through the samples and process or copy the file
    number = 1
    
    fastq1_path = process["fastq1_path"]
    fastq2_path = process["fastq2_path"]
    
    run_database = sra_runs["output_database"]
    # filter the samples which need processing
    filter_start_row = -1
    filter_num_rows = -1
    if sra_runs["filter_enable"] == 1:
        filter_start_row = sra_runs["filter_start_row"]
        filter_num_rows = sra_runs["filter_num_rows"]

    # load the run table
    df_run_samples = pd.read_csv(run_database)
    
    # Find the number of samples we need to process
    num_samples = df_run_samples.shape[0]
    # if we have a patient filter adjust the number we process
    if filter_num_rows != -1:
        num_samples = filter_num_rows

    sra_row_num = 0
    
    # Download Option
    if "curl" in aligncount["tool"]:
        for index, row in df_run_samples.iterrows():
            # If run filter exists then start on correct place
            if filter_start_row != -1 and sra_row_num < filter_start_row:
                sra_row_num += 1
                continue 
            # If run filter exists then finish on correct place
            if filter_num_rows != -1 and number > filter_num_rows:
                break
            
            if verbose:
                logging.info("# ---------------------------------------------------")
                logging.info("# Starting sample # " + str(number) + " out of " + str(num_samples))

            seed(99999 + number)  # Set the seed so it is always same random values

            # Ignore non passing cases
            if row["MEETS ERCC QC STANDARDS?"] != "PASS":
                biosample_dir = "./data/tmp/" + row["Run"]
                filename = biosample_dir + ".tgz"
                continue

            # Download
            url = row["DOWNLOAD URL"]
            biosample_dir = "./data/tmp/" + row["Run"]
            filename = biosample_dir + ".tgz"

            command = aligncount["tool"]
            command += " "
            command += aligncount["arguments"]
            command +=" "
            command += "-o " + filename
            command += " " + url

            if not os.path.exists(filename):
                if verbose==1:
                    logging.info(command)
                os.system(command)

            # Make dir and Unzip
            if not os.path.isdir(biosample_dir):
                if verbose:
                    logging.info("mkdir " + biosample_dir)
                if not "proxy" in aligncount["tool"]:
                    os.makedirs(biosample_dir)
            command = "tar -C " + biosample_dir + " -xzf " + filename
            if verbose==1:
                logging.info(command)
            if not "proxy" in aligncount["tool"]:
                os.system(command)
                # Extract Gene Count
                subfolder = [ f.path for f in os.scandir(biosample_dir) if f.is_dir() ][0]
            else:
                subfolder = biosample_dir + "/data"
            src_gene_count = subfolder + "/" + aligncount["read_counts_file"]
            # Copy Gene Count file
            dst_gene_count = biosample_dir + "_" + "ReadsPerGene.out.tab"
            if not "proxy" in aligncount["tool"]:
                copyfile(src_gene_count, dst_gene_count)
            if verbose==1:
                logging.info("cp " + src_gene_count + " " + dst_gene_count)

            # Generate pseudo random data
            if "proxy" in aligncount["tool"]:
                fp = open(dst_gene_count, 'w')
                fp.write("ReferenceID\tuniqueReadCount\ttotalReadCount\tmultimapAdjustedReadCount\tmultimapAdjustedBarcodeCount\n")
                # Output a bunch of random genes
                for gene_num in range(1, 31):
                    fp.write("tst-mir_" + str(gene_num) + ":test\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\t" + str(randint(0, 1000)) + "\n")
                fp.close()



            number += 1
        
    else:
        for run in df_run_samples["Run"]:
            # If run filter exists then start on correct place
            if filter_start_row != -1 and sra_row_num < filter_start_row:
                sra_row_num += 1
                continue 
            # If run filter exists then finish on correct place
            if filter_num_rows != -1 and number > filter_num_rows:
                break

            if verbose:
                logging.info("# ---------------------------------------------------")
                logging.info("# Starting sample # " + str(number) + " out of " + str(num_samples))

            seed(99999 + number)  # Set the seed so it is always same random values
                
            sample = {}
            sample["FASTQ_1"] = fastq1_path.replace("%run%", run)
            sample["FASTQ_2"] = fastq2_path.replace("%run%", run)
            
            # Step 2.1 : Process (ie create clean or copies of a pair sample)
            process_out_files = [None, None]
            process_out_files[0] = tmp_dir + "/out.1.fastq.gz"
            process_out_files[1] = tmp_dir + "/out.2.fastq.gz"
            process_out_files = invoke_process(sample, process, tmp_dir, process_out_files, verbose)

            # If we do not want to aligncount, then we can stop processing after first iteration
            if aligncount["enable"] != 1:
                break

            # Step 2.2 : Align (ie create a aligned BAM file)
            invoke_align(sample, aligncount, tmp_dir, process_out_files, cleanup, verbose)

            # Step 2.3 : Delete the tmp fastq.gz files from cutadapt step
            if process["enable"] == 1:
                if cleanup==1:
                    if verbose:
                        logging.info("rm " + process_out_files[0])
                        logging.info("rm " + process_out_files[1])
                    os.remove(process_out_files[0])        
                    os.remove(process_out_files[1])            
            number += 1

    if verbose:
        logging.info("# Finished")
        logging.info("# ---------------------------------------------------")
    return
        

def get_valid_genes(gene_name_filename, gene_names):
    '''
    Purpose : get LOCUS list of valid genes
    Arguments: 
    gene_name_filename     : file path of gene names file
    gene_names             : list of gene names to filter
    return list of LOCUS names
    '''

    # Check if file actually exists
    if not os.path.exists(gene_name_filename):
        return []
    df_Gene_Names = pd.read_csv(gene_name_filename)
    for gene in gene_names:
        df_Gene_Names = df_Gene_Names[ (df_Gene_Names["chr"] != gene)] 
    return df_Gene_Names["refseq"]


def process_merge_gene_counts(count, input_dir, cleanup, verbose):
    '''
    Purpose : merge all gene count files
    Arguments: 
    count           : dictionary of count merge config arguments
    input_dir       : Directory of files to merge
    cleanup         : if set to 1 then remove input files
    verbose : set to 1 if you want verbose output
    '''

    merged_count_df = pd.DataFrame()
    count_column = count["column_count"]
    processed_file = []

    run_database = count["run_database"]
    df_run_samples = pd.read_csv(run_database)
    
    if (count["format"] == "STAR") and ("PRUA" in count["features"]):
        df_run_samples["PRUA"] = -1.0

    skip_rows = count["skiprows"]
    if count["format"] != "STAR":
        skip_rows = 1
    
    number = 1
    indexes = set()
    print("Build Indexes")
    number_entries = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith(".tab"):
            continue
        sample_name = filename[0:filename.find("_")]
        if sample_name in count["skip_samples"]:
            # Ignore these samples as they are bad
            logging.info("Skipping sample: " + sample_name)
            df_run_samples = df_run_samples[df_run_samples["Run"] != sample_name]
            continue
        tab_file = input_dir + "/" + filename
        processed_file.append(tab_file)
        if verbose:
            logging.info("Analysing Input: " + filename)
        df = pd.read_csv(input_dir + "/" + filename, sep="\t", skiprows=skip_rows, header=None)
        df = df.set_index(0)
        indexes = indexes.union(df.index)
        number_entries += 1
    print("Finish Indexes")

    merged_count_df = pd.DataFrame(index=indexes)
    #print(indexes, len(indexes))

    
    # Now process each column in individual gene matrix files and append to merged gene matrix
    for filename in os.listdir(input_dir):
        if not filename.endswith(".tab"):
            continue
        sample_name = filename[0:filename.find("_")]
        if sample_name in count["skip_samples"]:
            # Ignore these samples as they are bad
            logging.info("Skipping sample: " + sample_name)
            df_run_samples = df_run_samples[df_run_samples["Run"] != sample_name]
            continue
        tab_file = input_dir + "/" + filename
        processed_file.append(tab_file)
        df = pd.read_csv(input_dir + "/" + filename, sep="\t", skiprows=skip_rows, header=None)
        
        df = df.set_index(0)   
        merged_count_df[sample_name] = df.iloc[:,count_column-1]
        print("Processing", sample_name, number, "of", number_entries)

        # For STAR open Log file to get PRUA (percentage of read uniquiely mapped) 
        if (count["format"] == "STAR") and ("PRUA" in count["features"]):
            PRUA = 0.0
            log_filename = tab_file.replace("ReadsPerGene.out.tab", "Log.final.out")
            with open(log_filename) as f:
                lines = f.readlines()
            for line in lines:
                if "Uniquely mapped reads" in line:
                    PRUA = float(re.findall("[-+]?\d*\.\d+|\d+", line)[0])
                    break
            df_run_samples.loc[df_run_samples["Run"] == sample_name, "PRUA"] = PRUA
        number += 1

    
    # filter only selected features
    df_run_samples = df_run_samples[count["features"]]
    # impute any columns requested
    for col in count["imputes"]:
        df_run_samples[col].fillna((df_run_samples[col].mean()), inplace=True)
    # rename any columns requested
    df_run_samples = df_run_samples.rename(count["rename"], axis=1)
    # remove rows with no PRUA set (ie not processed)
    if (count["format"] == "STAR") and ("PRUA" in count["features"]):
        df_run_samples = df_run_samples[df_run_samples["PRUA"] != -1]
    
    # transpose so columns are runs, and rows are genes
    #merged_count_df = merged_count_df.transpose()
    # make sure column and row orders match - valid ones are the intersection
    valid_sra_runs = list(set(df_run_samples["Run"].tolist()).intersection( set(merged_count_df.columns.to_numpy()) ))
    df_run_samples = df_run_samples[df_run_samples["Run"].isin(valid_sra_runs)]
    merged_count_df = merged_count_df[ valid_sra_runs ]

    # Keep only genes passing filter
    if count["enable_filter"]==1:
        if verbose == 1:
            logging.info("Filtering Genes: Keep " + count["filter_keep_genes"])
        merged_count_df = merged_count_df[merged_count_df.index.astype(str).str.startswith(count["filter_keep_genes"])]
        # Drop genes (chromosomes etc)
        if "filter_names" in count:
            filter_genes = get_valid_genes(count["filter_names"], count["filter_remove_genes"])
            if len(filter_genes) > 0:
                # Find the difference - these will be what we remove
                genes_to_remove = set(merged_count_df.index) - set(filter_genes)
                if verbose==1:
                    logging.info("Filtering Genes: Remove " + str(count["filter_remove_genes"]) + " Number=" + str(len(genes_to_remove)))
                    logging.info("Before Filter #genes=" + str(merged_count_df.shape[0]))
                merged_count_df = merged_count_df.drop(list(genes_to_remove), axis=0, errors='ignore')
                if verbose==1:
                    logging.info("After Filter #genes=" + str(merged_count_df.shape[0]))

    # rename and drop nan's
    df_run_samples = df_run_samples.replace(count["replace"]["from"], count["replace"]["to"])
    merged_count_df.index.name = ""
    
    if verbose == 1:
        print("Before Drop Nan size=", merged_count_df.shape[0])
        merged_count_df.to_csv("data/out/gene_matrix_full.tsv", sep='\t')
    
    # filter out miRNA
    #merged_count_df = merged_count_df[merged_count_df.index.str.find("miRNA") != -1]
    
    # code to combine miRNAs
    merged_count_df["miRNAs"] = merged_count_df.index.str.split(':').str[0].str[4:].str.lower().str.strip("-")
    merged_count_df = merged_count_df[merged_count_df["miRNAs"].str.startswith("mir")]
    merged_count_df = merged_count_df.groupby("miRNAs").sum()
    merged_count_df = merged_count_df.replace({0:np.nan})
    # Drop NA's and Fill
    merged_count_df = merged_count_df.dropna(thresh=count["thresh"]).fillna(1)

    if verbose == 1:
        print("After Drop Nan size=", merged_count_df.shape[0])
    
    # save files
    merged_count_df.to_csv(count["output_matrix"], sep='\t')
    df_run_samples.to_csv(count["output_features"], sep='\t', index=False)

    if verbose:
        logging.info("Output: " + count["output_matrix"] + " " + count["output_features"])

    # remove TAB input files
    if cleanup==1:
        for tab_file in processed_file:
            if verbose:
                logging.info("rm " + tab_file)
            os.remove(tab_file)
    
    return


def process_merge_bam(bam, input_dir, cleanup, verbose):
    '''
    Purpose : merge bam files
    Arguments: 
    bam             : dictionary of bam merge config arguments
    input_dir       : Directory of files to merge
    cleanup         : if set to 1 then remove input files
    verbose : set to 1 if you want verbose output
    '''
    
    command = bam["tool"]
    command += " "
    command += bam["arguments"]
    command += " "
    command += bam["output"]
    command += " "
    
    processed_file = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".bam"):
            processed_file.append(input_dir + "/" + filename)

            if cleanup==1:
                # Only process first file as BAM files are big and 
                # we only need to do a quality check on one
                break
       
    
    command += " ".join(processed_file)
    if verbose:
        logging.info(command)
    
    if os.path.exists(bam["tool"]):
        os.system(command)
    else:
        # Dummy code, Remove this code, just creates a fake output
        open(bam["output"], 'a').close()

    # remove bam input files
    if cleanup==1:
        for bam in processed_file:
            if verbose:
                logging.info("rm " + bam)
            os.remove(bam)

    return
        
 
def process_merge(count, bam, input_dir, cleanup, verbose):
    '''
    Purpose : merge all gene count files
    Arguments: 
    count           : dictionary of count merge config arguments
    bam             : dictionary of bam merge config arguments
    input_dir       : Directory of files to merge
    cleanup         : if set to 1 then remove input files
    verbose : set to 1 if you want verbose output
    '''

    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Merge")


    # Merge Gene Counts
    if count["enable"] == 1:
        process_merge_gene_counts(count, input_dir, cleanup, verbose)
             
    # Merge BAM merge
    if bam["enable"] == 1:
        process_merge_bam(bam, input_dir, cleanup, verbose)
        

    if verbose:
        logging.info("# Finished")
        logging.info("# ---------------------------------------------------")
    return


def process_normalize(deseq2, output_dir, cleanup, verbose):
    '''
    Purpose : normalize merged BAM script using R's DESeq2
    Arguments: 
    deseq2          : dictionary of normalize deseq2 config arguments
    output_dir      : output dir for all files (normalized matrix)
    cleanup         : if set to 1 then remove merged bam input file
    verbose : set to 1 if you want verbose output
    '''
    
    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Normalize")
    
    if os.path.exists(deseq2["Rscript"]):
        command = deseq2["Rscript"] + " "
    else:
        command = "Rscript "
    command += deseq2["source"]
    command += " "
    command += deseq2["input_counts"]
    command += " "
    command += deseq2["input_features"]
    command += " "
    command += output_dir + "/"
    if verbose:
        logging.info(command)
    os.system(command) 

    # Find Top Genes
    # Load the Normalized Count Matrix
    df_normalized_counts = pd.read_csv(output_dir + "/vst_transformed_counts.tsv", sep="\t", index_col=0)
    def l1_from_mean(row):
        return abs(row-row.mean()).sum()
    # Create a Spread Statistics from Mean
    df_top_genes = df_normalized_counts.apply(lambda row: l1_from_mean(row), axis=1)
    # Sort top ones based on highest spread
    df_top_genes = df_top_genes.sort_values(ascending=False)
    df_top_genes = df_top_genes.reset_index().rename({"index":"gene", 0:"Spread"}, axis=1)
    # Save out top genes table
    df_top_genes.to_csv(output_dir + "/top_genes.tsv", sep="\t")

    max_genes = deseq2["max_genes"]

    top_genes = df_top_genes["gene"][0:max_genes]
    # Filter based top genes in normalized count
    if verbose:
        logging.info("Filtering PCA on normalized counts")
    df_normalized = pd.read_csv(output_dir + "/normalized_counts.tsv", sep="\t", index_col=0)
    if max_genes < df_normalized.shape[0]:
        df_normalized = df_normalized.loc[top_genes,:]
    df_normalized.to_csv(output_dir + "/pca_normalized_counts.tsv", sep='\t')
    # Filter PCA based top genes in VST normalized count
    if verbose:
        logging.info("Filtering PCA on VST normalized counts")
    df_vst_normalized = pd.read_csv(output_dir + "/vst_transformed_counts.tsv", sep="\t", index_col=0)
    if max_genes < df_vst_normalized.shape[0]:
        df_vst_normalized = df_vst_normalized.loc[top_genes,:]
    df_vst_normalized.to_csv(output_dir + "/pca_vst_transformed_counts.tsv", sep='\t')
    

    if verbose:
        logging.info("# Finished")
        logging.info("# ---------------------------------------------------")
    return

