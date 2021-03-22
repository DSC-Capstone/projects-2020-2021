#!/usr/bin/env python
# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020

import os 
import sys
import logging

def fastqc(fastq, bam, inputs, outdir, verbose):
    '''
    Purpose : Produces a quality report on the fastqc files
    Arguments:
    fastq       : dictionary of fastq config arguments
    bam         : dictionary of bam config arguments
    inputs      : The fastqc file that is requested to be analysed
    outdir      : The output directory where report will be generated
    verbose     : set to 1 if you want verbose output
    '''

    if verbose:
        logging.info("# ---------------------------------------------------")
        logging.info("# Quality Check")

    # fastq quality check
    if fastq["enable"] == 1:
        for filename in os.listdir(inputs):
            if filename.endswith(".fastq.gz"):
                command = fastq["tool"]
                command += " "
                command += inputs + "/" + filename
                command += " --outdir="+outdir
                if fastq["extract"] == 1:
                    command += " --extract"
                if verbose:
                    logging.info(command)
                if os.path.exists(fastq["tool"]):
                    os.system(command)
                else:
                    logging.info("error " + fastq["tool"] + " not installed")

    # bam quality check
    if bam["enable"] == 1:
        for filename in os.listdir(inputs):
            if filename.endswith(".bam"):
                command = bam["tool"]
                command += " -jar" + jar["arguments"]
                command += " INPUT=" + inputs + "/" + filename
                command += " OUTPUT=" + outdir + "/" + filename + ".txt"
                
                if verbose:
                    logging.info(command)
                if os.path.exists(jar["arguments"]):
                    os.system(command)
                else:
                    logging.info("error " + bam["tool"] + " not installed")

    if verbose:
        logging.info("# Finished")
        logging.info("# ---------------------------------------------------")
