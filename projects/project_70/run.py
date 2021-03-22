#!/usr/bin/env python
# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020

import sys
import json
import os
import logging


# Add the paths to the source folders
sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/visualization')
sys.path.insert(0, 'src/quality')

# import the src modules
import etl 
import analysis 
import visualize 
import qc


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'.  
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    logging.basicConfig(filename="log.txt", filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    #logging.getLogger('genetics').setLevel(logging.DEBUG)


    success = False

    if ('clean' in targets) or ('all' in targets):
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        raw_dir = data_cfg["raw_data_directory"]
        tmp_dir = data_cfg["tmp_data_directory"]
        out_dir = data_cfg["out_data_directory"]

        etl.clean_data(raw_dir, tmp_dir, out_dir, data_cfg["verbose"])
        success = True

    if ('test-data' in targets) or ('test' in targets):
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # USE THE TEST DATA!
        sra_runs = data_cfg["sra_runs"]
        sra_runs["input_database"] = "test/testdata/SraRunTable.csv"
        sra_runs["input_database2"] = ""
        sra_runs["input_database3"] = ""

        # get the directories
        raw_dir = data_cfg["raw_data_directory"]
        tmp_dir = data_cfg["tmp_data_directory"]
        out_dir = data_cfg["out_data_directory"]
        # get the process tool 
        process = data_cfg["process"]
        # get the aligncount tool
        aligncount = data_cfg[data_cfg["aligncount"]]
        cleanup = data_cfg["cleanup"]   

        # If we are testing then we do not execute alignment tool as its too slow
        # Instead we do a procy generation of pseudo-generated random counts
        if ('test-data' in targets) or ('test' in targets):
            data_cfg["kallisto"]["tool"] = "kallisto-proxy"
            data_cfg["STAR"]["tool"] = "STAR-proxy"
            data_cfg["download"]["tool"] = "curl-proxy"


        data = etl.process_data(raw_dir, tmp_dir, out_dir, sra_runs, process, aligncount, cleanup, data_cfg["verbose"])
        success = True


    # data step, process and align
    if ('data' in targets) or ('all' in targets):
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # get the directories
        raw_dir = data_cfg["raw_data_directory"]
        tmp_dir = data_cfg["tmp_data_directory"]
        out_dir = data_cfg["out_data_directory"]
        # get the SRA info
        sra_runs = data_cfg["sra_runs"]
        # get the process tool 
        process = data_cfg["process"]
        # get the aligncount tool
        aligncount = data_cfg[data_cfg["aligncount"]]
        cleanup = data_cfg["cleanup"]

        # Reset the input database to test 
        if 'test' in targets:
            sra_runs["input_database"] = "test/testdata/SraRunTable.csv"
            sra_runs["input_database2"] = ""
            sra_runs["input_database3"] = ""
           
        data = etl.process_data(raw_dir, tmp_dir, out_dir, sra_runs, process, aligncount, cleanup, data_cfg["verbose"])
        success = True


    # merge
    if ('merge' in targets) or ('all' in targets) or ('test' in targets):
        with open('config/merge-params.json') as fh:
            merge_cfg = json.load(fh)

        # get the process tool 
        count = merge_cfg["count"]
        bam = merge_cfg["bam"]
        input_dir = merge_cfg["input"]
        cleanup = merge_cfg["cleanup"]

        # Don't filter for test
        if 'test' in targets:
            count["filter_names"] = ""
  
        data = etl.process_merge(count, bam, input_dir, cleanup, merge_cfg["verbose"])
        success = True


    # normalize
    if ('normalize' in targets) or ('all' in targets) or ('test' in targets):
        with open('config/normalize-params.json') as fh:
            normalize_cfg = json.load(fh)

        deseq2 = normalize_cfg["DESeq2"]
        output_dir = normalize_cfg["output_dir"]
        cleanup = normalize_cfg["cleanup"]
           
        data = etl.process_normalize(deseq2, output_dir, cleanup, normalize_cfg["verbose"])
        success = True


    if "qc" in targets:
        with open('config/qc-params.json') as fh:
            data_quality_cfg = json.load(fh)
        inputs = data_quality_cfg["inputs"]
        outdir = data_quality_cfg["outdir"]
        fastq = data_quality_cfg["fastq"]
        bam = data_quality_cfg["bam"]
        
        qc.fastqc(fastq, bam, inputs, outdir, data_quality_cfg["verbose"])
        success = True

    
    if ('analysis' in targets) or ('all' in targets) or ('test' in targets):
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        deseq2 = analysis_cfg["DESeq2"]
        output_prefix = analysis_cfg["output_prefix"]
        cleanup = analysis_cfg["cleanup"]

        # Limit analysis for test to one brain region/disorder
        if 'test' in targets:
            #deseq2["biofluid_regions"] = ["AnCg"]
            #deseq2["disorders"] = ["Major Depression"]
            deseq2["parallel"] = 0
        
           
        data = analysis.process_analysis(deseq2, output_prefix, cleanup, analysis_cfg["verbose"])
        success = True


    if ('visualize' in targets) or ('all' in targets) or ('test' in targets):
        with open('config/visualize-params.json') as fh:
            visualize_cfg = json.load(fh)
        out_dir = visualize_cfg["out_dir"]
        plot_path = visualize_cfg["plot_path"]
        gene_hist = visualize_cfg["gene_hist"]
        missing_plot = visualize_cfg["missing_plot"]
        sra_lm = visualize_cfg["sra_lm"]
        ma_plot = visualize_cfg["ma_plot"]
        heat_map = visualize_cfg["heat_map"]
        histogram = visualize_cfg["histogram"]
        corrmatrix = visualize_cfg["corrmatrix"]
        venn = visualize_cfg["venn"]
        volcano = visualize_cfg["volcano"]
        box_all = visualize_cfg["box_all"]
        reg_corr = visualize_cfg["reg_corr"]

        # Limit visualization for test
        if 'test' in targets:
            histogram["ylim"] = 10
            venn["pvalue_cutoff"] = 0.5
            box_all["enable"] = 0 
            reg_corr["enable"] = 0
            
          
        data = visualize.process_plots(out_dir, plot_path, gene_hist, missing_plot, sra_lm, ma_plot, heat_map, histogram, corrmatrix, venn, volcano, box_all, reg_corr, visualize_cfg["verbose"])
        success = True


    if 'report' in targets:
        with open('config/report-params.json') as fh:
            report_cfg = json.load(fh)
        command = report_cfg["tool"]
        command += " " + report_cfg["args"]
        os.system(command)
        if report_cfg["verbose"] == 1:
            logging.info(command)
        success = True

    if success == False:
        logging.error(str(targets) + " not found")

    return


if __name__ == '__main__':
    # run via:
    # python main.py data 

    # TODO - Add help option

    targets = sys.argv[1:]
    main(targets)
