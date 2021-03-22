#! /usr/bin/env python
import sys
import os
import json
sys.path.insert(0, 'src')
from etl import run_create_data_rscript
from analysis import (run_diff_exp_rscript, run_comparison_rscript,
                      run_statistics_rscript, generate_graphs)
from test import run_test_rscript
from real import run_real_rscript
from real import run_real_diff_exp_rscript
from real import run_comparison_real
import logging
from datetime import datetime

logging.basicConfig(filename="log.txt", filemode='a',
 datefmt='%H:%M:%S',
 level=logging.INFO)

def main(targets):
    #Create synthetic data with parameters represents different number of samples per condition, truly differentially expressed genes, etc.
    if 'build' in targets:
        with open('config/build-params.json') as fh:
            data_cfg = json.load(fh)

        #All the synthetic data below is created with 12,500 genes
        for i in range(1, 11):

            #Create baseline synthetic data #1 with 0 differentially expressed genes for 2 samples per condition 10x
            print("Generating baseline synthetic data v" + str(i) + " with 0 differentially expressed genes for 2 samples per condition")
            baseline0_0_2 = run_create_data_rscript(data_cfg.get("b0_0_2_v" + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get("output_fileB0_0_2_v" + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #1 with 0 differentially expressed genes for 5 samples per condition 10x
            print("Generating baseline synthetic data v" + str(i) + " with 0 differentially expressed genes for 5 samples per condition")
            baseline0_0_5 = run_create_data_rscript(data_cfg.get("b0_0_5_v" + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get("output_fileB0_0_5_v" + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #1 with 0 differentially expressed genes for 10 samples per condition 10x
            print("Generating baseline synthetic data v" + str(i) + " with 0 differentially expressed genes for 10 samples per condition")
            baseline0_0_10 = run_create_data_rscript(data_cfg.get("b0_0_10_v" + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get("output_fileB0_0_10_v" + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #2 with 1250 differentially expressed genes with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 2 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 2 samples per condition")
            baseline1250_0_2 = run_create_data_rscript(data_cfg.get('b1250_0_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB1250_0_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #2 with 1250 differentially expressed genes with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 5 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 5 samples per condition")
            baseline1250_0_5 = run_create_data_rscript(data_cfg.get('b1250_0_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB1250_0_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #2 with 1250 differentially expressed genes with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 10 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 1250 upregulated in condition 1 and 0 downregulated in condition 2 for 10 samples per condition")
            baseline1250_0_10 = run_create_data_rscript(data_cfg.get('b1250_0_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB1250_0_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #3 with 1250 differentially expressed genes with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 2 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 625 upregulated in condition 1 and 625 downregulated in condition 2 for 2 samples per condition")
            baseline625_625_2 = run_create_data_rscript(data_cfg.get('b625_625_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB625_625_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #3 with 1250 differentially expressed genes with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 5 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 625 upregulated in condition 1 and 625 downregulated in condition 2 for 5 samples per condition")
            baseline625_625_5 = run_create_data_rscript(data_cfg.get('b625_625_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB625_625_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #3 with 1250 differentially expressed genes with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 10 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 625 upregulated in condition 1 and 625 downregulated in condition 2 for 10 samples per condition")
            baseline625_625_10 = run_create_data_rscript(data_cfg.get('b625_625_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB625_625_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #4 with 4000 differentially expressed genes with 4000 upregulated in condition 1 & 0 in condition 2 for 2 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 4000 upregulated in condition 1 and 0 downregulated in condition 2 for 2 samples per condition")
            baseline4000_0_2 = run_create_data_rscript(data_cfg.get('b4000_0_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB4000_0_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #4 with 4000 differentially expressed genes with 4000 upregulated in condition 1 & 0 in condition 2 for 5 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 4000 upregulated in condition 1 and 0 downregulated in condition 2 for 5 samples per condition")
            baseline4000_0_5 = run_create_data_rscript(data_cfg.get('b4000_0_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB4000_0_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #4 with 4000 differentially expressed genes with 4000 upregulated in condition 1 & 0 in condition 2 for 10 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 4000 upregulated in condition 1 and 0 downregulated in condition 2 for 10 samples per condition")
            baseline4000_0_10 = run_create_data_rscript(data_cfg.get('b4000_0_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB4000_0_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #5 with 4000 differentially expressed genes with 2000 upregulated in condition 1 & 2000 in condition 2 for 2 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 2000 upregulated in condition 1 & 2000 in condition 2 for 2 samples per condition")
            baseline2000_2000_2 = run_create_data_rscript(data_cfg.get('b2000_2000_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB2000_2000_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #5 with 4000 differentially expressed genes with 2000 upregulated in condition 1 & 2000 in condition 2 for 5 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 2000 upregulated in condition 1 & 2000 in condition 2 for 5 samples per condition")
            baseline2000_2000_5 = run_create_data_rscript(data_cfg.get('b2000_2000_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB2000_2000_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create baseline synthetic data #5 with 4000 differentially expressed genes with 2000 upregulated in condition 1 & 2000 in condition 2 for 10 samples per condition
            print("Generating baseline synthetic data v" + str(i) + " with 2000 upregulated in condition 1 & 2000 in condition 2 for 10 samples per condition")
            baseline2000_2000_10 = run_create_data_rscript(data_cfg.get('b2000_2000_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp4000'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileB2000_2000_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #6 whose counts were drawn from poisson distribution with 0 genes differentially expressed for 2 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 0 differentially expressed for 2 samples per condition")
            poisson0_0_2 = run_create_data_rscript(data_cfg.get('p0_0_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP0_0_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #6 whose counts were drawn from poisson distribution with 0 genes differentially expressed for 5 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 0 differentially expressed for 5 samples per condition")
            poisson0_0_5 = run_create_data_rscript(data_cfg.get('p0_0_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP0_0_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #6 whose counts were drawn from poisson distribution with 0 genes differentially expressed for 10 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 0 differentially expressed for 10 samples per condition")
            poisson0_0_10 = run_create_data_rscript(data_cfg.get('p0_0_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP0_0_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #7 whose counts were drawn from poisson distribution with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 2 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 2 samples per condition")
            poisson625_625_2 = run_create_data_rscript(data_cfg.get('p625_625_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP625_625_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #7 whose counts were drawn from poisson distribution with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 5 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 5 samples per condition")
            poisson625_625_5 = run_create_data_rscript(data_cfg.get('p625_625_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP625_625_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #7 whose counts were drawn from poisson distribution with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 10 samples per condition
            print("Generating poisson synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 10 samples per condition")
            poisson625_625_10 = run_create_data_rscript(data_cfg.get('p625_625_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('poisson_dispersion'), data_cfg.get('type1'), data_cfg.get('outlier0'), data_cfg.get('output_fileP625_625_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #8 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating single synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 2 samples per condition")
            single0_0_2 = run_create_data_rscript(data_cfg.get('s0_0_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS0_0_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #8 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating single synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 5 samples per condition")
            single0_0_5 = run_create_data_rscript(data_cfg.get('s0_0_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS0_0_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #8 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating single synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 10 samples per condition")
            single0_0_10 = run_create_data_rscript(data_cfg.get('s0_0_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS0_0_10_v' + str(i)), data_cfg.get('seqdepth'))

                #Create synthetic data #9 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating single synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 2 samples per condition")
            single625_625_2 = run_create_data_rscript(data_cfg.get('s625_625_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS625_625_2_v' + str(i)), data_cfg.get('seqdepth'))

                #Create synthetic data #9 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating single synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 5 samples per condition")
            single625_625_5 = run_create_data_rscript(data_cfg.get('s625_625_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS625_625_5_v' + str(i)), data_cfg.get('seqdepth'))

                #Create synthetic data #9 where fraction of genes for which we selected a single sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating single synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 10 samples per condition")
            single625_625_10 = run_create_data_rscript(data_cfg.get('s625_625_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type2'), data_cfg.get('single_count'), data_cfg.get('output_fileS625_625_10_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #10 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating random synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 2 samples per condition")
            random0_0_2 = run_create_data_rscript(data_cfg.get('r0_0_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR0_0_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #10 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating random synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 5 samples per condition")
            random0_0_5 = run_create_data_rscript(data_cfg.get('r0_0_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR0_0_5_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #10 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 0 differentially expressed genes
            print("Generating random synthetic data v" + str(i) + " with 0 upregulated in condition 1 & 0 downregulated in condition 2 for 10 samples per condition")
            random0_0_10 = run_create_data_rscript(data_cfg.get('r0_0_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp0'), data_cfg.get('upregulated_ratio1'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR0_0_10_v' + str(i)), data_cfg.get('seqdepth'))

             #Create synthetic data #11 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating random synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 2 samples per condition")
            random625_625_2 = run_create_data_rscript(data_cfg.get('r625_625_2_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond2'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR625_625_2_v' + str(i)), data_cfg.get('seqdepth'))

            #Create synthetic data #11 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating random synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 5 samples per condition")
            random625_625_5 = run_create_data_rscript(data_cfg.get('r625_625_5_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond5'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR625_625_5_v' + str(i)), data_cfg.get('seqdepth'))

             #Create synthetic data #11 where fraction of genes for which we selected a random sample and multiplied the corresponding count with a factor between 5 and 10 with 625 genes expressed in cond 1 & 625 in cond 2
            print("Generating random synthetic data v" + str(i) + " with 625 upregulated in condition 1 & 625 downregulated in condition 2 for 10 samples per condition")
            random625_625_10 = run_create_data_rscript(data_cfg.get('r625_625_10_v' + str(i)), data_cfg.get('n_vars'), data_cfg.get('samples_per_cond10'), data_cfg.get('repl_id'), data_cfg.get('n_diffexp1250'), data_cfg.get('upregulated_ratio_half'), data_cfg.get('regular_dispersion'), data_cfg.get('type3'), data_cfg.get('random_outlier'), data_cfg.get('output_fileR625_625_10_v' + str(i)), data_cfg.get('seqdepth'))



    #Run differential gene expression analysis tools on the synthetic data
    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        samples = [2, 5, 10]
        for j in samples:
            for i in range(1, 11):
                #Run DESeq2 on the synthetic datasets above
                #Run DESeq2 on baseline0_0 with 2, 5, 10 samples per cond
                deseq2_b0_0_start = datetime.now()
                deseq2_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_b0_0_' + str(j) + '_dir'))
                deseq2_b0_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_b0_0_end - deseq2_b0_0_start))

                #Run DESeq2 on baseline1250_0 with 2, 5, 10 samples per cond
                deseq2_b1250_0_start = datetime.now()
                deseq2_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_b1250_0_' + str(j) + '_dir'))
                deseq2_b1250_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_b1250_0_end - deseq2_b1250_0_start))

                #Run DESeq2 on baseline625_625 with 2, 5, 10 samples per cond
                deseq2_b625_625_start = datetime.now()
                deseq2_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_b625_625_' + str(j) + '_dir'))
                deseq2_b625_625_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(deseq2_b625_625_end - deseq2_b625_625_start))

                #Run DESeq2 on baseline4000_0 with 2, 5, 10 samples per cond
                deseq2_b4000_0_start = datetime.now()
                deseq2_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_b4000_0_' + str(j) + '_dir'))
                deseq2_b4000_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_b4000_0_end - deseq2_b4000_0_start))

                #Run DESeq2 on baseline2000_2000 with 2, 5, 10 samples per cond
                deseq2_b2000_2000_start = datetime.now()
                deseq2_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_b2000_2000_' + str(j) + '_dir'))
                deseq2_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(deseq2_b2000_2000_end - deseq2_b2000_2000_start))

                #Run DESeq2 on poisson0_0 with 2, 5, 10 samples per cond
                deseq2_p0_0_start = datetime.now()
                deseq2_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_p0_0_' + str(j) + '_dir'))
                deseq2_p0_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_p0_0_end - deseq2_p0_0_start))

                #Run DESeq2 on poisson625_625 with 2, 5, 10 samples per cond
                deseq2_p625_625_start = datetime.now()
                deseq2_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_p625_625_' + str(j) + '_dir'))
                deseq2_p625_625_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(deseq2_p625_625_end - deseq2_p625_625_start))

                #Run DESeq2 on single0_0 with 2, 5, 10 samples per cond
                deseq2_s0_0_start = datetime.now()
                deseq2_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_s0_0_' + str(j) + '_dir'))
                deseq2_s0_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for single0_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_s0_0_end - deseq2_s0_0_start))

                #Run DESeq2 on single625_625 with 2, 5, 10 samples per cond
                deseq2_s625_625_start = datetime.now()
                deseq2_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_s625_625_' + str(j) + '_dir'))
                deseq2_s625_625_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for single625_625_" + str(j) + "_v" + str(i) + " " + str(deseq2_s625_625_end - deseq2_s625_625_start))

                #Run DESeq2 on random0_0 with 2, 5, 10 samples per cond
                deseq2_r0_0_start = datetime.now()
                deseq2_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_r0_0_' + str(j) + '_dir'))
                deseq2_r0_0_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for random0_0_" + str(j) + "_v" + str(i) + " " + str(deseq2_r0_0_end - deseq2_r0_0_start))

                #Run DESeq2 on random625_625 with 2, 5, 10 samples per cond
                deseq2_r625_625_start = datetime.now()
                deseq2_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp1'), analysis_cfg.get('Rmdfunc1'), analysis_cfg.get('DESeq2_r625_625_' + str(j) + '_dir'))
                deseq2_r625_625_end = datetime.now()
                logging.info("Elapsed time for DESeq2 for random625_625_" + str(j) + "_v" + str(i) + " " + str(deseq2_r625_625_end - deseq2_r625_625_start))

                #Run edgeR on the synthetic datasets above

                #Run edgeR on baseline0_0 with 2, 5, 10 samples per cond
                edgeR_b0_0_start = datetime.now()
                edgeR_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_b0_0_' + str(j) + '_dir'))
                edgeR_b0_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_b0_0_end - edgeR_b0_0_start))

                #Run edgeR on baseline1250_0 with 2, 5, 10 samples per cond
                edgeR_b1250_0_start = datetime.now()
                edgeR_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_b1250_0_' + str(j) + '_dir'))
                edgeR_b1250_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_b1250_0_end - edgeR_b1250_0_start))

                #Run edgeR on baseline625_625 with 2, 5, 10 samples per cond
                edgeR_b625_625_start = datetime.now()
                edgeR_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_b625_625_' + str(j) + '_dir'))
                edgeR_b625_625_end = datetime.now()
                logging.info("Elapsed time for edgeR for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(edgeR_b625_625_end - edgeR_b625_625_start))

                #Run edgeR on baseline4000_0 with 2, 5, 10 samples per cond
                edgeR_b4000_0_start = datetime.now()
                edgeR_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_b4000_0_' + str(j) + '_dir'))
                edgeR_b4000_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_b4000_0_end - edgeR_b4000_0_start))

                #Run edgeR on baseline2000_2000 with 2, 5, 10 samples per cond
                edgeR_b2000_2000_start = datetime.now()
                edgeR_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_b2000_2000_' + str(j) + '_dir'))
                edgeR_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for edgeR for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(edgeR_b2000_2000_end - edgeR_b2000_2000_start))

                #             #Run edgeR on poisson0_0 with 2, 5, 10 samples per cond
                edgeR_p0_0_start = datetime.now()
                edgeR_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_p0_0_' + str(j) + '_dir'))
                edgeR_p0_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_p0_0_end - edgeR_p0_0_start))

                #Run edgeR on poisson625_625 with 2, 5, 10 samples per cond
                edgeR_p625_625_start = datetime.now()
                edgeR_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_p625_625_' + str(j) + '_dir'))
                edgeR_p625_625_end = datetime.now()
                logging.info("Elapsed time for edgeR for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(edgeR_p625_625_end - edgeR_p625_625_start))

                #Run edgeR on single0_0 with 2, 5, 10 samples per cond
                edgeR_s0_0_start = datetime.now()
                edgeR_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_s0_0_' + str(j) + '_dir'))
                edgeR_s0_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for single0_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_s0_0_end - edgeR_s0_0_start))

                #Run edgeR on single625_625 with 2, 5, 10 samples per cond
                edgeR_s625_625_start = datetime.now()
                edgeR_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_s625_625_' + str(j) + '_dir'))
                edgeR_s625_625_end = datetime.now()
                logging.info("Elapsed time for edgeR for single625_625_" + str(j) + "_v" + str(i) + " " + str(edgeR_s625_625_end - edgeR_s625_625_start))

                #Run edgeR on random0_0 with 2, 5, 10 samples per cond
                edgeR_r0_0_start = datetime.now()
                edgeR_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_r0_0_' + str(j) + '_dir'))
                edgeR_r0_0_end = datetime.now()
                logging.info("Elapsed time for edgeR for random0_0_" + str(j) + "_v" + str(i) + " " + str(edgeR_r0_0_end - edgeR_r0_0_start))

                #Run edgeR on random625_625 with 2, 5, 10 samples per cond
                edgeR_r625_625_start = datetime.now()
                edgeR_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp2'), analysis_cfg.get('Rmdfunc2'), analysis_cfg.get('edgeR_r625_625_' + str(j) + '_dir'))
                edgeR_r625_625_end = datetime.now()
                logging.info("Elapsed time for edgeR for random625_625_" + str(j) + "_v" + str(i) + " " + str(edgeR_r625_625_end - edgeR_r625_625_start))


                #Run voom.limma on the synthetic datasets above

                #Run voom.limma on baseline0_0 with 2, 5, 10 samples per cond
                voom_b0_0_start = datetime.now()
                voom_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_b0_0_' + str(j) + '_dir'))
                voom_b0_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(voom_b0_0_end - voom_b0_0_start))

                #Run voom.limma on baseline1250_0 with 2, 5, 10 samples per cond
                voom_b1250_0_start = datetime.now()
                voom_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_b1250_0_' + str(j) + '_dir'))
                voom_b1250_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(voom_b1250_0_end - voom_b1250_0_start))

                #Run voom.limma on baseline625_625 with 2, 5, 10 samples per cond
                voom_b625_625_start = datetime.now()
                voom_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_b625_625_' + str(j) + '_dir'))
                voom_b625_625_end = datetime.now()
                logging.info("Elapsed time for voom.limma for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(voom_b625_625_end - voom_b625_625_start))

                #Run voom.limma on baseline4000_0 with 2, 5, 10 samples per cond
                voom_b4000_0_start = datetime.now()
                voom_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_b4000_0_' + str(j) + '_dir'))
                voom_b4000_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(voom_b4000_0_end - voom_b4000_0_start))

                #Run voom.limma on baseline2000_2000 with 2, 5, 10 samples per cond
                voom_b2000_2000_start = datetime.now()
                voom_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_b2000_2000_' + str(j) + '_dir'))
                voom_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for voom.limma for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(voom_b2000_2000_end - voom_b2000_2000_start))

                #Run voom.limma on poisson0_0 with 2, 5, 10 samples per cond
                voom_p0_0_start = datetime.now()
                voom_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_p0_0_' + str(j) + '_dir'))
                voom_p0_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(voom_p0_0_end - voom_p0_0_start))

                #Run voom.limma on poisson625_625 with 2, 5, 10 samples per cond
                voom_p625_625_start = datetime.now()
                voom_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_p625_625_' + str(j) + '_dir'))
                voom_p625_625_end = datetime.now()
                logging.info("Elapsed time for voom.limma for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(voom_p625_625_end - voom_p625_625_start))

                #Run voom.limma on single0_0 with 2, 5, 10 samples per cond
                voom_s0_0_start = datetime.now()
                voom_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_s0_0_' + str(j) + '_dir'))
                voom_s0_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for single0_0_" + str(j) + "_v" + str(i) + " " + str(voom_s0_0_end - voom_s0_0_start))

                #Run voom.limma on single625_625 with 2, 5, 10 samples per cond
                voom_s625_625_start = datetime.now()
                voom_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_s625_625_' + str(j) + '_dir'))
                voom_s625_625_end = datetime.now()
                logging.info("Elapsed time for voom.limma for single625_625_" + str(j) + "_v" + str(i) + " " + str(voom_s625_625_end - voom_s625_625_start))

                #Run voom.limma on random0_0 with 2, 5, 10 samples per cond
                voom_r0_0_start = datetime.now()
                voom_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_r0_0_' + str(j) + '_dir'))
                voom_r0_0_end = datetime.now()
                logging.info("Elapsed time for voom.limma for random0_0_" + str(j) + " _v" + str(i) + " " + str(voom_r0_0_end - voom_r0_0_start))

                #Run voom.limma on random625_625 with 2, 5, 10 samples per cond
                voom_r625_625_start = datetime.now()
                voom_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp3'), analysis_cfg.get('Rmdfunc3'), analysis_cfg.get('voom.limma_r625_625_' + str(j) + '_dir'))
                voom_r625_625_end = datetime.now()
                logging.info("Elapsed time for voom.limma for random625_625_" + str(j) + "_v" + str(i) + " " + str(voom_r625_625_end - voom_r625_625_start))


                #Run NOISeq on the synthetic datasets above

                #Run NOISeq on baseline0_0 with 2, 5, 10 samples per cond
                NOISeq_b0_0_start = datetime.now()
                NOISeq_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_b0_0_' + str(j) + '_dir'))
                NOISeq_b0_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_b0_0_end - NOISeq_b0_0_start))

                #Run NOISeq on baseline1250_0 with 2, 5, 10 samples per cond
                NOISeq_b1250_0_start = datetime.now()
                NOISeq_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_b1250_0_' + str(j) + '_dir'))
                NOISeq_b1250_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_b1250_0_end - NOISeq_b1250_0_start))

                #Run NOISeq on baseline625_625 with 2, 5, 10 samples per cond
                NOISeq_b625_625_start = datetime.now()
                NOISeq_b625_625_ = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_b625_625_' + str(j) + '_dir'))
                NOISeq_b625_625_end = datetime.now()
                logging.info("Elapsed time for NOISeq for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(NOISeq_b625_625_end - NOISeq_b625_625_start))

                #Run NOISeq on baseline4000_0 with 2, 5, 10 samples per cond
                NOISeq_b4000_0_start = datetime.now()
                NOISeq_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_b4000_0_' + str(j) + '_dir'))
                NOISeq_b4000_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_b4000_0_end - NOISeq_b4000_0_start))

                #Run NOISeq on baseline2000_2000 with 2, 5, 10 samples per cond
                NOISeq_b2000_2000_start = datetime.now()
                NOISeq_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_b2000_2000_' + str(j) + '_dir'))
                NOISeq_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for NOISeq for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(NOISeq_b2000_2000_end - NOISeq_b2000_2000_start))

                #Run NOISeq on poisson0_0 with 2, 5, 10 samples per cond
                NOISeq_p0_0_start = datetime.now()
                NOISeq_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_p0_0_' + str(j) + '_dir'))
                NOISeq_p0_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_p0_0_end - NOISeq_p0_0_start))

                #Run NOISeq on poisson625_625 with 2, 5, 10 samples per cond
                NOISeq_p625_625_start = datetime.now()
                NOISeq_p625_625_ = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_p625_625_' + str(j) + '_dir'))
                NOISeq_p625_625_end = datetime.now()
                logging.info("Elapsed time for NOISeq for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(NOISeq_p625_625_end - NOISeq_p625_625_start))

                #Run NOISeq on single0_0 with 2, 5, 10 samples per cond
                NOISeq_s0_0_start = datetime.now()
                NOISeq_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_s0_0_' + str(j) + '_dir'))
                NOISeq_s0_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for single0_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_s0_0_end - NOISeq_s0_0_start))

                #Run NOISeq on single625_625 with 2, 5, 10 samples per cond
                NOISeq_s625_625_start = datetime.now()
                NOISeq_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_s625_625_' + str(j) + '_dir'))
                NOISeq_s625_625_end = datetime.now()
                logging.info("Elapsed time for NOISeq for single625_625_" + str(j) + "_v" + str(i) + " " + str(NOISeq_s625_625_end - NOISeq_s625_625_start))

                #Run NOISeq on random0_0 with 2, 5, 10 samples per cond
                NOISeq_r0_0_start = datetime.now()
                NOISeq_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_r0_0_' + str(j) + '_dir'))
                NOISeq_r0_0_end = datetime.now()
                logging.info("Elapsed time for NOISeq for random0_0_" + str(j) + "_v" + str(i) + " " + str(NOISeq_r0_0_end - NOISeq_r0_0_start))

                #Run NOISeq on random625_625 with 2, 5, 10 samples per cond
                NOISeq_r625_625_start = datetime.now()
                NOISeq_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp4'), analysis_cfg.get('Rmdfunc4'), analysis_cfg.get('NOISeq_r625_625_' + str(j) + '_dir'))
                NOISeq_r625_625_end = datetime.now()
                logging.info("Elapsed time for NOISeq for random625_625_" + str(j) + "_v" + str(i) + " " + str(NOISeq_r625_625_end - NOISeq_r625_625_start))


                #Run ttest on the synthetic datasets above

                #Run ttest on baseline0_0 with 2, 5, 10 samples per cond
                ttest_b0_0_start = datetime.now()
                ttest_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_b0_0_' + str(j) + '_dir'))
                ttest_b0_0_end = datetime.now()
                logging.info("Elapsed time for ttest for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(ttest_b0_0_end - ttest_b0_0_start))

                #Run ttest on baseline1250_0 with 2, 5, 10 samples per cond
                ttest_b1250_0_start = datetime.now()
                ttest_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_b1250_0_' + str(j) + '_dir'))
                ttest_b1250_0_end = datetime.now()
                logging.info("Elapsed time for ttest for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(ttest_b1250_0_end - ttest_b1250_0_start))

                #Run ttest on baseline625_625 with 2, 5, 10 samples per cond
                ttest_b625_625_start = datetime.now()
                ttest_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_b625_625_' + str(j) + '_dir'))
                ttest_b625_625_end = datetime.now()
                logging.info("Elapsed time for ttest for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(ttest_b625_625_end - ttest_b625_625_start))

                #Run ttest on baseline4000_0 with 2, 5, 10 samples per cond
                ttest_b4000_0_start = datetime.now()
                ttest_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_b4000_0_' + str(j) + '_dir'))
                ttest_b4000_0_end = datetime.now()
                logging.info("Elapsed time for ttest for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(ttest_b4000_0_end - ttest_b4000_0_start))

                #Run ttest on baseline2000_2000 with 2, 5, 10 samples per cond
                ttest_b2000_2000_start = datetime.now()
                ttest_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_b2000_2000_' + str(j) + '_dir'))
                ttest_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for ttest for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(ttest_b2000_2000_end - ttest_b2000_2000_start))

                #Run ttest on poisson0_0 with 2, 5, 10 samples per cond
                ttest_p0_0_start = datetime.now()
                ttest_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_p0_0_' + str(j) + '_dir'))
                ttest_p0_0_end = datetime.now()
                logging.info("Elapsed time for ttest for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(ttest_p0_0_end - ttest_p0_0_start))

                #Run ttest on poisson625_625 with 2, 5, 10 samples per cond
                ttest_p625_625_start = datetime.now()
                ttest_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_p625_625_' + str(j) + '_dir'))
                ttest_p625_625_end = datetime.now()
                logging.info("Elapsed time for ttest for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(ttest_p625_625_end - ttest_p625_625_start))

                #Run ttest on single0_0 with 2, 5, 10 samples per cond
                ttest_s0_0_start = datetime.now()
                ttest_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_s0_0_' + str(j) + '_dir'))
                ttest_s0_0_end = datetime.now()
                logging.info("Elapsed time for ttest for single0_0_" + str(j) + "_v" + str(i) + " " + str(ttest_s0_0_end - ttest_s0_0_start))

                #Run ttest on single625_625 with 2, 5, 10 samples per cond
                ttest_s625_625_start = datetime.now()
                ttest_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_s625_625_' + str(j) + '_dir'))
                ttest_s625_625_end = datetime.now()
                logging.info("Elapsed time for ttest for single625_625_" + str(j) + "_v" + str(i) + " " + str(ttest_s625_625_end - ttest_s625_625_start))

                #Run ttest on random0_0 with 2, 5, 10 samples per cond
                ttest_r0_0_start = datetime.now()
                ttest_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_r0_0_' + str(j) + '_dir'))
                ttest_r0_0_end = datetime.now()
                logging.info("Elapsed time for ttest for random0_0_" + str(j) + "_v" + str(i) + " " + str(ttest_r0_0_end - ttest_r0_0_start))

                #Run ttest on random625_625 with 2, 5, 10 samples per cond
                ttest_r625_625_start = datetime.now()
                ttest_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp5'), analysis_cfg.get('Rmdfunc5'), analysis_cfg.get('ttest_r625_625_' + str(j) + '_dir'))
                ttest_r625_625_end = datetime.now()
                logging.info("Elapsed time for ttest for random625_625_" + str(j) + "_v" + str(i) + " " + str(ttest_r625_625_end - ttest_r625_625_start))


                #Run PoissonSeq on the synthetic datasets above

                #Run PoissonSeq on baseline0_0 with 2, 5, 10 samples per cond
                PoissonSeq_b0_0_start = datetime.now()
                PoissonSeq_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_b0_0_' + str(j) + '_dir'))
                PoissonSeq_b0_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_b0_0_end - PoissonSeq_b0_0_start))

                #Run PoissonSeq on baseline1250_0 with 2, 5, 10 samples per cond
                PoissonSeq_b1250_0_start = datetime.now()
                PoissonSeq_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_b1250_0_' + str(j) + '_dir'))
                PoissonSeq_b1250_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_b1250_0_end - PoissonSeq_b1250_0_start))

                #Run PoissonSeq on baseline625_625 with 2, 5, 10 samples per cond
                PoissonSeq_b625_625_start = datetime.now()
                PoissonSeq_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_b625_625_' + str(j) + '_dir'))
                PoissonSeq_b625_625_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_b625_625_end - PoissonSeq_b625_625_start))

                #Run PoissonSeq on baseline4000_0 with 2, 5, 10 samples per cond
                PoissonSeq_b4000_0_start = datetime.now()
                PoissonSeq_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_b4000_0_' + str(j) + '_dir'))
                PoissonSeq_b4000_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_b4000_0_end - PoissonSeq_b4000_0_start))

                #Run PoissonSeq on baseline2000_2000 with 2, 5, 10 samples per cond
                PoissonSeq_b2000_2000_start = datetime.now()
                PoissonSeq_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_b2000_2000_' + str(j) + '_dir'))
                PoissonSeq_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_b2000_2000_end - PoissonSeq_b2000_2000_start))

                #Run PoissonSeq on poisson0_0 with 2, 5, 10 samples per cond
                PoissonSeq_p0_0_start = datetime.now()
                PoissonSeq_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_p0_0_' + str(j) + '_dir'))
                PoissonSeq_p0_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_p0_0_end - PoissonSeq_p0_0_start))

                #Run PoissonSeq on poisson625_625 with 2, 5, 10 samples per cond
                PoissonSeq_p625_625_start = datetime.now()
                PoissonSeq_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_p625_625_' + str(j) + '_dir'))
                PoissonSeq_p625_625_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_p625_625_end - PoissonSeq_p625_625_start))

                #Run PoissonSeq on single0_0 with 2, 5, 10 samples per cond
                PoissonSeq_s0_0_start = datetime.now()
                PoissonSeq_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_s0_0_' + str(j) + '_dir'))
                PoissonSeq_s0_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for single0_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_s0_0_end - PoissonSeq_s0_0_start))

                #Run PoissonSeq on single625_625 with 2, 5, 10 samples per cond
                PoissonSeq_s625_625_start = datetime.now()
                PoissonSeq_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_s625_625_' + str(j) + '_dir'))
                PoissonSeq_s625_625_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for single625_625_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_s625_625_end - PoissonSeq_s625_625_start))

                #Run PoissonSeq on random0_0 with 2, 5, 10 samples per cond
                PoissonSeq_r0_0_start = datetime.now()
                PoissonSeq_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_r0_0_' + str(j) + '_dir'))
                PoissonSeq_r0_0_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for random0_0_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_r0_0_end - PoissonSeq_r0_0_start))

                #Run PoissonSeq on random625_625 with 2, 5, 10 samples per cond
                PoissonSeq_r625_625_start = datetime.now()
                PoissonSeq_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp6'), analysis_cfg.get('Rmdfunc6'), analysis_cfg.get('PoissonSeq_r625_625_' + str(j) + '_dir'))
                PoissonSeq_r625_625_end = datetime.now()
                logging.info("Elapsed time for PoissonSeq for random625_625_" + str(j) + "_v" + str(i) + " " + str(PoissonSeq_r625_625_end - PoissonSeq_r625_625_start))



                #Run ABSSeq on the synthetic datasets above

                #Run ABSSeq on baseline0_0 with 2, 5, 10 samples per cond
                ABSSeq_b0_0_start = datetime.now()
                ABSSeq_b0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_b0_0_' + str(j) + '_dir'))
                ABSSeq_b0_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for baseline0_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_b0_0_end - ABSSeq_b0_0_start))

                #Run ABSSeq on baseline1250_0 with 2, 5, 10 samples per cond
                ABSSeq_b1250_0_start = datetime.now()
                ABSSeq_b1250_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b1250_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_b1250_0_' + str(j) + '_dir'))
                ABSSeq_b1250_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for baseline1250_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_b1250_0_end - ABSSeq_b1250_0_start))

                #Run ABSSeq on baseline625_625 with 2, 5, 10 samples per cond
                ABSSeq_b625_625_start = datetime.now()
                ABSSeq_b625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_b625_625_' + str(j) + '_dir'))
                ABSSeq_b625_625_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for baseline625_625_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_b625_625_end - ABSSeq_b625_625_start))

                #Run ABSSeq on baseline4000_0 with 2, 5, 10 samples per cond
                ABSSeq_b4000_0_start = datetime.now()
                ABSSeq_b4000_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b4000_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_b4000_0_' + str(j) + '_dir'))
                ABSSeq_b4000_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for baseline4000_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_b4000_0_end - ABSSeq_b4000_0_start))

                #Run ABSSeq on baseline2000_2000 with 2, 5, 10 samples per cond
                ABSSeq_b2000_2000_start = datetime.now()
                ABSSeq_b2000_2000 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('b2000_2000_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_b2000_2000_' + str(j) + '_dir'))
                ABSSeq_b2000_2000_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for baseline2000_2000_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_b2000_2000_end - ABSSeq_b2000_2000_start))

                #Run ABSSeq on poisson0_0 with 2, 5, 10 samples per cond
                ABSSeq_p0_0_start = datetime.now()
                ABSSeq_p0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_p0_0_' + str(j) + '_dir'))
                ABSSeq_p0_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for poisson0_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_p0_0_end - ABSSeq_p0_0_start))

                #Run ABSSeq on poisson625_625 with 2, 5, 10 samples per cond
                ABSSeq_p625_625_start = datetime.now()
                ABSSeq_p625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('p625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_p625_625_' + str(j) + '_dir'))
                ABSSeq_p625_625_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for poisson625_625_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_p625_625_end - ABSSeq_p625_625_start))

                #Run ABSSeq on single0_0 with 2, 5, 10 samples per cond
                ABSSeq_s0_0_start = datetime.now()
                ABSSeq_s0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_s0_0_' + str(j) + '_dir'))
                ABSSeq_s0_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for single0_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_s0_0_end - ABSSeq_s0_0_start))

                #Run ABSSeq on single625_625 with 2, 5, 10 samples per cond
                ABSSeq_s625_625_start = datetime.now()
                ABSSeq_s625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('s625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_s625_625_' + str(j) + '_dir'))
                ABSSeq_s625_625_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for single625_625_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_s625_625_end - ABSSeq_s625_625_start))

                #Run ABSSeq on random0_0 with 2, 5, 10 samples per cond
                ABSSeq_r0_0_start = datetime.now()
                ABSSeq_r0_0 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r0_0_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_r0_0_' + str(j) + '_dir'))
                ABSSeq_r0_0_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for random0_0_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_r0_0_end - ABSSeq_r0_0_start))

                #Run ABSSeq on random625_625 with 2, 5, 10 samples per cond
                ABSSeq_r625_625_start = datetime.now()
                ABSSeq_r625_625 = run_diff_exp_rscript(analysis_cfg.get('in_dir'), analysis_cfg.get('r625_625_' + str(j) + '_v' + str(i)), analysis_cfg.get('diffExp7'), analysis_cfg.get('Rmdfunc7'), analysis_cfg.get('ABSSeq_r625_625_' + str(j) + '_dir'))
                ABSSeq_r625_625_end = datetime.now()
                logging.info("Elapsed time for ABSSeq for random625_625_" + str(j) + "_v" + str(i) + " " + str(ABSSeq_r625_625_end - ABSSeq_r625_625_start))




    if 'graph' in targets:
        with open('config/graph-params.json') as fh:
            graphing_cfg = json.load(fh)
        #probably should make use of a config file
        stats = run_statistics_rscript(graphing_cfg.get('diffExp1'), graphing_cfg.get('diffExp2'), graphing_cfg.get('diffExp3'), graphing_cfg.get('diffExp4'), graphing_cfg.get('diffExp5'), graphing_cfg.get('diffExp6'), graphing_cfg.get('data1'),
                                       graphing_cfg.get('data2'), graphing_cfg.get('data3'), graphing_cfg.get('data4'), graphing_cfg.get('data5'), graphing_cfg.get('data6'), graphing_cfg.get('data7'), graphing_cfg.get('data8'), graphing_cfg.get('data9'), graphing_cfg.get('data10'), graphing_cfg.get('data11'))
        #graph = generate_graphs()

    if 'compare' in targets:
        with open('config/comparison-params.json') as fh:
            compare_data_cfg = json.load(fh)

        comparison = run_comparison_rscript(compare_data_cfg.get('ABSSeq_dir'), compare_data_cfg.get('DESeq_dir'), compare_data_cfg.get('edgeR_dir'), compare_data_cfg.get('NOISeq_dir'), compare_data_cfg.get('ttest_dir'), compare_data_cfg.get('PoissonSeq_dir'), compare_data_cfg.get('voom_dir'), compare_data_cfg.get('out_dir'))



    if 'test' in targets:
        with open('config/test-params.json') as fh:
            t_data_cfg = json.load(fh)

        testing = run_test_rscript(**t_data_cfg)
        
        
        
    if 'real' in targets:
        with open('config/real-params.json') as fh:
            real_data_cfg = json.load(fh)
            
        #Create real datasets: schizo.rds, mdd.rds and bipolar.rds
        real = run_real_rscript(real_data_cfg["real_data_dir"])
        
        #Run DESeq2 on the real datasets

        #Run DESeq2 on Schizophrenia dataset
        deseq2_schizo_start = datetime.now()
        deseq2_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo_1'), real_data_cfg.get('diffExp1'), real_data_cfg.get('Rmdfunc1'), real_data_cfg.get('schizo_out'))
        deseq2_schizo_end = datetime.now()
        logging.info("Elapsed time for DESeq2 for schizo: " + str(deseq2_schizo_end - deseq2_schizo_start))

        #Run DESeq2 on Major Depressive Disorder dataset
        deseq2_mdd_start = datetime.now()
        deseq2_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd_1'), real_data_cfg.get('diffExp1'), real_data_cfg.get('Rmdfunc1'), real_data_cfg.get('mdd_out'))
        deseq2_mdd_end = datetime.now()
        logging.info("Elapsed time for DESeq2 for mdd: "  + str(deseq2_mdd_end - deseq2_mdd_start))

        #Run DESeq2 on Bipolar Disorder dataset
        deseq2_bipolar_start = datetime.now()
        deseq2_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar_1'), real_data_cfg.get('diffExp1'), real_data_cfg.get('Rmdfunc1'), real_data_cfg.get('bipolar_out'))
        deseq2_bipolar_end = datetime.now()
        logging.info("Elapsed time for DESeq2 for bipolar: " + str(deseq2_bipolar_end - deseq2_bipolar_start))
        
        
        #Run edgeR on the real datasets

        #Run edgeR on Schizophrenia dataset
        edgeR_schizo_start = datetime.now()
        edgeR_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo'), real_data_cfg.get('diffExp2'), real_data_cfg.get('Rmdfunc2'), real_data_cfg.get('schizo_out'))
        edgeR_schizo_end = datetime.now()
        logging.info("Elapsed time for edgeR for schizo: " + str(edgeR_schizo_end - edgeR_schizo_start))

        #Run edgeR on Major Depressive Disorder dataset
        edgeR_mdd_start = datetime.now()
        edgeR_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd'), real_data_cfg.get('diffExp2'), real_data_cfg.get('Rmdfunc2'), real_data_cfg.get('mdd_out'))
        edgeR_mdd_end = datetime.now()
        logging.info("Elapsed time for edgeR for mdd: " +  str(edgeR_mdd_end - edgeR_mdd_start))

        #Run edgeR on Bipolar Disorder dataset
        edgeR_bipolar_start = datetime.now()
        edgeR_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar'), real_data_cfg.get('diffExp2'), real_data_cfg.get('Rmdfunc2'), real_data_cfg.get('bipolar_out'))
        edgeR_bipolar_end = datetime.now()
        logging.info("Elapsed time for edgeR for bipolar: " + str(edgeR_bipolar_end - edgeR_bipolar_start))
        
  
        
        #Run voom.limma on the real datasets
            
        #Run voom.limma on Schizophrenia dataset
        voom_schizo_start = datetime.now()
        voom_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo'), real_data_cfg.get('diffExp3'), real_data_cfg.get('Rmdfunc3'), real_data_cfg.get('schizo_out'))
        voom_schizo_end = datetime.now()
        logging.info("Elapsed time for voom.limma for schizo: " + str(voom_schizo_end - voom_schizo_start))

        #Run voom.limma on Major Depressive Disorder dataset
        voom_mdd_start = datetime.now()
        voom_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd'), real_data_cfg.get('diffExp3'), real_data_cfg.get('Rmdfunc3'), real_data_cfg.get('mdd_out'))
        voom_mdd_end = datetime.now()
        logging.info("Elapsed time for voom.limma for : " + str(voom_mdd_end - voom_mdd_start))

        #Run voom.limma on Bipolar Disorder dataset
        voom_bipolar_start = datetime.now()
        voom_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar'), real_data_cfg.get('diffExp3'), real_data_cfg.get('Rmdfunc3'), real_data_cfg.get('bipolar_out'))
        voom_bipolar_end = datetime.now()
        logging.info("Elapsed time for voom.limma for bipolar: " + str(voom_bipolar_end - voom_bipolar_start))
         
        
        #Run ttest on the real datasets
            
        #Run ttest on Schizophrenia dataset
        ttest_schizo_start = datetime.now()
        ttest_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo'), real_data_cfg.get('diffExp5'), real_data_cfg.get('Rmdfunc5'), real_data_cfg.get('schizo_out'))
        ttest_schizo_end = datetime.now()
        logging.info("Elapsed time for ttest for schizo: " + str(ttest_schizo_end - ttest_schizo_start))

        #Run ttest on Major Depressive Disorder dataset
        ttest_mdd_start = datetime.now()
        ttest_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd'), real_data_cfg.get('diffExp5'), real_data_cfg.get('Rmdfunc5'), real_data_cfg.get('mdd_out'))
        ttest_mdd_end = datetime.now()
        logging.info("Elapsed time for ttest for mdd: " + str(ttest_mdd_end - ttest_mdd_start))

        #Run ttest on Bipolar Disorder dataset
        ttest_bipolar_start = datetime.now()
        ttest_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar'), real_data_cfg.get('diffExp5'), real_data_cfg.get('Rmdfunc5'), real_data_cfg.get('bipolar_out'))
        ttest_bipolar_end = datetime.now()
        logging.info("Elapsed time for ttest for bipolar: " + str(ttest_bipolar_end - ttest_bipolar_start))
        
        
        
        #Run PoissonSeq on the real datasets
        
        #Run PoissonSeq on Schizophrenia dataset
        PoissonSeq_schizo_start = datetime.now()
        PoissonSeq_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo'), real_data_cfg.get('diffExp6'), real_data_cfg.get('Rmdfunc6'), real_data_cfg.get('schizo_out'))
        PoissonSeq_schizo_end = datetime.now()
        logging.info("Elapsed time for PoissonSeq for schizo: " + str(PoissonSeq_schizo_end - PoissonSeq_schizo_start))

        #Run PoissonSeq on Major Depressive Disorder dataset
        PoissonSeq_mdd_start = datetime.now()
        PoissonSeq_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd'), real_data_cfg.get('diffExp6'), real_data_cfg.get('Rmdfunc6'), real_data_cfg.get('mdd_out'))
        PoissonSeq_mdd_end = datetime.now()
        logging.info("Elapsed time for PoissonSeq for mdd: " + str(PoissonSeq_mdd_end - PoissonSeq_mdd_start))

        #Run PoissonSeq on Bipolar Disorder dataset
        PoissonSeq_bipolar_start = datetime.now()
        PoissonSeq_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar'), real_data_cfg.get('diffExp6'), real_data_cfg.get('Rmdfunc6'), real_data_cfg.get('bipolar_out'))
        PoissonSeq_bipolar_end = datetime.now()
        logging.info("Elapsed time for PoissonSeq for bipolar: " + str(PoissonSeq_bipolar_end - PoissonSeq_bipolar_start))


        
        #Run ABSSeq on the real datasets
        
        #Run ABSSeq on Schizophrenia dataset
        ABSSeq_schizo_start = datetime.now()
        ABSSeq_schizo = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('schizo'), real_data_cfg.get('diffExp7'), real_data_cfg.get('Rmdfunc7'), real_data_cfg.get('schizo_out'))
        ABSSeq_schizo_end = datetime.now()
        logging.info("Elapsed time for ABSSeq for schizo: " + str(ABSSeq_schizo_end - ABSSeq_schizo_start))

        #Run ABSSeq on Major Depressive Disorder dataset
        ABSSeq_mdd_start = datetime.now()
        ABSSeq_mdd = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('mdd'), real_data_cfg.get('diffExp7'), real_data_cfg.get('Rmdfunc7'), real_data_cfg.get('mdd_out'))
        ABSSeq_mdd_end = datetime.now()
        logging.info("Elapsed time for ABSSeq for mdd: " + str(ABSSeq_mdd_end - ABSSeq_mdd_start))

        #Run ABSSeq on Bipolar Disorder dataset
        ABSSeq_bipolar_start = datetime.now()
        ABSSeq_bipolar = run_real_diff_exp_rscript(real_data_cfg.get('in_dir'), real_data_cfg.get('bipolar'), real_data_cfg.get('diffExp7'), real_data_cfg.get('Rmdfunc7'), real_data_cfg.get('bipolar_out'))
        ABSSeq_bipolar_end = datetime.now()
        logging.info("Elapsed time for ABSSeq for bipolar: " + str(ABSSeq_bipolar_end - ABSSeq_bipolar_start))
        
        
        #Comparing the tools performances on the real dataset
        comparison = run_comparison_real(real_data_cfg.get('schizo_out'), real_data_cfg.get('schizo_deseq'), real_data_cfg.get('schizo_edgeR'), real_data_cfg.get('schizo_ttest'), real_data_cfg.get('schizo_voom'), real_data_cfg.get('schizo_poisson'), real_data_cfg.get('schizo_abseq'), real_data_cfg.get('compare_out'))
        
        comparison = run_comparison_real(real_data_cfg.get('mdd_out'), real_data_cfg.get('mdd_deseq'), real_data_cfg.get('mdd_edgeR'), real_data_cfg.get('mdd_ttest'), real_data_cfg.get('mdd_voom'), real_data_cfg.get('mdd_poisson'), real_data_cfg.get('mdd_abseq'), real_data_cfg.get('compare_out'))
        
        comparison = run_comparison_real(real_data_cfg.get('bipolar_out'), real_data_cfg.get('bipolar_deseq'), real_data_cfg.get('bipolar_edgeR'), real_data_cfg.get('bipolar_ttest'), real_data_cfg.get('bipolar_voom'), real_data_cfg.get('bipolar_poisson'), real_data_cfg.get('bipolar_abseq'), real_data_cfg.get('compare_out'))
    return

if __name__ == '__main__':
    targets = sys.argv[1]
    main(targets)
