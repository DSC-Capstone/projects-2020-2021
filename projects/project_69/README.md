# Title: Genetic Overlap between Alzheimer's, Parkinson’s, and healthy patients

#### Capstone Project: Data Science DSC180B

#### Section B04: Genetics

#### Authors: Saroop Samra, Justin Lu, Xuanyu Wu

#### Date : 2/2/2021

### Overview

This repository code is for the replication project for the paper: Profiles of Extracellular miRNA in Cerebrospinal Fluid and Serum from Patients with Alzheimer’s and Parkinson’s Diseases Correlate with Disease Status and Features of Pathology (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094839). The data includes that miRNA sequences from tissues from two biofluids (serum and cerebrospinal fluid), and is from 69 patients with Alzheimer's disease, 67 with Parkinson's disease and 78 neurologically normal controls using next generation small RNA sequencing (NGS).


### Running the project

•	To install the dependencies, run the following command from the root directory of the project: 

    pip install -r requirements.txt


### target: data
•	To process the data, from the root project directory run the command:

    python3 run.py data

•   The data pipeline step takes the .fastq compressed files as input and then applies two transformations: process and align

•	This pipeline step also uses an additional CSV file that is the SRA run database, a sample looks like as follows:

    Run expired_age    CONDITION    BIOFLUID     
    SRR1568567  40  Parkinson's Disease Cerebrospinal 



•   The configuration files for the data step are stored in config/data-params.json. These include the parameters for the tools as well as the directories used for storing the raw, temporary and output files.

    "raw_data_directory": "./data/raw",
    "tmp_data_directory": "./data/tmp",
    "out_data_directory": "./data/out",

•   The configuration also includes an attribute to the SRA run input database (described above), and an attribute of where to store that in the data folder. Additional filter attributes are included for ease of use to avoid processing all patients, if this filter_enable is set it will only process a subset of SRA rows (filter_start_row to filter_start_row + filter_num_rows).

    "sra_runs" : {
        "input_database" : "/datasets/SRP046292/exRNA_Atlas_CORE_Results.csv",
        "input_database2" : "/datasets/SRP046292/SraRunTable.csv",
        "input_database3" : "/datasets/SRP046292/Table_S1.csv",
        "output_database" : "data/raw/exRNA_Atlas_CORE_Results.csv",
        "filter_enable" : 0,
        "filter_start_row" : 120,
        "filter_num_rows" : 10   
    },
    

•	An optional transformation of the data is "process" that uses the following data configuration below that will invoke cutadapt which finds and removes adapter sequences. The attributes include the adapters (r1 and r2) to identify the start and end of pairs are a JSON array. The attribute enable allows to disable this cleaning step, instead it will simply copy the paired files from the source dataset. The arguments attribute allows flexible setting of any additional attribute to the cutadapt process. Finally, we have two wildcard paths that indicate the location of the SRA fastq pair files (fastq1 and fastq2).

    "process" : {
        "enable" : 1,
        "tool" : "/opt/conda/bin/cutadapt",
        "r1_adapters" : ["AAAAA", "GGGG"],
        "r2_adapters" : ["CCCCC", "TTTT"],
        "arguments" : "--pair-adapters --cores=4",
        "fastq1_path" : "/datasets/srp073813/%run_1.fastq.gz", 
        "fastq2_path" : "/datasets/srp073813/%run_2.fastq.gz"
    },
    
•   The second transformation of the data is "aligncount" that can be set to either use download, STAR or Kallisto. The choice is controlled by the aligncount attribute:

    "aligncount" : "download",

•   download step will use the ftp location of the gzip file in the Sra table and download using the curl command and unzips and the extracts the readCounts_gencode_sense.txt which represents thae gene counts for the sample. 

    "download" : {
        "enable" : 1,
        "tool" : "curl",
        "arguments" : "-L -R",
        "read_counts_file" : "readCounts_gencode_sense.txt"
    },

•   kallisto uses the index_file attribute is the location of the directory of the reference genome, which for this replication project was GRCh37_E75. The arguments attribute allows flexible setting of any additional attribute to the kallisto process. Including the bootstaro samples.The attribute enable allows to disable this alignment step, this is useful for debugging the process prior step, for example, you can run quality checks on the processed fastq files before proceeding to alignment. 

    "kallisto" : {
        "enable" : 1,
        "tool" : "/opt/kallisto_linux-v0.42.4/kallisto",
        "index_file" : "/datasets/srp073813/reference/kallisto_transcripts.idx",
        "arguments" : "quant -b 8 -t 8"
    },

•   STAR uses the gene_path attribute is the location of the directory of the reference genome, which for this replication project was GRCh37_E75 as described in the reference_gene attribute. The arguments attribute allows flexible setting of any additional attribute to the STAR process. Including TranscriptomeSAM in the quantMode arguments will also output bam files. Additionally, the log file gets outputted which has PRUA (percentage of reads uniquely aligned). The attribute enable allows to disable this alignment step, this is useful for debugging the process prior step, for example, you can run quality checks on the processed fastq files before proceeding to alignment. 

    "STAR" : {
        "enable" : 1,
        "tool" : "/opt/STAR-2.5.2b/bin/Linux_x86_64_static/STAR",
        "reference_gene" : "GRCh37_E75",
        "gene_path" : "/path/to/genomeDir",
        "arguments" : "--runMode alignReads --quantMode GeneCounts --genomeLoad LoadAndKeep --readFilesCommand zcat --runThreadN 8"
    },




•   The process and align transformation work on each of the samples. After each sample iteration, the temporary fastq files will be deleted to reduce storage requirements.


•   Example processing:

    python3 run.py data

    # ---------------------------------------------------
    # Process
    # ---------------------------------------------------
    # ---------------------------------------------------
    # Starting sample # 1 out of 1
    # ---------------------------------------------------
    # Starting sample # 1 out of 343
    curl-proxy -L -R -o ./data/tmp/SRR1568613.tgz ftp://ftp.genboree.org/exRNA-atlas/grp/Extracellular%20RNA%20Atlas/db/exRNA%20Repository%20-%20hg19/file/exRNA-atlas/exceRptPipeline_v4.6.2/KJENS1-Alzheimers_Parkinsons-2016-10-17/sample_SAMPLE_1022_CONTROL_SER_fastq/CORE_RESULTS/sample_SAMPLE_1022_CONTROL_SER_fastq_KJENS1-Alzheimers_Parkinsons-2016-10-17_CORE_RESULTS_v4.6.2.tgz
    sh: curl-proxy: command not found
    mkdir ./data/tmp/SRR1568613
    tar -C ./data/tmp/SRR1568613 -xzf ./data/tmp/SRR1568613.tgz
    cp ./data/tmp/SRR1568613/data/readCounts_gencode_sense.txt ./data/tmp/SRR1568613_ReadsPerGene.out.tab
    # ---------------------------------------------------
    # Starting sample # 2 out of 343
    curl-proxy -L -R -o ./data/tmp/SRR1568457.tgz ftp://ftp.genboree.org/exRNA-atlas/grp/Extracellular%20RNA%20Atlas/db/exRNA%20Repository%20-%20hg19/file/exRNA-atlas/exceRptPipeline_v4.6.2/KJENS1-Alzheimers_Parkinsons-2016-10-17/sample_SAMPLE_0427_PD_CSF_fastq/CORE_RESULTS/sample_SAMPLE_0427_PD_CSF_fastq_KJENS1-Alzheimers_Parkinsons-2016-10-17_CORE_RESULTS_v4.6.2.tgz
    sh: curl-proxy: command not found
    mkdir ./data/tmp/SRR1568457
    tar -C ./data/tmp/SRR1568457 -xzf ./data/tmp/SRR1568457.tgz
    cp ./data/tmp/SRR1568457/data/readCounts_gencode_sense.txt ./data/tmp/SRR1568457_ReadsPerGene.out.tab
    # ---------------------------------------------------


### target: merge
•   To merge gene count and/or BAM files generated from the data target, from the root project directory run the command:

    python3 run.py merge

•   The configuration files for the data step are stored in config/count-params.json. These include the parameters for the count merge and bam merge and it's associated arguments.

•   The format attrbute informs if to process downlload, kallisto (or STAR) files. The gene counts are merged into a TSV file and as well as a feature table based on the SRA run table. Additional STAR attributes in the JSON allow you to specify skiprows used when processing the  gene count files as well as identifying the column from the  gene matrix file to use as the column used to. There is an additional imputes attribute that allows you to impute any column with missing data. The attributes also include an optional "filter_names" gene table used to remove genes as well as removing false-positive genes. Finally, we can rename the feature columns before we save out the feature table.

    "count" : {
        "enable" : 1,
        "format" : "download",
        "skiprows" : 4,
        "column_count" : 1,
        "skip_samples" : ["SRR1568391"],
        "enable_filter" : 0,
        "filter_keep_genes" : "NM_",
        "filter_remove_genes" : ["chrX", "chrY"],
        "filter_names" : "/datasets/srp073813/reference/Gene_Naming.csv",
        "run_database" : "data/raw/exRNA_Atlas_CORE_Results.csv",
        "imputes" : ["TangleTotal"],
        "features" : ["Run", "CONDITION", "expired_age", "BIOFLUID", "sex", "PMI", "sn_depigmentation", "Braak score", "TangleTotal", "Plaque density", "PlaqueTotal"],
        "rename" : {"CONDITION" : "Disorder", "BIOFLUID" : "Biofluid", "Braak score" : "Braak_Score", "Plaque density" : "Plaque_density"},
        "replace" : {"from":["Parkinson's Disease", "Alzheimer's Disease", "Cerebrospinal fluid", "Healthy Control"], "to":["Parkinson", "Alzheimer", "Cerebrospinal", "Control"]},
        "output_matrix" : "data/out/gene_matrix.tsv",
        "output_features" : "data/out/features.tsv"
    },

•   For bam merging, which should not be enabled by default, we use the "samtools" merge feature that takes all the BAM files and combine them into one merged BAM file. 


    "bam" : {
        "enable" : 0,
        "output" : "data/tmp/merged.bam",
        "tool" : "/usr/local/bin/samtools",
        "arguments" : "merge --threads 8"
    },


•   Example processing:

    python3 run.py merge

    # ---------------------------------------------------
    # Merge
    Input: SRR3438605_ReadsPerGene.out.tab
    Input: SRR3438604_ReadsPerGene.out.tab
    Output: data/out/gene_matrix.tsv data/out/features.tsv
    # Finished
    # ---------------------------------------------------



### target: normalize
•   To normalize the aligned merge counts, from the root project directory run the command:

    python3 run.py normalize

•   The configuration files for the data step are stored in config/normalize-params.json. 

•   We use a custom R script which uses the DESeq2 module to take the input merged gene counts and the experiment features and outputs two normalized counts files. The analysis is done for all samples in the SRA run table. The output_dir sets the output location for the normalized count matrix files. One file is the standard normalized counts using the DESeq2 module, and the second normalized count file is after a Variable Stablization Transform (LRT). We also have a "max_genes" attribute that will filter the genes and removes ones that have little to no variance across disorder vesus control.

•   The data JSON configuration file also holds an array of samples, a sample looks like as follows:
    
    {
        "output_dir" : "data/out",
        "DESeq2" : {
            "Rscript" : "/opt/conda/envs/r-bio/bin/Rscript",
            "source" : "src/data/normalize.r",
            "input_counts" : "data/out/gene_matrix.tsv",
            "input_features" : "data/out/features.tsv",
            "max_genes" : 8000
        },
        "cleanup" : 0,
        "verbose": 1
    }


•   Example processing:

    python3 run.py normalize

    # ---------------------------------------------------
    # Normalize
    Rscript  src/data/normalize.r data/out/gene_matrix.tsv data/out/features.tsv data/out/
    [1] "Output data/out/normalized_counts.tsv data/out/vst_transformed_counts.tsv"
    # Finished
    # ---------------------------------------------------


### target: analysis
•   To perform the analysis for the gene counts, from the root project directory run the command:

    python3 run.py analysis

•   The configuration files for the data step are stored in config/analysis-params.json. 

•   We use a custom R script which uses the DESeq2 module to take the input merged gene counts and the experiment features and outputs 2 sets of files for each biofluid region. Each biofluid region will compare a disorder versus Control. This will result in a total of 4 sets of files (2 biofluid regions x 2 disorder pair comparisons). Each output set includes a Likelihood Ratio Test (LRT) using the full and reduced model as specified in the attributes below as well as a MA-Plot and Heatmap. The additional attributes include the property of doing parallel processing for DESeq2.
    
    {
        "output_prefix" : "data/out/%biofluid_region%",
        "DESeq2" : {
            "Rscript" : "/opt/conda/envs/r-bio/bin/Rscript",
            "biofluid_regions" : ["Cerebrospinal", "Serum"],
            "disorders" : ["Parkinson", "Alzheimer"],
            "control" : "Control",
            "input_counts" : "data/out/pca_normalized_counts.tsv",
            "input_features" : "data/out/features.tsv",
            "source" : "src/analysis/analysis.r",
            "full" : "expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal+Disorder",
            "reduced" : "expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal",
            "parallel" : 0
        },
        "cleanup" : 0,
        "verbose": 1
    }


•   Example processing:

    python3 run.py analysis

    # ---------------------------------------------------
    # Analysis
    Cerebrospinal x Parkinson vs Control
    Rscript src/analysis/analysis.r data/out/Cerebrospinal/Parkinson/gene_matrix.tsv data/out/Cerebrospinal/Parkinson/features.tsv data/out/Cerebrospinal/Parkinson/ full=expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal+Disorder reduced=expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal charts=1 parallel=0


### target: visualize

•   The visualize pipeline step can be invoked as follows:

    python3 run.py visualize

•   The configuration files for the data step are stored in config/visualize-params.json. The output will include multiple sets of charts: Gene Spread Variance Histogram, SRA Linear Correlation between SRA chart, MA-Plot 2x2 chart, Heat Map 2x2 chart, 2x2 Histogram, 4x4 Correlation Matrix and a Disorder Venn Diagram. Each chart type has flexible settings to control the input and layout for the charts as shown below:

    "gene_hist" : {
        "enable" : 1,
        "max_genes" : 8000,
        "nbins" : 100,
        "title" : "Distribution of Genes Based on Spread Metric: All vs Top Genes"
    },
    "missing_plot" : {
        "enable" : 1,
        "title" : "Percentage of Missing Genes over"
    },
    "sra_lm" : {
        "enable" : 1,
        "sra" : ["SRR1568567", "SRR1568584"],
        "normalized_counts" : "data/out/normalized_counts.tsv",
        "vst_counts" : "data/out/vst_transformed_counts.tsv",
        "title" : "%sra% Regression Log(Norm) v VST counts"
    },
    "ma_plot" : {
        "enable" : 1,
        "biofluid_regions" : ["Cerebrospinal", "Serum"],
        "disorders" : ["Parkinson", "Alzheimer"],
        "src_image" : "MAplot.png",
        "title" : "MA Plot: Biofluid Region vs Disorder"
    },
    "heat_map" : {
        "enable" : 1,
        "biofluid_regions" : ["Cerebrospinal", "Serum"],
        "disorders" : ["Parkinson", "Alzheimer"],
        "src_image" : "heatmap.png",
        "title" : "Heat Map: Biofluid Region vs Disorder"
    },
    "histogram" : {
        "enable" : 1,
        "biofluid_regions" : ["Cerebrospinal", "Serum"],
        "disorders" : ["Parkinson", "Alzheimer"],
        "title" : "Histograms Differential Gene Expression vs Control",
        "ylim" : 55
    },
    "corrmatrix" : {
        "enable" : 1,
        "title" : "Spearman Correlations of log2 fold gene expression"
    },
    "venn" : {
        "enable" : 1,
        "biofluid_regions" : ["Cerebrospinal", "Serum"],
        "disorders" : ["Parkinson", "Alzheimer"],
        "pvalue_cutoff" : 0.05,
        "title" : "Venn Diagram Disorders"
    },


•   Example processing:

    python3 run.py visualize

    # ---------------------------------------------------
    # Visualize
    # Finished
    # ---------------------------------------------------


### target: qc

•   The quality pipeline step can be invoked as follows:

    python3 run.py qc

•   The configuration files for the data step are stored in config/qc-params.json. These include the parameters for the output directory where the quality HTML reports will be outputted. 

    "outdir" : "data/out",
    "inputs" : "data/tmp",

•   For fastq files, the quality tool attribute is set to fastqc and that includes attributes to extract reports or keep them in a zip file. To enable this quality check make sure you set the cleanup to 0 in the data configuration pipeline as well as to disable the STAR processing, this will retain the fastq.qz files after the data pipeline step is executed.

    "fastq" : {
        "enable" : 1,
        "tool" : "/opt/FastQC/fastqc",
        "extract" : 1   
    },

•   For bam files, the quality tool attribute is set to picard and that includes attributes such as collecting alignment summary metrics. To enable this quality check make sure you set the cleanup to 0 in the data configuration pipeline and add 'TranscriptomeSAM' to the arguments for STAR which will then output BAM files that will be retained after the data pipeline step is executed.

    "bam" : {
        "enable" : 1,
        "tool" : "java",
        "jar" : "/opt/picard-tools-1.88/CollectAlignmentSummaryMetrics.jar"
    },
    

•   Example processing:

    python3 run.py qc

    # ---------------------------------------------------
    # Quality Check
    fastqc data/tmp/out.1.fastq.gz --outdir=data/out --extract
    fastqc data/tmp/out.2.fastq.gz --outdir=data/out --extract
    java -jar /opt/picard-tools-1.88/CollectAlignmentSummaryMetrics.jar INPUT=data/tmp/SRR3438604_Aligned.bam OUTPUT=data/out/SRR3438604_Aligned.bam.txt
    java -jar /opt/picard-tools-1.88/CollectAlignmentSummaryMetrics.jar INPUT=data/tmp/SRR3438605_Aligned.bam OUTPUT=data/out/SRR3438605_Aligned.bam.txt
    # Finished
    # ---------------------------------------------------


### target: report
•   To generate the report from the notebook, run this command:

    python3 run.py report

•   The configuration files for the data step are stored in config/report-params.json. 

    {
        "tool": "jupyter",
        "args": "nbconvert --no-input --to html --output report.html notebooks/report.ipynb",
        "verbose" : 1
    }


### target: clean 

•	To clean the data (remove it from the working project), from the root project directory run the command:

python3 run.py clean


### target: all 

•   The all target will execute the following steps in sequence: data, merge, normalize, analysis and visualize. It can be executed as follows:

python3 run.py all


### Future Work

•	New pipeline step: predict. This step will use the model to predict the classification for a given miRNA sequences on the test data and reporting the classification errors



### Major Change History


Date:  2/2/2021

Work completed:

- Created new visualization, Volcano Plot, wrote the code and implemented it into our pipeline in the visualize step 
- Updated the code pipeline to make the correlation matrix more meaningful by adding color
- Finished descriptions for EDA plots


Date:  1/19/2021

Work completed:

- Got all steps in pipeline to work with new data (data, merge, normalize, analysis, visualize) 
- Used LRT Hypothesis Testing and have updated all previous quarter visualizations to work for our new data set
- Compared the outputs of 2 samples that failed FastQC/ERCC quality check with 2 samples that passed
- Developed and organized EDA code for gene matrix (missingness, correlation between sequence count and numerical features of the samples) 


Date:  1/12/2020

Work completed:

- Created repo, initial version from the DSC180A Genetics project
- Added new download step and modified data target to use new SRA
- Wrote out background information/introduction sections of the report, researched our diseases (Alzheimer’s/Parkinson’s) and data sources (miRNA, serum/CSF)
- Developed and organized EDA code for features in the SRA run table (box plots, histograms, bar plots, etc)



### Responsibilities


* Saroop Samra, developed the original codebase based on the DSC180A genetics replication project. Saroop ported the code to support the new miRNA dataset including adding a new download step in the data target. She worked on modifying the code and configuration files for the merge, normalize, analysis and visualize targets to process and generate the visualizations from the DSC180A project. She got the new Volcano Plot to work for our dataset and wrote basic descriptions for the visualizations including what significant patterns exists (MA plot, heatmap, histogram, venn diagram, correlation matrix).

* Justin Lu, wrote out background information/introduction sections of the report. Justin did the data quality control check with FastQC (focused on the FastQC report outputs that we acquired instead of actually running FastQC since we still do not have access to the raw .fastq data), wrote in descriptions for visualizations and some of the EDA in our final report notebook. He updated the code pipeline with colored correlation matrix.

* Xuanyu Wu, generated around 20 EDA plots for features that describe our merged dataset (incl. box plots, histograms, bar plots, etc) Xuanyu created EDA plots to explore the missingness of the gene count matrix and the basic correlation of each sequence with the numerical features we have selected. She also finished the descriptions for EDA plots and analysis.




