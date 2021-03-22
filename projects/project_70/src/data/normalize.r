#!/usr/bin/env Rscript

# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020


# MAIN CITED PAGE:
# Cite: https://bioinformatics.uconn.edu/resources-and-events/tutorials-2/rna-seq-tutorial-with-reference-genome/#

# Add this to use the R from RStudio which has faster DESeq2
.libPaths( "/opt/conda/envs/r-bio/lib/R/library" )


# check if package not installed
#if (!requireNamespace("BiocManager", quietly = TRUE)) {
#    install.packages("BiocManager") #, lib="/tmp", repos = "http://cran.us.r-project.org")
#	BiocManager::install("DESeq2")
#}

# load library but don't show messages
suppressPackageStartupMessages( library( "DESeq2" ) )
suppressPackageStartupMessages( library("gplots") )

# get the command line arguments
args = commandArgs(trailingOnly=TRUE)

# verify args
if (length(args) < 3) {
	stop("Missing arguments : <gene matrix file> <experiment file> <output_prefix>>")
}

gene_matrix_file = args[1]
experiment_file = args[2]
output_prefix = args[3]

# convert string to variables we can pass to R
# Cite: https://win-vector.com/2018/09/01/r-tip-how-to-pass-a-formula-to-lm/
# full <- as.forumla(paste(full, collapse = " + "))


# I used the following example to base my code on
# Cite: https://dputhier.github.io/ASG/practicals/rnaseq_diff_Snf2/rnaseq_diff_Snf2.html

# Load to tables
experiment_features <- read.table(file=experiment_file, sep="\t", header=TRUE)
gene_count.table <- read.table(file=gene_matrix_file, sep="\t", header=TRUE, row.names=1)


# Deal with NaN's: This is now done in Python before this script is called
#gene_count.table[is.na(gene_count.table)] <- 0


## Use the DESeqDataSetFromMatrix to create a DESeqDataSet object
#deseq_data <- DESeqDataSetFromMatrix(countData = round(gene_count.table + 1), colData = experiment_features, design = ~ expired_age+sex+PMI+sn_depigmentation+Braak_score+TangleTotal+Plaque_density+PlaqueTotal+Disorder)
# Add 1 to avoid error: normalize: every gene contains at least one zero, cannot compute log geometric 
deseq_data <- DESeqDataSetFromMatrix(countData = gene_count.table, colData = experiment_features, design = ~ expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal+Disorder)




dds_norm <-  estimateSizeFactors(deseq_data)

normalized_counts <- counts(dds_norm, normalized=TRUE)
write.table(normalized_counts, file=paste(output_prefix, "normalized_counts.tsv", sep = ""), sep="\t", quote=F, col.names=NA)

#normalized_values <- sizeFactors(dds_norm)
#write.table(normalized_values, file=paste(output_prefix, "normalized_counts.tsv", sep = ""), sep="\t", quote=F, col.names=NA)

# Variance Stabilizing Transformation (VST)
vsd <- varianceStabilizingTransformation(deseq_data, blind=T, fitType='parametric')
# Output VST
write.table(as.data.frame(assay(vsd)), file = paste(output_prefix, "vst_transformed_counts.tsv", sep=""), sep = '\t', col.names=NA)

print(paste("Output", paste(output_prefix, "normalized_counts.tsv", sep = ""), paste(output_prefix, "vst_transformed_counts.tsv", sep="")))


# PCA: Dont do this as we just output PCA plots instead
#pca1 <- prcomp(t(as.data.frame(assay(vsd))))
#summary(pca1)
#write.table(summary(pca1)$importance , file = paste(output_prefix, "PCA_summary.tsv", sep=""), sep = '\t', col.names=NA)

# PCA Plots
png(paste(output_prefix, "PCAplot_Disorder.png", sep = ""))
plotPCA(vsd, intgroup = "Disorder", ntop = 500)
dev.off()

png(paste(output_prefix, "PCAplot_Biofluid.png", sep = ""))
plotPCA(vsd, intgroup = "Biofluid", ntop = 500)
dev.off()


