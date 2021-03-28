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
#    install.packages("BiocManager")
#	BiocManager::install("DESeq2")
#}

library("RColorBrewer")
#if (!requireNamespace("gplots", quietly = TRUE)) {
#    install.packages('gplots',repos = "http://cran.us.r-project.org")
#}
suppressPackageStartupMessages( library("gplots") )

# load library but don't show messages
suppressPackageStartupMessages( library( "DESeq2" ) )

# Parallel Processing to make it faster
library("BiocParallel")

# Print the DESeq version
print("DESeq Package version=")
packageVersion("DESeq2")

# get the command line arguments
args = commandArgs(trailingOnly=TRUE)

# verify args
if (length(args) < 7) {
	stop("Missing arguments : <gene matrix file> <experiment file> <output_prefix> <full> <reduced> charts=[0|1] parallel=[0|1]")
}

gene_matrix_file = args[1]
experiment_file = args[2]
output_prefix = args[3]
full = args[4]
reduced = args[5]
charts = args[6]
parallel = args[7]

# convert string to variables we can pass to R
# Cite: https://win-vector.com/2018/09/01/r-tip-how-to-pass-a-formula-to-lm/
# full <- as.forumla(paste(full, collapse = " + "))



# I used the following example to base my code on
# Cite: https://dputhier.github.io/ASG/practicals/rnaseq_diff_Snf2/rnaseq_diff_Snf2.html

# Load to tables
experiment_features <- read.table(file=experiment_file, sep="\t", header=TRUE)
gene_count.table <- read.table(file=gene_matrix_file, sep="\t", header=TRUE, row.names=1)
print("Loading Complete")


# Deal with NaN's: Now done in Pandas
#gene_count.table[is.na(gene_count.table)] <- 0

# TODO: PRUA, percentage of genes uniquely aligned
# TODO : Remove Non-convergent genes
# TODO: correct for PRUA by computing residuals to linear model regression PRUA on nornalized gene...

## Use the DESeqDataSetFromMatrix to create a DESeqDataSet object
#deseq_data <- DESeqDataSetFromMatrix(countData = gene_count.table, colData = experiment_features, design = ~ Age+PMI+pH+PRUA+Disorder)
# Add 1 to avoid error: normalize: every gene contains at least one zero, cannot compute log geometric 

print(paste("gene count dimensions", dim(gene_count.table)))
print(paste("experiment features dimensions", dim(experiment_features)))

deseq_data <- DESeqDataSetFromMatrix(countData = round(gene_count.table+1), colData = experiment_features, design = ~ expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal+Disorder)
print("Matrix Set Complete")

dds_norm <-  estimateSizeFactors(deseq_data)
print("estimateSizeFactors complete")

# Likelihood Ratio Test (LRT)
if (parallel == "parallel=1") {
	# Parallel version
	dds_lrt <- DESeq(deseq_data, test="LRT", parallel=TRUE, BPPARAM=MulticoreParam(workers=4), reduced =~ expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal)
} else {
	dds_lrt <- DESeq(deseq_data, test="LRT", reduced =~ expired_age+sex+PMI+sn_depigmentation+Braak_Score+TangleTotal+Plaque_density+PlaqueTotal) 
}

print("DESeq LRT setup complete")  


# Extract results
res_LRT <- results(dds_lrt)

print("DESeq LRT results complete")

# Save LRT results
write.table(res_LRT, file=paste(output_prefix, "lrt.tsv", sep = ""), sep="\t", quote=F, col.names=NA)

print("DESeq LRT saving complete")

if (charts == "charts=1") {
	# Output MAPlot
	png(paste(output_prefix, "MAplot.png", sep = ""))
	plotMA(res_LRT, ylim=c(-8,8),main = "")
	dev.off()

	print("DESeq MA Plot complete")

	# Variance Stabilizing Transformation (VST)
	vsd <- varianceStabilizingTransformation(deseq_data, blind=T)

	print("DESeq VST Setup complete")

	# Heatmap of data of VST with 1000 top expressed genes with heatmap.2
	select <- order(rowMeans(counts(dds_lrt,normalized=T)),decreasing=T)[1:20]
	my_palette <- colorRampPalette(c("blue",'white','red'))(n=20)
	png(paste(output_prefix, "heatmap.png", sep = ""))
	heatmap.2(assay(vsd)[select,], col=my_palette,
	          scale="row", key=T, keysize=1, symkey=T,
	          density.info="none", trace="none",
	          cexCol=0.6, labRow=F,
	          main="")
	dev.off()

	print("DESeq Heatmaps complete")

}


