#command arguments from config/analysis-params.json which specifies what tool is being used & in/out file directories
myArgs <- commandArgs(trailingOnly = TRUE)
library('compcodeR')
#library('baySeq')
library('ABSSeq')
library("PoissonSeq")

indir <- myArgs[1]
data <- myArgs[2]
tool <- myArgs[3]
rmdFunc <- myArgs[4]
outdir <- myArgs[5]
#change to include 2 and 10 when added
sample_no <- c('5')
versions <- c('v1', 'v2', 'v3', 'v4', 'v5',
              'v6', 'v7', 'v8', 'v9', 'v10')

if (tool == 'DESeq2') {
    #Run DESeq2 on synthetic data
    runDiffExp(data.file = file.path(indir, data), result.extent = tool, Rmdfunction = rmdFunc, output.directory = file.path(outdir), fit.type = "parametric", test = "Wald", beta.prior = TRUE, independent.filtering = TRUE, cooks.cutoff = TRUE, impute.outliers = TRUE)
    }

if (tool == 'edgeR.exact') {
    #Run edgeR on synthetic data
    runDiffExp(data.file = file.path(indir, data), result.extent = tool, Rmdfunction = rmdFunc, output.directory = file.path(outdir), norm.method = "TMM", disp.type = "tagwise", trend.method = "movingave")
    }

if (tool == 'voom.limma' | tool == 'ttest' | tool == 'NOISeq') {
    #Run voom limma, ttest, or NOISeq on synthetic data
    runDiffExp(data.file = file.path(indir, data), result.extent = tool, Rmdfunction = rmdFunc, output.directory = file.path(outdir), norm.method = "TMM")
    }

#run PoissonSeq on data
if (tool == 'PoissonSeq') {
    #reformat compcode data to work for PoissonSeq
    path <- paste("~/RNASeqToolComparison",indir,data, sep="/")
    temp_data <- readRDS(path)
    n <- temp_data@count.matrix
    s <- substr(outdir, nchar(outdir),nchar(outdir))
    if (s == '0') {
        s <- '10'
    }

    #create sample labels
    y <- c(replicate(strtoi(s), 1), replicate(strtoi(s), 2))
    type <-'twoclass'
    pair <- FALSE

    #trasform data format and run PoissonSeq package
    transformed_data <- list(n=n, y=y, type=type, pair=pair)
    pois_res <- PS.Main(transformed_data)
    ordered_pois_res <- pois_res[order(pois_res['gname'], decreasing=FALSE),]

    #label with real classifications (0 or 1)
    labels <- matrix(c(temp_data@variable.annotations$differential.expression, c(1:NROW(n))), ncol=2)
    colnames(labels) <- c("actual", "gene_number")
    results <- merge(x=labels, y=pois_res, by.x="gene_number", by.y="gname", all.x=TRUE)
    result_df <- as.data.frame(results)

    #generate predictions
    result_df$prediction <- ifelse(result_df$pval < 0.05, 1, 0)
    result_df$prediction[is.na(result_df$prediction)] <- 0
    result_df$dif <- abs(result_df$actual - result_df$prediction)

    #save data in outdir
    v <- substr(data, nchar(data)-5, nchar(data)-4)
    if (v=='10') {
        v <- 'v10'
    }
}

#run ABSSeq on data
if (tool == 'ABSSeq') {
    #reformat compcode data to work for PoissonSeq
    path <- paste("~/RNASeqToolComparison",indir,data, sep="/")
    temp_data <- readRDS(path)
    s <- substr(outdir, nchar(outdir),nchar(outdir))
    if (s == '0') {
        s <- '10'
    }
    #create sample labels
    groups <- c(replicate(strtoi(s), 1), replicate(strtoi(s), 2))

    #trasform data format and run PoissonSeq package
    absdata <- ABSDataSet(temp_data@count.matrix, groups)
    obj <- ABSSeq(absdata, useaFold=TRUE)
    abs_res <- results(obj,c("Amean","Bmean","foldChange","pvalue","adj.pvalue"))

    #label with real classifications (0 or 1)
    labels <- matrix(c(temp_data@variable.annotations$differential.expression, c(1:NROW(temp_data@count.matrix))), ncol=2)
    colnames(labels) <- c("actual", "gene_number")
    abs_res <- cbind(gene = rownames(abs_res), abs_res)
    rownames(abs_res) <- 1:nrow(abs_res)
    abs_res <- cbind(gene_number = rownames(abs_res), abs_res)
    rownames(abs_res) <- 1:nrow(abs_res)
    results <- merge(x=labels, y=abs_res, by="gene_number",all.x=TRUE)

    #generate predictions
    result_df <- as.data.frame(results)
    result_df$prediction <- ifelse(result_df$adj.pval < 0.05, 1, 0)
    result_df$prediction[is.na(result_df$prediction)] <- 0
    result_df$dif <- abs(result_df$actual - result_df$prediction)
    result_df <- result_df[c("gene","Amean","Bmean","foldChange","pvalue","adj.pvalue","actual","prediction","dif")]

    #save data in outdir
    v <- substr(data, nchar(data)-5, nchar(data)-4)
    if (v=='10') {
        v <- 'v10'
    }
}
