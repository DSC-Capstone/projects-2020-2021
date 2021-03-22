#command arguments from config/analysis-params.json which specifies what tool is being used & in/out file directories
myArgs <- commandArgs(trailingOnly = TRUE)
library('compcodeR')
library('ABSSeq')
library("PoissonSeq")

indir <- myArgs[1]
data <- myArgs[2]
tool <- myArgs[3]
rmdFunc <- myArgs[4]
outdir <- myArgs[5]

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

if (tool == 'PoissonSeq') {
    #reformat compcode data to work for PoissonSeq
    outdir <- myArgs[6]
    path <- paste("~/RNASeqToolComparison", indir, data, sep="/")
    temp_data <- readRDS(path)
                         
    n <- temp_data@count.matrix
    labels <- temp_data@sample.annotations['condition']
    labels[which(labels[,1] == 'Control', arr.ind=TRUE), 2] <- 1
    labels[which(labels[,1] != 'Control', arr.ind=TRUE), 2] <- 2
    y <- labels$V2
    type <-'twoclass'
    pair <- FALSE
    transformed_data <- list(n=n, y=y, type=type, pair=pair)
    pois_res <- PS.Main(transformed_data)
    ordered_pois_res <- pois_res[order(pois_res['gname'], decreasing=FALSE),]
    
    
    filename <- paste(substr(outdir, 10, nchar(outdir)),"_",tool,".rds",sep="")
    out <- paste("~/RNASeqToolComparison", outdir, filename,  sep="/")
    saveRDS(ordered_pois_res, out)
}

if (tool == 'ABSSeq') {
    #reformat compcode data to work for PoissonSeq
    outdir <- myArgs[6]
    path <- paste("~/RNASeqToolComparison", indir, data, sep="/")
    temp_data <- readRDS(path)
                         
    labels <- temp_data@sample.annotations['condition']
    labels[which(labels[,1] == 'Control', arr.ind=TRUE), 2] <- 1
    labels[which(labels[,1] != 'Control', arr.ind=TRUE), 2] <- 2
    groups <- labels$V2
    
    absdata <- ABSDataSet(temp_data@count.matrix, groups)
    obj <- ABSSeq(absdata, useaFold=TRUE)
    abs_res <- results(obj,c("Amean","Bmean","foldChange","pvalue","adj.pvalue"))
    abs_res <- as.data.frame(abs_res)
                   
    filename <- paste(substr(outdir, 10, nchar(outdir)),"_",tool,".rds",sep="")      
    out <- paste("~/RNASeqToolComparison", outdir, filename,  sep="/")
    saveRDS(abs_res, out)
}
