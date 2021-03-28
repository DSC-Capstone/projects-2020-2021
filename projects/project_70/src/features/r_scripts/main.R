# code pulled from vignette: http://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#quick-start

# install packages
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("DESeq2")
# BiocManager::install("pasilla")
# BiocManager::install("apeglm")

# ---------------- IMPORTS ---------------- #

# import packages
library("DESeq2")

# ---------------- DATA ---------------- #


brs <- c("AnCg", "nAcc", "DLPFC")
ds <- c("MD", "BP", "SZ")
dn <- c("Major Depression", "Bipolar Disorder", "Schizophrenia")

for (i in c(1, 2, 3)) {
  for (j in c(1, 2, 3)) {
    # import our data
    dataname <- paste(brs[i], paste(ds[j], "csv", sep = "."), sep = "_")
    print(dataname)
    
    ctsdir = paste("./data/features/subcts/", paste("subcts_", dataname, sep = ""), sep = "")
    cts <- as.matrix(read.csv(ctsdir,row.names="target_id"))
    print(ctsdir)
    
    coldatadir = paste("./data/features/subcoldata/", paste("subcoldata_", dataname, sep = ""), sep = "")
    coldata <- read.csv(coldatadir , row.names = "Run")
    coldata$brain_region <- factor(coldata$brain_region)
    coldata$Disorder <- factor(coldata$Disorder)
    print(coldatadir)
    
    # look at the data
    head(cts,2)
    head(coldata, 2)
    
    # the same samples
    all(rownames(coldata) %in% colnames(cts))
    # but not the same order!
    all(rownames(coldata) == colnames(cts))
    
    # sort to be in the same order
    cts <- cts[, rownames(coldata)]
    all(rownames(coldata) == colnames(cts))
    
    # ---------------- DESeqDataSet ---------------- #
    
    # create our DESeqDataSet object
    dds <- DESeqDataSetFromMatrix(countData = cts,
                                  colData = coldata,
                                  design = ~ Age + PMI + pH + Disorder)
    
    # set up metadata
    featureData <- data.frame(gene = rownames(cts))
    mcols(dds) <- DataFrame(mcols(dds), featureData)
    mcols(dds)
    
    # ----------------- FILTERING ------------------ #
    
    # pre-filter
    keep <- rowSums(counts(dds)) >= 10
    dds <- dds[keep,]
    
    # factors in R
    dds$Disorder <- factor(dds$Disorder, levels = c("Control", dn[j]))
    print(dn[j])
    
    # ----------------- DEA ---------------------- #
    
    # carries out: LRT, estimation of dispersion:
    dds <- DESeq(dds, test = "LRT", reduced = ~Age + PMI + pH)
    res <- results(dds)
    
    # carries out VST
    vstname = gsub(" ", ".", paste("Disorder", dn[j], "vs", "Control", sep = "_"))
    
    outdir = paste("./data/features/LRT/paper", dataname, sep = "/")
    print(outdir)
    write.csv(res, outdir)
  }
}


