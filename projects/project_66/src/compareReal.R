library(compcodeR)
library(ggplot2)

myArgs <- commandArgs(trailingOnly = TRUE)

dataset <- myArgs[1]
tool1 <- myArgs[2]
tool2 <- myArgs[3]
tool3 <- myArgs[4]
tool4 <- myArgs[5]
tool5 <- myArgs[6]
tool6 <- myArgs[7]
out_dir <- myArgs[8]

if (dataset == 'out/real/schizo') {
    deseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool1, sep="/"))
    deseq_results <- convertcompDataToList(deseq)$result
    deseq_results$prediction <- ifelse(deseq_results$pval < 0.05, 1, 0)
    deseq_results$prediction[is.na(deseq_results$prediction)] <- 0
    deseq_results$ID <- seq.int(nrow(deseq_results))
    deseq_count <- sum(deseq_results$prediction)

    edgeR <- readRDS(paste("~/RNASeqToolComparison", dataset, tool2, sep="/"))
    edgeR_results <- convertcompDataToList(edgeR)$result
    edgeR_results$prediction <- ifelse(edgeR_results$pval < 0.05, 1, 0)
    edgeR_results$prediction[is.na(edgeR_results$prediction)] <- 0
    edgeR_results$ID <- seq.int(nrow(edgeR_results))
    edgeR_count <- sum(edgeR_results$prediction)

    ttest <- readRDS(paste("~/RNASeqToolComparison", dataset, tool3, sep="/"))
    ttest_results <- convertcompDataToList(ttest)$result
    ttest_results$prediction <- ifelse(ttest_results$pval < 0.05, 1, 0)
    ttest_results$prediction[is.na(ttest_results$prediction)] <- 0
    ttest_results$ID <- seq.int(nrow(ttest_results))
    ttest_count <- sum(ttest_results$prediction)

    voom <- readRDS(paste("~/RNASeqToolComparison", dataset, tool4, sep="/"))
    voom_results <- convertcompDataToList(voom)$result
    voom_results$prediction <- ifelse(voom_results$pval < 0.05, 1, 0)
    voom_results$prediction[is.na(voom_results$prediction)] <- 0
    voom_results$ID <- seq.int(nrow(voom_results))
    voom_count <- sum(voom_results$prediction)

    poisson <- readRDS(paste("~/RNASeqToolComparison", dataset, tool5, sep="/"))
    poisson$prediction <- ifelse(poisson$pval < 0.05, 1, 0)
    poisson$prediction[is.na(poisson$prediction)] <- 0
    poisson$ID <- seq.int(nrow(poisson))
    poisson_count <- sum(poisson$prediction)

    abseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool6, sep="/"))
    abseq$prediction <- ifelse(abseq$pval < 0.05, 1, 0)
    abseq$prediction[is.na(abseq$prediction)] <- 0
    abseq$ID <- seq.int(nrow(abseq))
    abseq_count <- sum(abseq$prediction)


    df <- data.frame(Tools=c("DESeq2", "edgeR.exact", "ttest", "voom.limma", "PoissonSeq", "ABSSeq"), 
                     "Significant genes"=c(deseq_count, edgeR_count, ttest_count, voom_count, poisson_count, abseq_count))


    p <- ggplot(data=df, aes(x=Tools, y=Significant.genes, fill=Tools)) +
      geom_bar(stat="identity")+
      geom_text(aes(label=Significant.genes), vjust=-0.3, size=3.5)+
      theme_minimal()

    ggsave(filename="Schizophrenia_counts.png", path=out_dir, device="png")
    }

if (dataset == 'out/real/mdd') {
    deseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool1, sep="/"))
    deseq_results <- convertcompDataToList(deseq)$result
    deseq_results$prediction <- ifelse(deseq_results$pval < 0.05, 1, 0)
    deseq_results$prediction[is.na(deseq_results$prediction)] <- 0
    deseq_results$ID <- seq.int(nrow(deseq_results))
    deseq_count <- sum(deseq_results$prediction)

    edgeR <- readRDS(paste("~/RNASeqToolComparison", dataset, tool2, sep="/"))
    edgeR_results <- convertcompDataToList(edgeR)$result
    edgeR_results$prediction <- ifelse(edgeR_results$pval < 0.05, 1, 0)
    edgeR_results$prediction[is.na(edgeR_results$prediction)] <- 0
    edgeR_results$ID <- seq.int(nrow(edgeR_results))
    edgeR_count <- sum(edgeR_results$prediction)

    ttest <- readRDS(paste("~/RNASeqToolComparison", dataset, tool3, sep="/"))
    ttest_results <- convertcompDataToList(ttest)$result
    ttest_results$prediction <- ifelse(ttest_results$pval < 0.05, 1, 0)
    ttest_results$prediction[is.na(ttest_results$prediction)] <- 0
    ttest_results$ID <- seq.int(nrow(ttest_results))
    ttest_count <- sum(ttest_results$prediction)

    voom <- readRDS(paste("~/RNASeqToolComparison", dataset, tool4, sep="/"))
    voom_results <- convertcompDataToList(voom)$result
    voom_results$prediction <- ifelse(voom_results$pval < 0.05, 1, 0)
    voom_results$prediction[is.na(voom_results$prediction)] <- 0
    voom_results$ID <- seq.int(nrow(voom_results))
    voom_count <- sum(voom_results$prediction)

    poisson <- readRDS(paste("~/RNASeqToolComparison", dataset, tool5, sep="/"))
    poisson$prediction <- ifelse(poisson$pval < 0.05, 1, 0)
    poisson$prediction[is.na(poisson$prediction)] <- 0
    poisson$ID <- seq.int(nrow(poisson))
    poisson_count <- sum(poisson$prediction)

    abseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool6, sep="/"))
    abseq$prediction <- ifelse(abseq$pval < 0.05, 1, 0)
    abseq$prediction[is.na(abseq$prediction)] <- 0
    abseq$ID <- seq.int(nrow(abseq))
    abseq_count <- sum(abseq$prediction)


    df <- data.frame(Tools=c("DESeq2", "edgeR.exact", "ttest", "voom.limma", "PoissonSeq", "ABSSeq"), 
                     "Significant genes"=c(deseq_count, edgeR_count, ttest_count, voom_count, poisson_count, abseq_count))


    p <- ggplot(data=df, aes(x=Tools, y=Significant.genes, fill=Tools)) +
      geom_bar(stat="identity")+
      geom_text(aes(label=Significant.genes), vjust=-0.3, size=3.5)+
      theme_minimal()

    ggsave(filename="MDD_counts.png", path=out_dir, device="png")
    }

if (dataset == 'out/real/bipolar') {
    deseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool1, sep="/"))
    deseq_results <- convertcompDataToList(deseq)$result
    deseq_results$prediction <- ifelse(deseq_results$pval < 0.05, 1, 0)
    deseq_results$prediction[is.na(deseq_results$prediction)] <- 0
    deseq_results$ID <- seq.int(nrow(deseq_results))
    deseq_count <- sum(deseq_results$prediction)
    
    edgeR <- readRDS(paste("~/RNASeqToolComparison", dataset, tool2, sep="/"))
    edgeR_results <- convertcompDataToList(edgeR)$result
    edgeR_results$prediction <- ifelse(edgeR_results$pval < 0.05, 1, 0)
    edgeR_results$prediction[is.na(edgeR_results$prediction)] <- 0
    edgeR_results$ID <- seq.int(nrow(edgeR_results))
    edgeR_count <- sum(edgeR_results$prediction)
    
    ttest <- readRDS(paste("~/RNASeqToolComparison", dataset, tool3, sep="/"))
    ttest_results <- convertcompDataToList(ttest)$result
    ttest_results$prediction <- ifelse(ttest_results$pval < 0.05, 1, 0)
    ttest_results$prediction[is.na(ttest_results$prediction)] <- 0
    ttest_results$ID <- seq.int(nrow(ttest_results))
    ttest_count <- sum(ttest_results$prediction)
    
    voom <- readRDS(paste("~/RNASeqToolComparison", dataset, tool4, sep="/"))
    voom_results <- convertcompDataToList(voom)$result
    voom_results$prediction <- ifelse(voom_results$pval < 0.05, 1, 0)
    voom_results$prediction[is.na(voom_results$prediction)] <- 0
    voom_results$ID <- seq.int(nrow(voom_results))
    voom_count <- sum(voom_results$prediction)
    
    poisson <- readRDS(paste("~/RNASeqToolComparison", dataset, tool5, sep="/"))
    poisson$prediction <- ifelse(poisson$pval < 0.05, 1, 0)
    poisson$prediction[is.na(poisson$prediction)] <- 0
    poisson$ID <- seq.int(nrow(poisson))
    poisson_count <- sum(poisson$prediction)
    
    abseq <- readRDS(paste("~/RNASeqToolComparison", dataset, tool6, sep="/"))
    abseq$prediction <- ifelse(abseq$pval < 0.05, 1, 0)
    abseq$prediction[is.na(abseq$prediction)] <- 0
    abseq$ID <- seq.int(nrow(abseq))
    abseq_count <- sum(abseq$prediction)

    df <- data.frame(Tools=c("DESeq2", "edgeR.exact", "ttest", "voom.limma", "PoissonSeq", "ABSSeq"), 
                     "Significant genes"=c(deseq_count, edgeR_count, ttest_count, voom_count, poisson_count, abseq_count))

    p <- ggplot(data=df, aes(x=Tools, y=Significant.genes, fill=Tools)) +
      geom_bar(stat="identity")+
      geom_text(aes(label=Significant.genes), vjust=-0.3, size=3.5)+
      theme_minimal()

    ggsave(filename="Bipolar_counts.png", path=out_dir, device="png")
    }



