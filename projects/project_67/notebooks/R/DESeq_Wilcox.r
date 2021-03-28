# ---------------- IMPORTS ---------------- #
BiocManager::install('EnhancedVolcano')

library("pasilla")
library("DESeq2")
library("apeglm")
library("corrplot")
library("dplyr")
library("RColorBrewer")
library("gplots")
library("pheatmap")
library("EnhancedVolcano")

library("broom")
library("tibble")

# ---------------- DATA ---------------- #
setwd("~/Documents/miRNA")

cts <- as.matrix(read.csv(file = "miRNA_counts_1.csv", row.names="target_id"))
coldata <- read.csv("Filtered_Runs.csv", row.names="Run")
coldata <- coldata[,c("GROUP", "Age", "gender")]
coldata$GROUP <- factor(coldata$GROUP)
cts <- cts[, rownames(coldata)]
all(rownames(coldata) == colnames(cts))

# ---------------- DESeqDataSet ---------------- #

# create DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = floor(cts),
                              colData = coldata,
                              design = ~ GROUP + Age + gender)

# set up metadata
featureData <- data.frame(gene=rownames(cts))
mcols(dds) <- DataFrame(mcols(dds), featureData)

# ---------------- FILTERING ---------------- #

# pre-filter
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]

# factors in R!
dds$GROUP <- factor(dds$GROUP, levels = c("control", "alzheimer patient"))

dds <- DESeq(dds)
res <- results(dds)

# Log fold change
#Need to change coef 
resLFC <- lfcShrink(dds, coef="GROUP_alzheimer.patient_vs_control", type="apeglm")

EnhancedVolcano(resLFC,
                lab = rownames(resLFC),
                x = 'log2FoldChange',
                y = 'pvalue')

resOrdered <- resLFC[order(resLFC$pvalue),]
sum(resOrdered$padj < 0.1, na.rm=TRUE)

res05 <- results(dds, alpha=0.05)
summary(res05)

#Wilcox Test

counts <- as.matrix(read.csv(file = "wilcox_padded.csv"))
ads <- as.matrix(read.csv(file = "ads.csv"))
ctls <- as.matrix(read.csv(file = "ctls.csv"))

test <- wilcox.test(as.numeric(x = ads[1, -c(1:2)], y = ctls[1, -c(1:2)]))

log2s <- c()

log2 <- log2(mean(as.numeric(ads[1, -c(1:2)]))/mean(as.numeric(ctls[1, -c(1:2)])))

log2s[1] <- log2

test_df <- tibble(tidy(test))

for (row in 2:nrow(counts)){
  wilcox <- wilcox.test(as.numeric(x = ads[row, -c(1:2)], y = ctls[row, -c(1:2)]))
  test_df <- add_row(test_df, tidy(wilcox))
  log2 <- log2(mean(as.numeric(ads[row, -c(1:2)]))/mean(as.numeric(ctls[row, -c(1:2)])))
  log2s[row] <- log2 
}

test_df <- add_column(test_df, log2fold = log2s)

wilcox_test <- read.csv(file = "wilcox_test.csv")

EnhancedVolcano(wilcox_test,
                lab = "target_id",
                x = 'log2fold',
                y = 'p.value')

write.table(test_df, file = "wilcox_r.csv")

