library("biomaRt")
library("flashClust")
options(ggrepel.max.overlaps = Inf)

directory <- "G://courses//WI21//DSC\ 180B//temp"
setwd(directory)

res <- read.csv("dds_res_temp_filtered.csv", row.names = "X", sep = ",")
res <- res[res$pvalue < 1, ]

library(EnhancedVolcano)
library(WGCNA)

EnhancedVolcano(res,
                lab = rownames(res),
                selectLab = c("ENSG00000260097.2", "ENSG00000196549.10", "ENSG00000132196.13"),
                xlab = bquote(~Log[2]~ 'fold change'),
                x = 'log2FoldChange',
                y = 'pvalue', 
                pCutoff = 0.1,
                FCcutoff = 0.5,
                labCol = 'black',
                labFace = 'bold',
                labSize = 4,
                legendPosition = 'right',
                legendLabSize = 12,
                xlim = c(-1, 1),
                drawConnectors = FALSE,
                arrowheads = FALSE)