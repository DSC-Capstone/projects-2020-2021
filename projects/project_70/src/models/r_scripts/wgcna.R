options(ggrepel.max.overlaps = Inf)

directory <- "G://courses//WI21//DSC\ 180B//temp"
setwd(directory)

res <- read.csv("dds_res_temp_filtered.csv", row.names = "X", sep = ",")
res <- res[res$pvalue < 1, ]

gene.names = row.names(res)
mydata.trans=t(res)
coldata <- read.csv("SRA_case_table_temp.csv", sep = ",", row.names = "SRR.Number")

softPower = 12

n=dim(res)[1]
datExpr=mydata.trans[,gene.names[1:n]]
#datExpr = res[1:n, ]

TOM=TOMsimilarityFromExpr(datExpr,
                          networkType = "signed",
                          TOMType = "signed",
                          power = softPower)

colnames(TOM) =rownames(TOM) =gene.names
dissTOM=1-TOM

net = blockwiseModules(datExpr, power = 16,
                       TOMType = "signed", minModuleSize = 30, 
                       corType = "bicor",
                       reassignThreshold = 0, mergeCutHeight = 0.25,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = FALSE,
                       verbose = 3,
                       minClusterSize = 5,
                       maxBlockSize = 40000)

geneTree = net$dendrograms[[1]]
moduleLabels = net$colors 
moduleColors = labels2colors(net$colors)

plotDendroAndColors(geneTree, 
                    labels2colors(net$unmergedColors),
                    "Module colors", 
                    dendroLabels = F,
                    hang = 0.03,
                    addGuide = TRUE, 
                    guideHang = 0.05)
