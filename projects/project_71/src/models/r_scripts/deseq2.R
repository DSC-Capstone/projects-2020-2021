library("DESeq2")

# Set the working directory
directory <- "G://courses//WI21//DSC\ 180B//temp"
setwd(directory)

# Set the prefix for each output file name
outputPrefix <- "Opod_DESeq2"

sampleFiles<- c("abundance_7949794.tsv", "abundance_7949817.tsv")
sampleNames <- c("Control", "Experie")
sampleCondition <- c("Control","Experimental")
sampleTable <- data.frame(sampleName = sampleNames, fileName = sampleFiles, condition = sampleCondition)

# ctsdir = paste("./data/features/subcts/", paste("subcts_", dataname, sep = ""), sep = "")
cts <- as.matrix(read.csv("htseq_cts_gene_filtered.csv", sep = ",", row.names = "ensemble_id"))

# coldatadir = paste("./data/features/subcoldata/", paste("subcoldata_", dataname, sep = ""), sep = "")
#coldata <- read.csv("SRA_case_table.csv", sep = ",", row.names = "SRR.Number")
coldata <- read.csv("SRA_case_table_temp.csv", sep = ",", row.names = "SRR.Number")
coldata <- coldata[, -(1)]
coldata$Age <- factor(coldata$Age)
#coldata$Disorder <- factor(coldata$Disorder)

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
                              design = ~ Age + Brain.pH + RIN)

# set up metadata
featureData <- data.frame(gene = rownames(cts))
mcols(dds) <- DataFrame(mcols(dds), featureData)
mcols(dds)

# ----------------- FILTERING ------------------ #

# pre-filter
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]

# factors in R

#dds$Disorder <- factor(dds$Disorder, levels = c(dn[j], "Control"))
#print(dn[j])

# ----------------- DEA ---------------------- #

# carries out: LRT, estimation of dispersion:
dds <- DESeq(dds, test = "LRT", reduced = ~Age + Brain.pH)
dds$Group <- factor(dds$Group, levels = c("Control", "Experimental"))
# dds <- makeExampleDESeqDataSet(n=20000, m=20)
vsd <- vst(dds, blind = FALSE)

res <- results(dds)

plotPCA(vsd, intgroup = c("Group"))

# outdir = paste("./data/features/LRT/vst", dataname, sep = "/")
outdir = "./dds_res_temp_filtered.csv"
print(outdir)
write.csv(res, outdir)
