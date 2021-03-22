
myArgs <- commandArgs(trailingOnly = TRUE)

library('compcodeR')
library('ABSSeq')
library("PoissonSeq")

data_name <- myArgs[1]
num_vars <- strtoi(myArgs[2])
samples_per_cond <- strtoi(myArgs[3])
rep_id <- strtoi(myArgs[4])
seq_depth <- as.numeric(myArgs[5])
num_diff_exp <- strtoi(myArgs[6])
ratio_upregulated <- as.numeric(myArgs[7])
dispersion_num <- strtoi(myArgs[8])
output_dir <- myArgs[9]
indir <- myArgs[10]
data <- myArgs[11]
tool1 <- myArgs[12]
rmdFunc1 <- myArgs[13]
tool2 <- myArgs[14]
rmdFunc2 <- myArgs[15]
tool3 <- myArgs[16]
rmdFunc3 <- myArgs[17]
tool4 <- myArgs[18]
rmdFunc4 <- myArgs[19]
tool5 <- myArgs[20]
rmdFunc5 <- myArgs[21]
outdir <- myArgs[22]

#Create a synthetic data with 100 genes differentially expressed with 50 genes expressed in condition 1 & 50 genes expressed in condition 2
test_synthetic_data <- generateSyntheticData(dataset = data_name, n.vars = num_vars, samples.per.cond = samples_per_cond, repl.id = rep_id, seqdepth = seq_depth, n.diffexp = num_diff_exp, fraction.upregulated = ratio_upregulated, fraction.non.overdispersed = dispersion_num, between.group.diffdisp = FALSE, filter.threshold.total = rep_id, filter.threshold.mediancpm = dispersion_num, output.file = output_dir)

#Run DESeq2 on the synthetic dataset created above
runDiffExp(data.file = file.path(indir, data), result.extent = tool1, Rmdfunction = rmdFunc1, output.directory = file.path(outdir), fit.type = "parametric", test = "Wald", beta.prior = TRUE, independent.filtering = TRUE, cooks.cutoff = TRUE, impute.outliers = TRUE)

#Run edgeR on the synthetic dataset created above
runDiffExp(data.file = file.path(indir, data), result.extent = tool2, Rmdfunction = rmdFunc2, output.directory = file.path(outdir), norm.method = "TMM", disp.type = "tagwise", trend.method = "movingave")

#Run NOISeq on the synthetic dataset created above
runDiffExp(data.file = file.path(indir, data), result.extent = tool3, Rmdfunction = rmdFunc3, output.directory = file.path(outdir), norm.method = "TMM")

#Run voom.limma on the synthetic dataset created above
runDiffExp(data.file = file.path(indir, data), result.extent = tool4, Rmdfunction = rmdFunc4, output.directory = file.path(outdir), norm.method = "TMM")

#Run ttest on the synthetic dataset created above
runDiffExp(data.file = file.path(indir, data), result.extent = tool5, Rmdfunction = rmdFunc5, output.directory = file.path(outdir), norm.method = "TMM")

#Run PoissonSeq on the synthetic dataset created above
path <- paste("~/RNASeqToolComparison",indir,data, sep="/")
temp_data <- readRDS(path)
n <- temp_data@count.matrix
y <- c(1,1,1,1,1,2,2,2,2,2)
type <-'twoclass'
pair <- FALSE
transformed_data <- list(n=n, y=y, type=type, pair=pair)
pois_res <- PS.Main(transformed_data)
ordered_pois_res <- pois_res[order(pois_res['gname'], decreasing=FALSE),]
labels <- matrix(c(temp_data@variable.annotations$differential.expression, c(1:NROW(n))), ncol=2)
colnames(labels) <- c("actual", "gene_number")
results <- merge(x=labels, y=pois_res, by.x="gene_number", by.y="gname", all.x=TRUE)
result_df <- as.data.frame(results)
result_df$prediction <- ifelse(result_df$pval < 0.05, 1, 0)
result_df$prediction[is.na(result_df$prediction)] <- 0
result_df$dif <- abs(result_df$actual - result_df$prediction)

filename <- paste(substr(data, 0, nchar(data)-4),"_", tool, ".rds",sep="")
out <- paste("~/RNASeqToolComparison/out/test", filename,  sep="/")
saveRDS(result_df, out)

#Run ABSSeq on the synthetic dataset created above
path <- paste("~/RNASeqToolComparison",indir,data, sep="/")
temp_data <- readRDS(path)
groups <- c(1,1,1,1,1,2,2,2,2,2)
absdata <- ABSDataSet(temp_data@count.matrix, groups)

obj <- ABSSeq(absdata, useaFold=TRUE)
abs_res <- results(obj,c("Amean","Bmean","foldChange","pvalue","adj.pvalue"))

labels <- matrix(c(temp_data@variable.annotations$differential.expression, c(1:NROW(temp_data@count.matrix))), ncol=2)
colnames(labels) <- c("actual", "gene_number")
abs_res <- cbind(gene = rownames(abs_res), abs_res)
rownames(abs_res) <- 1:nrow(abs_res)
abs_res <- cbind(gene_number = rownames(abs_res), abs_res)
rownames(abs_res) <- 1:nrow(abs_res)
results <- merge(x=labels, y=abs_res, by="gene_number",all.x=TRUE)
result_df <- as.data.frame(results)
result_df$prediction <- ifelse(result_df$adj.pval < 0.05, 1, 0)
result_df$prediction[is.na(result_df$prediction)] <- 0
result_df$dif <- abs(result_df$actual - result_df$prediction)
result_df <- result_df[c("gene","Amean","Bmean","foldChange","pvalue","adj.pvalue","actual","prediction","dif")]

filename <- paste(substr(data, 0, nchar(data)-4),"_", tool, ".rds",sep="")
out <- paste("~/RNASeqToolComparison/out/test", filename,  sep="/")
saveRDS(result_df, out)

