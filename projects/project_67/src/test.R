
myArgs <- commandArgs(trailingOnly = TRUE)

library('compcodeR')
library('ABSSeq')
library("PoissonSeq")
library('ggplot2')
library('cvAUC')

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
test_synthetic_data <- generateSyntheticData(dataset = data_name, n.vars = num_vars, samples.per.cond = samples_per_cond, repl.id = rep_id, seqdepth = seq_depth, n.diffexp = num_diff_exp, fraction.upregulated = ratio_upregulated, fraction.non.overdispersed = dispersion_num, between.group.diffdisp = FALSE, filter.threshold.total = rep_id, filter.threshold.mediancpm = dispersion_num, output.file = file.path(indir, data))

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

filename <- paste(substr(data, 0, nchar(data)-4),"_PoissonSeq.rds",sep="")
out <- paste("~/RNASeqToolComparison/out/test", filename,  sep="/")
saveRDS(result_df, out)

#Run ABSSeq on the synthetic dataset created above
path <- paste("~/RNASeqToolComparison",indir,data, sep="/")
temp_data <- readRDS(path)
groups <- c(1,1,1,1,1,2,2,2,2,2)
absdata <- ABSDataSet(temp_data@count.matrix, groups)

obj <- ABSSeq(absdata, useaFold=TRUE)
abs_res <- ABSSeq::results(obj,c("Amean","Bmean","foldChange","pvalue","adj.pvalue"))

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

filename <- paste(substr(data, 0, nchar(data)-4),"_ABSSeq.rds",sep="")
out <- paste("~/RNASeqToolComparison/out/test", filename,  sep="/")
saveRDS(result_df, out)

#See how many genes each tools considered are differentially expressed
#test_ABSSeq <- readRDS(file.path(outdir, "test_ABSSeq.rds"))
#test_ABSSeq_diff_exp <- sum(test_ABSSeq["prediction"])
test_DESeq2 <- convertcompDataToList(readRDS(file.path(outdir, "test_DESeq2.rds")))
test_DESeq2_diff_exp <- nrow(test_DESeq2$result.table[test_DESeq2$result.table$pvalue < 0.05,])
test_edgeR <- convertcompDataToList(readRDS(file.path(outdir, "test_edgeR.exact.rds")))
test_edgeR_diff_exp<- nrow(test_edgeR$result.table[test_edgeR$result.table$pvalue < 0.05,])
test_NOISeq <- convertcompDataToList(readRDS(file.path(outdir, "test_NOISeq.rds")))
test_NOISeq_diff_exp <- nrow(test_NOISeq$result.table[test_NOISeq$result.table$probabilities < 0.05,])
test_ttest <- convertcompDataToList(readRDS(file.path(outdir, "test_ttest.rds")))
test_ttest_diff_exp <- nrow(test_ttest$result.table[test_ttest$result.table$pvalue < 0.05,])
test_PoissonSeq <- readRDS(file.path(outdir, "test_PoissonSeq.rds"))
test_PoissonSeq_diff_exp <- sum(test_PoissonSeq["prediction"])
test_voom <- convertcompDataToList(readRDS(file.path(outdir, "test_voom.limma.rds")))
test_voom_diff_exp <- nrow(test_voom$result.table[test_voom$result.table$pvalue < 0.05,])
#create a dataframe showing number of genes considered differentially expressed by each tool:
test_df <- data.frame(DESeq2 = c(test_DESeq2_diff_exp),
                      edgeR = c(test_edgeR_diff_exp),
                      NOISeq = c(test_NOISeq_diff_exp),
                      ttest = c(test_ttest_diff_exp),
                      voom = c(test_voom_diff_exp),
                      PoissonSeq = c(test_PoissonSeq_diff_exp))
                             #ABSSeq = c(test_ABSSeq_diff_exp))

tools <- c('DESeq2', 'edgeR.exact', 'voom.limma', 'ttest', 'PoissonSeq', 'ABSSeq')
m <- matrix(ncol=6, nrow=length(tools))
i<-1
for (tool in tools){
    file <- paste("test_",tool,".rds",sep="")
    path <- paste("~/RNASeqToolComparison/out/test",file,sep="/")
    des <- readRDS(path)
    if ((tool != 'ABSSeq') & (tool != 'PoissonSeq')) {
        des_table <- des@result.table
        des_df <- as.data.frame(des_table)
        des_df$prediction <- ifelse(des_df$pval < 0.05, 1, 0)
        des_df$prediction[is.na(des_df$prediction)] <- 0
        des_df$actual <- des@variable.annotations$differential.expression
        des_df$dif <- des_df$prediction - des_df$actual
    } else {
        des_df <- des
    }
    TOTALP <- sum(des_df$prediction)
    FP <- length(which(des_df$dif == 1))
    FDR <- FP/TOTALP
    TP <- length(which(des_df$actual==1 & des_df$prediction==1))
    TN <- length(which(des_df$actual==0 & des_df$prediction==0))
    N <- length(which(des_df$actual == 0))
    P <- length(which(des_df$actual == 1))
    sensitivity <- TP/P
    specificity <- TN/N
    accuracy <- NROW(which(des_df$dif==0)) / NROW(des_df)
    if (length(unique(des_df$actual)) == 1) {
        auc <- 'NULL'
    } else {
        auc <- cvAUC::AUC(des_df$prediction, des_df$actual)
    }
    m[i,] <- c(tool,FDR, sensitivity, specificity, auc, accuracy)
    i<-i+1
}
colnames(m) <- c('Tool', 'FDR', 'Sensitivty', 'Specificity', 'AUC', 'Accuracy')
df <- as.data.frame(m)
ggplot(df, aes(x=Tool, y=AUC)) + geom_boxplot()
ggsave('out/test/test_auc_plot.png')

write.table(m, 'out/test/statistics.csv')

print("The tools have finished running on the test data; The AUC graph & summary of the statistics have been created in out/test folder")
