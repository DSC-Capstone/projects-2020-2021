myArgs <- commandArgs(trailingOnly = TRUE)
library('cvAUC')
library(ggplot2)

tools <- c(myArgs[1], myArgs[2], myArgs[3], myArgs[4], myArgs[5], myArgs[6])
data_list <- c(myArgs[7], myArgs[8], myArgs[9], myArgs[10], myArgs[11], myArgs[12], myArgs[13])

data_0_list <- c(myArgs[14], myArgs[15], myArgs[16], myArgs[17])

sample_no <- c('2', '5', '10')
versions <- c('v1', 'v2', 'v3', 'v4', 'v5',
              'v6', 'v7', 'v8', 'v9', 'v10')
m <- matrix(ncol=9, nrow=length(tools)*length(data_list)*length(sample_no)*length(versions))
m0 <- matrix(ncol=5, nrow=length(tools)*length(data_0_list)*length(sample_no)*length(versions))
i<-1
j<-1
for (tool in tools){
    for (data in data_list) {
        for (sample in sample_no) {
            for (v in versions) {
                dir <- paste(data,sample,sep="_")
                file <- paste(dir,v,paste(tool,"rds",sep="."),sep="_")
                if (tool=='edgeR.exact') {
                    path <- paste("~/RNASeqToolComparison/out",'edgeR',dir,file,sep="/")
                } else {
                    path <- paste("~/RNASeqToolComparison/out",tool,dir,file,sep="/")
                }
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
                    auc <- AUC(des_df$prediction, des_df$actual)
                }
                m[i,] <- c(tool, data, sample, v, FDR, sensitivity, specificity, auc, accuracy)
                i<-i+1
            }
        }
    }
    for (data in data_0_list) {
        for (sample in sample_no) {
            for (v in versions) {
                dir <- paste(data,sample,sep="_")
                file <- paste(dir,v,paste(tool,"rds",sep="."),sep="_")
                if (tool=='edgeR.exact') {
                    path <- paste("~/RNASeqToolComparison/out",'edgeR',dir,file,sep="/")
                } else {
                    path <- paste("~/RNASeqToolComparison/out",tool,dir,file,sep="/")
                }
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
                FPR <- sum(des_df$prediction) / NROW(des_df)
                m0[j,] <- c(tool, data, sample, v, FPR)
                j<-j+1
            }
        }
    }
}
colnames(m) <- c('Tool', 'Data', 'SPC', 'Version', 'FDR', 'Sensitivity', 'Specificity', 'AUC', 'Accuracy')
colnames(m0) <- c('Tool', 'Data', 'SPC', 'Version', 'FPR')
write.table(m, 'out/results/statistics.csv')
write.table(m0, 'out/results/statistics_0.csv')

#df <- as.data.frame(m)
#df0 <- as.data.frame(m0)
df <- read.csv(file = 'out/results/statistics.csv', sep = ' ')
df0 <- read.csv(file = 'out/results/statistics_0.csv', sep = ' ')
df$SPC <- factor(df$SPC, levels = c('2', '5', '10'))
df0$SPC <- factor(df0$SPC, levels = c('2', '5', '10'))
for (data in data_list) {
    df_base <- df[df$Data == data,]
    ggplot(df_base, aes(x=SPC, y=AUC, color=Tool)) + ggtitle(paste('AUC on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/auc/',data,'_auc_plot.png',sep=''))
}

for (data in data_list) {
    df_base <- df[df$Data == data,]
    ggplot(df_base, aes(x=SPC, y=Accuracy, color=Tool)) + ggtitle(paste('Accuracy on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/accuracy/',data,'_accuracy_plot.png',sep=''))
}

for (data in data_list) {
    df_base <- df[df$Data == data,]
    ggplot(df_base, aes(x=SPC, y=FDR, color=Tool)) + ggtitle(paste('FDR on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/fdr/',data,'_fdr_plot.png',sep=''))
}

for (data in data_list) {
    df_base <- df[df$Data == data,]
    ggplot(df_base, aes(x=SPC, y=Sensitivity, color=Tool)) + ggtitle(paste('Sensitivity on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/sensitivity/',data,'_sensitivity_plot.png',sep=''))
}

for (data in data_list) {
    df_base <- df[df$Data == data,]
    ggplot(df_base, aes(x=SPC, y=Specificity, color=Tool)) + ggtitle(paste('Specificity on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/specificity/',data,'_specificity_plot.png',sep=''))
}

for (data in data_0_list) {
    df0_base <- df0[df0$Data == data,]
    ggplot(df0_base, aes(x=SPC, y=FPR, color=Tool)) + ggtitle(paste('False Positive Rate on ',data,sep='')) + geom_boxplot()
    ggsave(paste('out/results/graphs/fpr/',data,'_fpr_plot.png',sep=''))
}
