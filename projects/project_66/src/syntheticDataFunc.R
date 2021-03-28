#command arguments from config/build-params.json which builds the synthetic data
myArgs <- commandArgs(trailingOnly = TRUE)

#install necessary packages
library('compcodeR')

data <- myArgs[1]
num_vars <- strtoi(myArgs[2])
samples_per_cond <- strtoi(myArgs[3])
rep_id <- strtoi(myArgs[4])
num_diff_exp <- strtoi(myArgs[5])
ratio_upregulated <- as.numeric(myArgs[6])
dispersion_num <- as.numeric(myArgs[7])
outlier_type <- myArgs[8]
outlier_ratio <- as.numeric(myArgs[9])
output_dir <- myArgs[10]
seq_depth <- as.numeric(myArgs[11])

#create the synthetic data with desired distribution of counts (poisson or baseline) with 0 outliers
if (outlier_type == 'none') {
    syntheticData <- generateSyntheticData(dataset = data, n.vars = num_vars, samples.per.cond = samples_per_cond, repl.id = rep_id, seqdepth = seq_depth, n.diffexp = num_diff_exp, fraction.upregulated = ratio_upregulated, fraction.non.overdispersed = dispersion_num, between.group.diffdisp = FALSE, filter.threshold.total = rep_id, filter.threshold.mediancpm = dispersion_num, output.file = output_dir)
    }

#create the synthetic data which selects a single sample & multiplies the corresponding count with a factor between 5 & 10
if (outlier_type == 'single') {
    syntheticData <- generateSyntheticData(dataset = data, n.vars = num_vars, samples.per.cond = samples_per_cond, repl.id = rep_id, seqdepth = seq_depth, n.diffexp = num_diff_exp, fraction.upregulated = ratio_upregulated, fraction.non.overdispersed = dispersion_num, between.group.diffdisp = FALSE, filter.threshold.total = rep_id, filter.threshold.mediancpm = dispersion_num, single.outlier.high.prob = outlier_ratio, single.outlier.low.prob = outlier_ratio, output.file = output_dir)
    }

#create the synthetic data which selects a random sample & multiplies the corresponding count with a factor between 5 & 10
if (outlier_type == 'random') {
    syntheticData <- generateSyntheticData(dataset = data, n.vars = num_vars, samples.per.cond = samples_per_cond, repl.id = rep_id, seqdepth = seq_depth, n.diffexp = num_diff_exp, fraction.upregulated = ratio_upregulated, fraction.non.overdispersed = dispersion_num, between.group.diffdisp = FALSE, filter.threshold.total = rep_id, filter.threshold.mediancpm = 0, random.outlier.high.prob = outlier_ratio, random.outlier.low.prob = outlier_ratio, output.file = output_dir)
    }
