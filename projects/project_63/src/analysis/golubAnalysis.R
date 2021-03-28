train_stats <- function(data) {
  #find the t statistics, degrees of freedom, and p values of the training set
  g1 = 3:29 #ALL patients
  g2 = 30:40 #AML patients
  t.stat = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$statistic})
  df = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$parameter})
  p.value = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$p.value})
  train_list = list(t.stat, df, p.value)
  names(train_list) = c('t.stat','df','p.value')
  return(train_list)
}

test_stats <- function(data) {
  #find the t statistics, degrees of freedom, and p values of the test set
  g1 = 3:22 #ALL
  g2 = 23:36 #AML
  t.stat = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$statistic})
  df = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$parameter})
  p.value = apply(X=data, MARGIN=1, FUN=function(X){t.test(as.numeric(X[g1]),as.numeric(X[g2]))$p.value})
  test_list = list(t.stat, df, p.value)
  names(test_list) = c('t.stat','df','p.value')
  return(test_list)
}

transform_data <- function(data, train) { #takes data as input, outputs the z-scores
  if (train == TRUE) {
    stats = train_stats(data)
  }
  else {
    stats = test_stats(data)
  }
  t.stat = stats$t.stat
  df = stats$df
  pos.t.stat = t.stat[which(t.stat > 0)]
  pos.df = df[which(t.stat > 0)]
  pos.areas = 1 - pt(pos.t.stat, df=pos.df) #area to the left of each t statistic (with given degrees of freedom)
  pos.z.scores = -1*qnorm(pos.areas) #get respective z scores from each area
  #do the same for negative t statistics
  neg.t.stat = t.stat[which(t.stat < 0)] 
  neg.df = df[which(t.stat < 0)]
  neg.areas = pt(neg.t.stat, df=neg.df)
  neg.z.scores = qnorm(neg.areas)
  z.scores = c(pos.z.scores, neg.z.scores)
  return(z.scores)
}

qq_plot <- function(data, outdir, train, transformed = FALSE) {
  #create qq plots, if transformed = TRUE then create plot for z scores
  if (transformed == TRUE) {
    jpeg(paste(outdir,'qq_plot_transformed.png',sep='/'))
    z.scores = transform_data(data, train)
    qqnorm(z.scores, main=' QQ plot for transformed z scores')
    abline(0,1,lwd=2,col='red')
  }
  else {
    if (train==TRUE) {
      jpeg(paste(outdir, 'qq_plot_train', sep='/'))
      x = train_stats(data)$t.stat
      qqnorm(x, main= ' QQ plot for t statistics')
      abline(0,1,lwd=2,col='red')
    }
    else {
      jpeg(paste(outdir, 'qq_plot_test', sep='/'))
      x = test_stats(data)$t.stat
      qqnorm(x, main= ' QQ plot for t statistics')
      abline(0,1,lwd=2,col='red')
    }
  }
  
  dev.off()
}

hist_p <- function(data, outdir, train) {
  #create histogram for the p-value
  if (train == TRUE) {
    jpeg(paste(outdir,'train_pval_hist.png',sep='/'))
    p.value = train_stats(data)$p.value
  }
  else {
    jpeg(paste(outdir,'test_pval_hist.png',sep='/'))
    p.value = test_stats(data)$p.value
  }
  
  p = seq(0,1,by=0.01)
  hist(p.value, freq=F)
  lines(p, dunif(p), lwd=2, col='red')
  
  dev.off()
}

hist_tstat <- function(data,outdir,train) {
  #create histogram for tstat
  if (train == TRUE) {
    jpeg(paste(outdir,'train_tstat_hist.png',sep='/'))
    t.stat = train_stats(data)$t.stat
  }
  else {
    jpeg(paste(outdir,'test_tstat_hist.png',sep='/'))
    t.stat = test_stats(data)$t.stat
  }
  x = seq(-8,8,by=0.1)
  hist(t.stat, breaks=x, freq=F, ylim=c(0, 0.4)) 
  lines(x, dnorm(x), lwd=2, col='red')
  
  dev.off()
}

estimate_p0 <- function(z.scores) {
  #Estimate p0 to fit the empirical null distribution
  z.index = c()
  interval = 0.2
  for (i in 1:length(z.scores)) {
    if (z.scores[i] > -interval & z.scores[i] < interval)
      z.index <- append(z.index, i)
  }
  intervalZ <- z.scores[z.index]
  
  z.density <- density(z.scores)
  f = c()
  for (z in sort(intervalZ)) {
    temp  <- approx(z.density$x, z.density$y, xout=z)$y
    f <- append(f, temp)
  }
  
  p0 = exp(mean(log(f) - log(dnorm(sort(intervalZ)))))
  return(p0)
}

empirical_p0 <- function(z.scores) {
  z.sd = IQR(z.scores)/1.349
  z.med = median(z.scores)
  
  z.index = c()
  interval = 0.2
  for (i in 1:length(z.scores)) {
    if (z.scores[i] > -interval & z.scores[i] < interval)
      z.index <- append(z.index, i)
  }
  intervalZ <- z.scores[z.index]
  
  z.density <- density(z.scores)
  f = c()
  for (z in sort(intervalZ)) {
    temp  <- approx(z.density$x, z.density$y, xout=z)$y
    f <- append(f, temp)
  }
  
  p0_emp = exp(mean(log(f) - log(dnorm(sort(intervalZ), z.med, z.sd))))
  return(p0_emp)
}

hist_zscores <- function(data, outdir, train){
  #create histogram for zscores
  if (train == TRUE){
    jpeg(paste(outdir, 'train_zscore_hist.png', sep='/'))
  }
  else {
    jpeg(paste(outdir, 'test_zscores_hist.png', sep='/'))
  }

  # Theoretical Null (dotted)
  z.scores = transform_data(data, train)
  x = seq(-5, 5, by = 0.1)
  hist(z.scores, 100, prob=T, ylim=c(0, 0.4)) 
  lines(x, dnorm(x), lwd=2, lty=3, col='red')
  
  # Theoretical null scaled (dashed)
  p0 = estimate_p0(z.scores)
  lines(x, p0*dnorm(x), lwd = 2, lty=2, col= 'red')
  
  # Empirical Null (solid)
  p0_emp = empirical_p0(z.scores)
  z.sd = IQR(z.scores)/1.349
  z.med = median(z.scores)
  lines(x, p0_emp*dnorm(x, z.med, z.sd), lwd = 2, col='red')
  
  legend(1.4, 0.4, legend = c(
    "Theoretical Null Distribution", 
    "Scaled Theoretical Null", 
    "Empirical Null Distribution"), lty=c(3,2,1), cex=0.8, col='red')
  
  dev.off()
}

plot_metrics <- function(data, outdir, train) {
  #Plot the TPR, FPR, and FDR of both theoretical and empirical null
  
  z.scores = transform_data(data, train)
  z.sd = IQR(z.scores)/1.349
  z.med = median(z.scores)
  x = seq(-5,5,by=0.1)
  m = length(z.scores)
  # Under theoretical null N(0,1)
  p_null = 1-pnorm(x)
  p_hat = rep(0, length(x))
  
  for (i in 1:length(x)) {
    p_hat[i] = sum((z.scores) > x[i])/m
  }
  p0 = estimate_p0(z.scores)
  p0_emp = empirical_p0(z.scores)
  # Theoretical
  fpr = p_null
  tpr = (p_hat-p0*p_null)/(1-p0)
  fdr = p0*p_null/p_hat
  # Empirical
  p_emp_null = 1 - pnorm(x, z.med, z.sd)
  fpr_emp = p_emp_null
  tpr_emp = (p_hat-p0_emp*p_emp_null)/(1-p0_emp)
  fdr_emp = p0_emp*p_emp_null/p_hat
  
  # Plot TPR
  if (train == TRUE){
    jpeg(paste(outdir, 'fpr_train.png', sep='/'))
  }
  else {
    jpeg(paste(outdir, 'fpr_test.png', sep='/'))
  }
  plot(x, fpr, ylim = c(0,1), type = 'l', lwd=2, col='red', main = 'False Positive Rate', xlab="z-scores threshold")
  lines(x, fpr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
  legend(2, 0.95, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)
  
  dev.off()
  
  # Plot FPR
  if (train == TRUE){
    jpeg(paste(outdir, 'tpr_train.png', sep='/'))
  }
  else {
    jpeg(paste(outdir, 'tpr_test.png', sep='/'))
  }
  plot(x, tpr, ylim = c(0,1.5), type = 'l', lwd=2, col='red', main = 'True Positive Rate', xlab="z-scores threshold")
  lines(x, tpr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
  legend(2, 1.4, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)
  
  dev.off()
  
  # Plot FDR
  if (train == TRUE){
    jpeg(paste(outdir, 'fdr_train.png', sep='/'))
  }
  else {
    jpeg(paste(outdir, 'fdr_test.png', sep='/'))
  }
  plot(x, fdr, ylim = c(0,1), type = 'l', lwd=2, col='red', main = 'False Discovery Rate', xlab="z-scores threshold")
  lines(x, fdr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
  legend(-4.5, 0.2, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)
  
  # Plot AOC ROC
  if (train == TRUE){
    jpeg(paste(outdir, 'roc_train.png', sep='/'))
  }
  else {
    jpeg(paste(outdir, 'roc_test.png', sep='/'))
  }
  plot(x, fpr, ylim = c(0,1), type = 'l', lwd=2, col='red', main = 'False Positive Rate', xlab="z-scores threshold")
  lines(x, fpr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
  legend(2, 0.95, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)
  
  dev.off()
  
  dev.off()
}




generate_plots_golub <- function(data, outdir, train) {
  #generate plots and save in output directory
  hist_tstat(data, outdir,train)
  hist_p(data, outdir, train)
  hist_zscores(data, outdir, train)
  qq_plot(data, outdir, train)
  qq_plot(data, outdir, train, transformed=TRUE)
  plot_metrics(data, outdir, train)
}