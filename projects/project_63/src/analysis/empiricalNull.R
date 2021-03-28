source("src/analysis/GolubAnalysis.R")
#Load dataset
#Testing
df1 <- read.table('data/cleaned/golub1')
#Training
df2 <- read.table('data/cleaned/golub2')

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

estimate_p0(transform_data(df1, FALSE))
estimate_p0(transform_data(df2, TRUE))



t.stat = test_stats(df2)$t.stat
p.value = test_stats(df2)$p.value
z.scores = transform_data(df2,TRUE)

p = seq(0,1,by=0.01)
hist(p.value, freq=F)
lines(p, dunif(p), lwd=2, col='red')

x = seq(-8,8,by=0.1)
hist(t.stat, 100, prob=T, ylim=c(0,0.4)) 
lines(x, dnorm(x), lwd=2, col='red')

x = seq(-5,5,by=0.1)
hist(z.scores, 100, prob=T, ylim=c(0,0.4)) 
lines(x, dnorm(x), lwd=2, col='red')

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
hist(z.scores, 100, prob=T, ylim=c(0,0.4))
lines(z.density, col='red')

plot(sort(intervalZ), log(f), ylim=c(-1.5, -0.7))
lines(sort(intervalZ), log(dnorm(sort(intervalZ))), col='red')

p0 = exp(mean(log(f) - log(dnorm(sort(intervalZ)))))
p0


#Estimation with Median and IQR
t.stat = test_stats(df1)$t.stat
p.value = test_stats(df1)$p.value
z.scores = transform_data(df1,FALSE)

x = seq(-8,8,by=0.1)
hist(t.stat, 100, prob=T, ylim=c(0,0.4)) 
lines(x, dnorm(x), lwd=2, col='red')

z.sd = IQR(z.scores)/1.349
# (quantile(z.scores, 0.6) - quantile(z.scores, 0.4))/(qnorm(0.6) - qnorm(0.4))

z.med = median(z.scores)

#Function to find the mode of data
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
## Use mode of the smoothed histogram curve
z.mode = getmode(z.scores)
#Treat the mode as the median and find the 25th and 75th percentile
z.lower = c()
z.upper = c()
for (i in z.scores) {
  if (i < z.mode) {
    z.lower <- append(z.lower, i)
  } 
  else{
    z.upper <- append(z.upper, i)
  }
}
qrt25 = median(z.lower)
qrt75 = median(z.upper)
z.sd_mode = (qrt75-qrt25)/1.349

x = seq(-5,5,by=0.1)
hist(z.scores, 100, prob=T, ylim=c(0,0.4))
lines(x, dnorm(x, z.med, z.sd), lwd=2, col='red')
lines(x, dnorm(x, z.mode, z.sd_mode), lwd=2, col='blue')

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
plot(sort(intervalZ), log(f), ylim=c(-1.5, -1.2))
lines(sort(intervalZ), log(dnorm(sort(intervalZ), z.med, z.sd)), col='red')

p0_emp = exp(mean(log(f) - log(dnorm(sort(intervalZ), z.med, z.sd))))
p0_emp

x = seq(-5, 5, by = 0.01)
hist(z.scores, 100, prob=T, ylim=c(0, 0.4)) 
# Theoretical null N(0,1) (dotted)
lines(x, dnorm(x), lwd=2, lty=3, col='red')
# Theoretical null N(0,1) scaled by p0 = 0.61 (dashed)
lines(x, p0*dnorm(x), lwd = 2, lty=2, col= 'red')
# Empirical null N(-0.07, 1.53^2) (solid)
lines(x, p0_emp*dnorm(x, z.med, z.sd), lwd = 2, col='red')
legend(1.4, 0.4, legend = c("Theoretical Null Distribution", "Scaled Theoretical Null", "Empirical Null Distribution"), lty=c(3,2,1), cex=0.8, col='red')

# FDR Estimation
m = length(z.scores)
# Under theoretical null N(0,1)
p_null = 1-pnorm(x)
p_hat = rep(0, length(x))
for (i in 1:length(x)) {
  p_hat[i] = sum((z.scores) > x[i])/m
}
# Calculate FPR, TPR, and FDR
# Theoretical
fpr = p_null
tpr = (p_hat-p0*p_null)/(1-p0)
fdr = p0*p_null/p_hat
# Empirical
p_emp_null = 1 - pnorm(x, z.med, z.sd)
fpr_emp = p_emp_null
tpr_emp = (p_hat-p0_emp*p_emp_null)/(1-p0_emp)
fdr_emp = p0_emp*p_emp_null/p_hat

# Plot FPR, TPR, and FDR
plot(x, fpr, ylim = c(0,1), type = 'l', lwd=2, col='red', main = 'False Positive Rate')
lines(x, fpr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
legend(2, 0.95, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)

plot(x, tpr, ylim = c(0,1.5), type = 'l', lwd=2, col='red', main = 'True Positive Rate')
lines(x, tpr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
legend(2, 1.4, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)

plot(x, fdr, ylim = c(0,1), type = 'l', lwd=2, col='red', main = 'False Discovery Rate')
lines(x, fdr_emp, ylim=c(0,1), type='l', lwd=2, col='blue')
legend(-4.5, 0.2, legend = c("Theoretical", "Empirical"), col = c('red', 'blue'), lty=1:1, cex=0.8)

#ROC
plot(fdr, fpr, ylim=c(0,1), type='l', lwd=2, col='red', main="ROC Curve")
lines(fdr_emp, fpr_emp, lwd=2, col='blue')

# Under empirical null N(-0.066, 1.534)


plot(x, tpr_emp, ylim=c(0,1.5))

plot(x, fdr_emp, type='l', lwd=2)


#Theoretical null yields appearant low error rate but may  be wrong
#Because additional variance in histogram due to confounders (factors)
#Empirical adjust for those unknown factors empirically
#Lower detection power but the error rate may be more realistic (closer to the true error rate)
#Classfiy subject (supervised) vs classify genes (unsupervised) requires different methods

