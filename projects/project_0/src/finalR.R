calculate_p0 <- function(gender) {
  data = data.matrix(cardio_train['height'])
  data = data[cardio_train['gender'] == gender]
  
  data
  data.hist <- hist(data, 100, probability = T)
  z <- data.hist$mids
  
  q1 = quantile(data,0.25)
  q1
  q3 = quantile(data,0.75)
  q3
  index = which(z > q1-iqr & z < q3+iqr)
  
  plot(z, data.hist$density, type='h')
  
  x = z[index]
  y = data.hist$density[index]
  
  median_height = median(data)
  std_height = IQR(data)/1.349
  
  plot(x, log(y), xlab = 'height', ylab='log density')
  legend("topleft", legend=c("Theoretical", "p0*Theoretical"),
         col=c("red", "green"), lty=1:1, cex=0.8)
  lines(x, log(dnorm(x, mean=median_height, sd=std_height)), col='red', lwd=2)
  
  log.p0 = mean(log(y)-log(dnorm(x, mean=median_height, sd=std_height)))
  std = sd(log(y)-log(dnorm(x, mean=median_height, sd=std_height)))/sqrt(length(x))
  
  p0 = exp(log.p0)
  lines(x, log(p0*dnorm(x, mean=median_height, sd=std_height)), col='green', lwd=2)

  print(p0)
  print(std)
  
  
}

calculate_p0(1)
calculate_p0(2)


fdr_fpr <- function(gender) {
  data = data.matrix(cardio_train['height'])
  data = data[cardio_train['gender'] == gender]
  
  data.hist <- hist(data, 100, probability = T)
  z <- data.hist$mids
  
  q1 = quantile(data,0.25)
  q1
  q3 = quantile(data,0.75)
  q3
  index = which(z > q1-iqr & z < q3+iqr)
  
  plot(z, data.hist$density, type='h')
  
  x = z[index]
  y = data.hist$density[index]
  
  median_height = median(data)
  std_height = IQR(data)/1.349
  m = length(data)
  
  Z.expected = p0*m*(1-pnorm(z, mean=median_height,sd=std_height))
  Z.obtained = rep(0, length(z))
  for(i in 1:length(z)) {
    Z.obtained[i] = sum(data > z[i])
  }
  
  plot(z, Z.expected/(p0*m), col='red', type='l', lwd=2, ylab='FPR')
  lines(z, Z.expected/Z.obtained, col='black', type='l', lwd=2, ylab='FDR')
}
fdr_fpr(1)
