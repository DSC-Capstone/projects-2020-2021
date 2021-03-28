library(data.table)
library("rjson")


clean_actual <- function(datadir) {
  df_actual <- read.table(file=datadir, sep=',', skip=1)
  names(df_actual)[names(df_actual) == "V1"] <- "patient"
  names(df_actual)[names(df_actual) == "V2"] <- "cancer"
  return(df_actual)
}

clean_independent <- function(datadir) {
  df_independent <- read.table(file=datadir, sep=',')
  old_col = c(
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','V29','V30',
    'V31','V32','V33','V34','V35','V36','V37','V38','V39','V40',
    'V41','V42','V43','V44','V45','V46','V47','V48','V49','V50',
    'V51','V52','V53','V54','V55','V56','V57','V58','V59','V60',
    'V61','V62','V63','V64','V65','V66','V67','V68','V69','V70')
  new_col = as.character(df_independent[1,])
  setnames(df_independent, old=old_col, new=new_col)
  df_independent <- df_independent[-c(1),]
  return(df_independent)
}

clean_train <- function(datadir) {
  df_train <- read.table(file=datadir, sep=',')
  old_col = c(
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','V29','V30',
    'V31','V32','V33','V34','V35','V36','V37','V38','V39','V40',
    'V41','V42','V43','V44','V45','V46','V47','V48','V49','V50',
    'V51','V52','V53','V54','V55','V56','V57','V58','V59','V60',
    'V61','V62','V63','V64','V65','V66','V67','V68','V69','V70',
    'V71','V72','V73','V74','V75','V76','V77','V78')
  new_col = as.character(df_train[1,])
  setnames(df_train, old=old_col, new=new_col)
  df_train <- df_train[-c(1),]
  return(df_train)
}

clean_data <- function () {
  data_cfg<- fromJSON(file='config/data-params.json')
  df_actual = clean_actual(data_cfg$raw_actual)
  df_independent = clean_independent(data_cfg$raw1)
  df_train = clean_train(data_cfg$raw2)
  df_1 <- data.frame(df_independent$`Gene Description`, df_independent$`Gene Accession Number`)
  df_2 <- data.frame(df_train$`Gene Description`,df_train$`Gene Accession Number`)
  
  #function to return type of cancer given patient's number
  find_cancer <- function(patient){
    return(df_actual$cancer[df_actual$patient==patient])
  }
  
  #Append columns to the first dataset and sort by cancer types
  pat_1 <- colnames(df_independent)[3:length(colnames(df_independent))]
  for (x in pat_1){
    if(x!='call'){
      if(find_cancer(x)=='ALL'){
        df_1[[x]] = df_independent[[x]]
      }
    }
  }
  for (x in pat_1){
    if(x!='call'){
      if(find_cancer(x)=='AML'){
        df_1[[x]] = df_independent[[x]]
      }
    }
  }
  #For dataset 1, the first 20 columns(col3 to 22) are ALL while the last 14 (col23 to 36) are AML
  
  #Append columns to the second dataset and sort by cancer types
  pat_2 <- colnames(df_train)[3:length(colnames(df_train))]
  for (x in pat_2){
    if(x!='call'){
      if(find_cancer(x)=='ALL'){
        df_2[[x]] = df_train[[x]]
      }
    }
  }
  for (x in pat_2){
    if(x!='call'){
      if(find_cancer(x)=='AML'){
        df_2[[x]] = df_train[[x]]
      }
    }
  }
  #For dataset 2, the first 27 columns(col3 to 29) are ALL while the last 11 (col30 to 40) are AML
  
  #save the datasets
  write.table(df_1, data_cfg$datadir1,row.names = TRUE, col.names = TRUE)
  write.table(df_2, data_cfg$datadir2,row.names = TRUE, col.names = TRUE)
  write.csv(df_1, data_cfg$datadir1csv,row.names = TRUE)
  write.csv(df_2, data_cfg$datadir2csv,row.names = TRUE)
}