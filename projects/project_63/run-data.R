library("rjson")

source("src/data/etl.R")
source("src/analysis/GolubAnalysis.R")
source("src/cleaning/GolubCleaning.R")


main <- function(target) {
  #Runs the main project pipeline.
  #params: target: string must contain 'data','analysis','model'.
  
  if(grepl('data',target, fixed =TRUE)) {
    data_cfg<- fromJSON(file='config/data-params.json')
    clean_data()
    df1 = get_data(data_cfg$datadir1)
    df2 = get_data(data_cfg$datadir2)
  }
  #should work
  if(grepl('analysis',target, fixed=TRUE)) {
    analysis_cfg<- fromJSON(file='config/analysis-params.json')
    generate_plots_golub(df1, analysis_cfg$outdir, train=FALSE)
    generate_plots_golub(df2, analysis_cfg$outdir, train=TRUE)
  }
}

if (!interactive()) {
  target = commandArgs(trailingOnly=TRUE)
  main(target)
}
