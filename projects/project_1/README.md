# DSC180B-Capstone-Project
- dataset available for download at https://www.kaggle.com/crawford/gene-expression
  - #Make sure to unzip files into 'data/raw' folder#
- get data: in the command line enter `Rscript run-data.R data`
- analysis: in the command line enter `Rscript run-data.R analysis`
  - the resulting graphs will be in the data/out folder 

- data: contains the raw and cleaned versions of the datasets we're working with. Also will hold the graphs from analysis
- src: contains the analysis, cleaning, and data etl scripts.
  - analysis: golubAnalysis.R contains the script we used to do tests and generate plots
  - cleaning: golubCleaning.R contains the script we used to clean the raw datasets found in data/raw
  - data: etl.R contains the scirpt to extract the data for run-data.R
  
Acknowledgements
- Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression

  - Science 286:531-537. (1999). Published: 1999.10.14

  - T.R. Golub, D.K. Slonim, P. Tamayo, C. Huard, M. Gaasenbeek, J.P. Mesirov, H. Coller, M. Loh, J.R. Downing, M.A. Caligiuri, C.D. Bloomfield, and E.S. Lander
