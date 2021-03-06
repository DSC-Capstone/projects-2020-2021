# DSC180B_Capstone_Project
Ryan Cummings,
Gregory Thein,
Justin Kang,
Prof. Shannon Ellis,
Code Artifact Checkpoint

#### In this you will find our Checkpoint Code, We are in the B04 Genetics domain and this is our Capstone Project. For our Capstone Project we are looking at Alzheimer's Diseased Patient's Blood miRNA Data. Our Pipeline functions are seen in the all_pipeline.py file. Running the full pipeline takes multiple hours to run and implements the tools in our Genetics Pipeline (FastQC, CutAdapt, Kallisto, DESeq2). Our project implements both python and R to perform successful analysis on our dataset of blood based miRNA in which we find miRNAs with significantly changed expression level.

#### Our repo consists of 4 folders, and 3 files (a .gitignore, the README, and the run.py). The 4 folders consist of: config, notebooks, references, src. Inside config is our data-params.json file, eda-params.json file, report-params.json, analyze-params.json, viz-params.json, and test-params.json. These files specifies the data-input and output locations/file paths that is necessary for this Checkpoint's data retrieval. The eda-params file specifies the input and output of the report generated by the `eda` call, while the test-params has the names of the samples that we run the `test` keyword argument on, report-params has the input and output locations of the full report that is generated at the end of the `all` call, analyze-params has the the filepaths for the input/output of the analyze notebook that is ran when `analyze` is passed as a target, viz-params has the locations for the notebook that is generated when the `viz` param is called. Notebooks folder consists of all of our .ipynb files that we used for testing, and as a dev tool (to see what we did along the way). It also contains the notebooks that are converted for each of the targets that can be passed into our program. References has our SRARunTable from the patients we used in our project, and also contains static images that are loaded for some of the notebooks when converting to output report. The data folder is where we created the symlink between our folder and the dataset on DSMLP. The data folder (and test/testdata) will also consist of the data/out information once the `test` keyword is ran, specifically the output from Kallisto is stored here. The contents of our src folder contains our etl.py file, eda.py file, utils.py file, test_pipeline.py, and all_pipeline.py. Our etl.py file is where our file is extracting the dataset from the DSMLP's /teams dataset. Utils.py is where we created a function that turns a notebook into an HTML format, which then outputs that HTML file as a report. test_pipeline and all_pipeline contain the pipeline that is created for our project, varying slightly since test is only ran on a portion while all is ran on the entire dataset!

### Project Decisions

- Our project focus shifted from looking at gene expression data for Alzheimer's Disease patients, to observing blood sample data of patients diagnosed with Alzheimer's Disease and a control group. This was done in large part because of the lack of access to the databases we initially wanted to retrieve data from
- After spending time searching for a viable replacement dataset on Recount2, we set on data from SRA Study SRP022043 and downloaded the data onto DSMLP from the SRA Run Selector Tool 
- We initially implemented the dockerfile for this project based on the dockerfile used in last quarters replication and had hoped to implement TrimGalore as a new tool into our pipeline. Incompatibility issues, however, led us to drop TrimGalore as tool and stick with running Cutadapt and FastQC separately.
- The Kallisto reference file was originally stored in our data file in our Github but the `.gitignore` was hiding that file when we would pull the repo. We need it in order to run Kallisto so we moved it to our teams directory on DSMLP.


### Project Targets:
#### all
Runs entire pipeline on all of the data. Running `all` will run the full pipeline from scratch, this does take hours and sometimes even days to run, it can be ran from scratch but is not needed to be ran from scratch to see our results!
```
{
    "outdir": "data/report",
    "report_in_path": "notebooks/Alzheimers-Biomarker-Analysis.ipynb",
    "report_out_path": "report/Alzheimers-Biomarker-Analysis.html"
}
```
#### test
Runs part of pipeline on a couple fastq files. Implements fastqc and kallisto. Then generates this report!
```
{
  "test_1": "SRR837440.fastq.gz",
  "test_2": "SRR837444.fastq.gz"
}
```
#### data
In Progress! Gets and outputs the data and generates the report as well!
```
{
  "file_path": "/teams/DSC180A_FA20_A00/b04genetics/group_1/raw_data"
}
```
#### eda
Runs EDA process. Makes report with data and plots figures.
```
{
    "outdir": "data/report",
    "report_in_path": "notebooks/EDA.ipynb",
    "report_out_path": "notebooks/EDA.html"
}
```
#### viz
Runs Visualization process. Simply outputs all the charts and graphs used in the project.
```
{
    "outdir": "data/report",
    "report_in_path": "notebooks/Viz.ipynb",
    "report_out_path": "notebooks/Viz.html"
}
```

#### analyze
Runs the Notebook used for our Analysis portion of the project. Generating the plots that are used to explain our results.
```
{
    "outdir": "data/report",
    "report_in_path": "notebooks/analyze.ipynb",
    "report_out_path": "notebooks/analyze.html"
}
```


#### Running `python run.py all` will run the full pipeline from scrath, this does take hours and sometimes even days to run, it can be ran from scratch but is not needed to be ran from scratch to see our results! Other keywords that can be passed into the funciton are `test eda data viz analyze`. Running `python run.py test` is actually the most recommended one, this gives you the full pipeline experience on a fraction of the data, running in just a few minutes. Portions of the code can also be ran with `python run.py data` or `python run.py eda` or a combination of these: `python run.py data eda` etc. We also printed steps along the way to notify the user what is currently running in the pipeline. Our code assumes it is ran on the DSMLP Servers! Without running on the DSMLP Servers we would not be able to access the data, which is why it is important to be connected to the server.



### Responsibilities

Ryan: 
Ryan created the Pipeline that we are using for our project so far: FastQC, Cutadapt, FastQC (2), and Kallisto. Along with formatting the Github repo to the Cookiecutter Data Science standard. 

Justin: 
Justin worked mainly on getting the report side of the project complete. He, alongside Gregory, spent time researching what MicroRNA and biomarkers are to include as part of our background. Researching additional information about Alzheimer???s Disease was also completed. He also worked on getting the initial structure of the report completed prior to the checkpoint. 

Gregory: 
Gregory, alongside Justin worked on the researching miRNA and biomarkers, and their relation to AD. Furthermore, he helped research various parameters and settings for parts of the pipeline. 

All assisted in the implementation of the pipeline alongside editing/reviewing each other???s work. 
