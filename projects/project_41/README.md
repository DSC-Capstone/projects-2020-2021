# AutoPhrase for Financial Documents Interpretation 

Our main targets are data preparation, feature encoding, eda (optional), train, report (optional), and test. Users can configure parameters for these targets in the ./config files.


## Data Prep

The data preparation target scrapes, cleans, and consolidates companies' 8-K documents. Furthermore, it curates features such as EPS as well as price movements for the given companies.
<br />
* `data_dir` is the file path to download files: 8-K's, EPS, etc.
* `raw_dir` is the directory to the raw data
* `raw_8k_fp` is the file path with newly downloaded 8-K's (should be the same as to_dir)
* `raw_eps_fp` is the file path with newly downloaded EPS information (should be the same as to_dir)
* `processed_dir` is the directory to the processed data
* `testing` is the status of whether we are doing testing (by default is false)


## Feature Encoding

The feature encoding target creates encoded text vectors for each 8-K: both unigrams and quality phrases outputed by the AutoPhrase method.
<br />
* `data_file` is the file path with all data files from data prep target: processed, raw, models, etc.
* `phrase_file` is the file path to the quality phrases outputted by AutoPhrase
* `n_unigrams` sets the top n unigrams to be encoded based on PMI (may not be exacly `n_unigrams` total due to overlap of top unigrams within each class)
* `threshhold` takes quality phrases with a quality score above it to be encoded


## Train

Trains Random Forest models using 3 feature sets on encoded data: baseline, baseline + unigrams, and baseline + phrases. The selected classifier and set model parameters were decided through comparing validation accuracy.  

* `data_dir` is the file path with all data files from data prep targed: processed, raw, models, etc.
* `input_file` is the file path (from `data_dir`) to outputed files by the feature encoding target
* `output_file` is the desired file path to download trained, outputed models
* `testing` is the status of whether we are doing testing (by default is false)


## EDA and Report (optional)

Exports Jupyter notebooks to HTML with EDA and result analysis from the models.
<br />
* `report_name` is the desired name of report
* `data_dir` is the file path with all data files from data prep target: processed, raw, models, etc.
* `notebook_dir` is the file path containing the repo's notebooks
* `notebook_file` is the desired file path (from `notebook_dir`) of outputed notebook
* `report_dir` is the desired directory our outputed HTML report
* `report_file` is the desired file name of outputed report in `report_dir`


## Test

Test target will run the whole project with only test data


## Correct order of excution

* data_prep: `python run.py data_prep`
* feature_encoding: `python run.py feature_encoding`
* (optional) eda: `python run.py eda`
* train: `python run.py train`
* (optional) report: `python run.py report`


## Project Links

* [Project Website](https://shy218.github.io/dsc180-project/)
* [AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase)
