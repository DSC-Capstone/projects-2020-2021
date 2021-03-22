# DSC180B-Capstone-Project
DSC Capstpne Project: A Prediction Model for Battery Remaining Time
## Usage Instructions
We provided 4 targets for your usage. They are `data`, `eda`, `model`, and `test`.

`test` would run our project on test data, which provides a miniature of what we have done on a smaller dataset. 

An exmple for running our project through terminal is `python run.py data`, which will show our main dataset. To run other branches, just replace `data` with `eda`, `model` or`test`.


## Description of Contents
```
PROJECT
├── config
    └── data-params.json
    └── inputs.json
├── notebooks
    └── DSC180B_Presentation.ipynb
├── references
    └── README.md
└── src
    ├── EDA
        └──  feature_selection.py
    ├── data
        ├── Loading_Data.py
        ├── minimini_battery_event2.csv
        ├── minimini_battery_info2.csv
        ├── minimini_device_use1.csv
        ├── minimini_device_use2.csv
        ├── minimini_hw1.csv
        ├── minimini_hw2.csv
        ├── minimini_process1.csv
        └──minimini_process2.csv
    └── model
        └──hypothesis_testing.py
├── .gitignore
├── README.md
└── run.py
└── submission.json
```


### `config/`
* `data-params.json`: It contains the file path for our dataset
* `inputs.json`: It contains the argument inputs

### `notebooks/`
* `DSC180B_Code.ipynb`: EDA, Hypothesis Testing and Visual Presentation on our project

### `references/`
* `README.md`: External sources

### `src/`
* `EDA/feature_selection.py`: code for selecting desired features for our prediction model
* `data/Loading_Data.py`: code for load our dataset
* `data/minimini_[DIFFERENT FILE NAME]_csv`: sample dataset
* `model/hypothesis_testing.py` : code for prediction model and hypothesis testing

### `run.py`
* Main driver for this project.
