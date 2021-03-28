# Recipe-Recommendation

## Introduction
This repository contains the code and models needed to create a recommender system for recipe recommendations. Included in the repository are a few simple baselines as well as the final model utilized by our recommender system for recipe recommendation. The data for this project was pulled from Kaggle (https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) and (https://www.kaggle.com/kaggle/recipe-ingredients-dataset/home).

##  Files

For this project, we have files for running the code, retriving the data, and processing it into the desired output.

There are several files that will be obtained from Kaggle, but are not part of this repository. This is because they are data files and are too large to be version controlled.

The following files were created by us in order to create and run our baselines and final model.

> run.py: Passes in the location of the data folder. Runs etl on the Kaggle data and stores the data in the data output folder. Runs baselines and model on the dataset. Evaluates models to see how well the baselines did in comparison to the final model.

> mostPop.py: Baseline file that uses a top popular model to determine what users would rate certain recipes.

> randomFor.py: Baseline file that uses a random forest model to determine what cuisine type each recipe would fall under.

> conBased.py: Baseline file that uses a content based model with cosine similarity to determine what recipe to recommend based on what ingredients are listed by the user.

> requirements.txt: Contains the amount of processing resources recommended to run the files within a few hours each, and the packages needed and the versions that were used to run all the processes.

> Preliminary EDA.ipynb: Inside the notebook directory. Notebook containing the exploratory data analyses that was taken on the data to further understand and gain insight on the data we were using.

##  Directories

The following directories were created by us in order to be able to store and retain the necessary information needed for different purposes.

> config: Contains a list of all the config files that determines the parameters of each file. Use these files according to their use to change the parameters and change which subset of data you are running the processes on. Make sure you are changing the file paths correctly and throughout the entire config file.

> data: Contains the data retrieved after running the repository, and the cleaned and processed data by us.

> testData: Contains the randomly generated testData that we would run our models against to see how well they performed. Much smaller than full dataset, and allows for easy tracking to gain the most insight from how the model works.

## Running the Code
Prior to running the code, make sure that you install all the packages listed in *requirements.txt* 

In order to obtain the data, one can simply run the run.py python file with the command data or all. This will prompt the script to download the data and store it in the proper place for you.

### Creating the Data

To create the processed data, run this following command on the command line terminal:
```
python run.py data
```
Where the data will be returned into new files usable by our models in this project and placed in the data directory.

### Running all the model targets

If you want to run all of these together, run this following command on the command line terminal:
```
python run.py all
```
Where the all targets (excluding *test*) will run one after another.

### Testing all the model targets

To test how if the repository and all the models and scripts are working, run this following command on the command line terminal
```
python run.py test
```
Where the targets above will all be run one after another, but on small test data so that we can observe how the models and scripts are working.

## Repository Organization

To ensure the code runs properly, it must have the same folders and files locations. etl.py must be inside src and data folders, and data must be in the data folder. mostPop.py, must be in the src and baselines folders.
