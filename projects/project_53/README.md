# ForumRec

## Introduction
This repository is for the ForumRec project, a recommendation system, that recommends users questions they are adept to answer on the [Super User](superuser.com) forum of [StackExchange](https://stackexchange.com). The website can be reached at [jackzlin.com](jackzlin.com) and the repository for the website can be reached at [ForumRecWeb](https://github.com/okminz/ForumRecWeb).

##  Files

For this project, we have files for retriving the data, running the models, and processing it into the desired output. The files are described below and explain the purpose of each part of the repository.

> etl.py: Passes in the configs file related to it. The process for taking the data from the data files, extracting the necessary information, and splitting them into questions and answers after a certain date. This is to maintain a higher predictive function and while utilizing as much of the information as e can after extracting.

> api_etl.py: Passes in the configs file path related to it. It uses the [Stack Exchange API](https://api.stackexchange.com/) on the Super User forum to gather recent questions and answers from the Super User forum, concatenates the questions and answers and saves them into a continuously updating file so it can be used for new recommendations.

> run.py: Runs etl on the data. Runs api_etl on the api parameters. Runs the hybrid model (collaborative and content-based filterling) creation and recommendation files creation along with evaluating the model. Runs the baseline file to get the baseline comparison values. Can also run all of these files on test data.

> create_model.py: Passes in the configs file related to it. This file will run create_inputs.py and gather inputs from that file. It will then use those inputs to a generate a model. 

> create_inputs.py: Passes in the configs file related to it. This file will use processed questions and answers to generate the necessary matrices and files and save them so that they may be used in create_model.py to create a model.

> model.py: Passes in the configs file related to it a boolean parameter to determine if running for baselines (default is False). This file the generated model and gather its inputs in order to make recommendations for based on the interactions between users, questions, answers, text, and tags. These recommendations will be returned in a file. It will also produce the evaluations of the model using precision, recall, auc score, and recriprical rank.

> new_user.py: Passes in the configs file related to it. This file will take in user response data from the website in order to gather new and fresh recommendations for the user by fitting partially to the already generated model and replacing it. This script is used for the website's cold start function mostly. 

> create_baseline.py: Passes in the configs file related to it. Baseline file that return the recommendations given to a user using a simple collaborative filtering model so that it can be compared against the model's recommendations and evaluation metrics.

> requirements.txt: Contains the amount of processing resources recommended to run the files and the packages needed and the versions that were used to run all the processes.

> LICENSE: A file that contains the reuse and use licenses for this repository.

> SuperUser EDA.ipynb: Inside the notebook directory. Notebook containing the exploratory data analyses that was taken on the data to further understand and gain insight on the data we were using, and how we can use the data to build the recommendation system we wanted.
> 
> SuperUser API.ipynb: Inside the notebook directory. Notebook containing the explorationa and trials of using the Stack Exchange API to further understand and gain insight on how to get the desire questions and answers from it. It was used to build the api-etl process.

> ETL.ipynb: Inside the notebook directory. Notebook containing the etl processes and the results to look at the etl.

> NLP.ipynb: Inside the notebook directory. Notebook containing the natural language processing code that looks at the text data and creates a model through that code.

##  Directories

The following directories were created by us in order to be able to store and retain the necessary information needed for different purposes.

> config: Contains a list of all the config files that determines the parameters of each file. Use these files according to their use to change the parameters and change which subset of data you are running the processes on. Make sure you are changing the file paths correctly and throughout the entire config file.

> data: Location to put the original raw data in. It also would contain a final directory which would contain the data retrieved after running the repository.

> test: Contains inside the data directory and inside that the data used to run the repository on a small subset of the data to ensure the models and the scripts are running correctly.

> src: Contains inside the src, models, and baselines folders which contain the etl, model creation, input creation, model, new user, and baselines python files that are described above.

> notebooks: Contains multiple notebooks that explore the data. The notebooks are described above.

## Running the Code
Prior to running the code, make sure that you install all the packages listed in *requirements.txt* 

In order to obtain the data, one can follow the processes below:

### Creating the Data

To create the processed data, run this following command on the command line terminal:
```
python run.py data
```
Where the data will be replaced processed and be returned into new files usable by our models in this project and placed in the data directory.

### Gathering data from the API

To get new questions and answers from Super User since the last pull of API data, run this following command on the command line terminal:
```
python run.py api
```
Where the API data will be processed and be returned into a usable continuous file placed in the data directory.

### Running the cosine models

To run the hybrid filtering model on the data, run this following command on the command line terminal:
```
python run.py models
```
Where hybrid filtering model and its inputs will be created and used to generate new recommendations and then evaluated using specific metrics.

### Comparing the model to baselines

To determine how our model compares to the baselines model, run this following command on the command line terminal:
```
python run.py baselines
```
Where the baseline model will be evaluated and recommendations will be returned using the same data as the hybrid filtering model.

### Running all the model targets

If you want to run all of these together, run this following command on the command line terminal:
```
python run.py all
```
Where the all 4 targets (excluding *test*) will run one after another in the order presented above.

### Testing all the model targets

To test how if the repository and all the models and scripts are working, run this following command on the command line terminal
```
python run.py test
```
Where all of the targets ('data', 'api', 'models' and 'baselines') above will all be run one after another in the order presented above, but on small test data so that we can observe how the models and scripts are working.

## Responsibilities
Yo Jerimijenko-Conley: 

Jasraj Johl: Created the ETL process, worked with the Super User API, created the code repository and made sure it was clean and runnable, and created the website's design through HTML and CSS integrating it partially with Flask.

Jack Lin: 
