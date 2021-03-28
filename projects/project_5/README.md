
# The Spread of Misinformation on Reddit
Observing how different forms of misinformation and conspiracies are spread through social media.

## Overview
With the amount of actively spread misinformation circulating popular social media platforms, our goal is to explore if various forms of misinformation follow varying patterns of diffusion. The scope of our project will be limited to two misinformation types -- myth and political misinformation, and focused on Reddit. Our data will be obtained from Reddit archive [pushshift.io](http://pushshift.io).

## Contents
- `src` contains the source code of our project, including algorithms for data extraction, analysis, and modelling.
- `notebooks` contain some examples of the models this code will generate, detailing our findings under the circumstances in which we conducted our testing.
- `config` contains easily changable parameters to test the data under various circumstances or change directories as needed.
- `run.py` will build and run different the different parts of the source code, as needed by the user.
- `references` cite the sources we used to construct this project.
- `requirements.txt` lists the Python package dependencies of which the code relies on. 

## How to Run
- Install the dependencies by running `pip install -r requirements.txt` from the root directory of the project.
- Alternatively, you may reference our Docker image to recreate our environment, located [here](https://hub.docker.com/r/cindyhuynh/reddit-misinformation).
- Due to the open source nature of the PushShift archive, there is no need for any API use or developer account. 

### Building the project stages using `run.py`
- To download the data, run `python run.py data`
	- This downloads reddit comments from specified subreddits between a certain time period. The subreddits and time period are specified in `config/data_params.json`
- To create visualizations of EDA charts, run `python run.py eda`
	- This creates bar charts representing the dataset we have collected. It also shows visualizes statitics of one-time posters and average number of posts in each category and subreddit.
- To get user polarities, run `python run.py user_polarity`
	- This generates a metric for all users collected in the data, getting filepaths from `config/user_polarity_params.json`
- To generate common user matrices, run `python run.py matrices`
	- This creates two matrices demonstrating, for every possible pair of subreddits, the number and average user polarity of the users in that subset. Filepaths are specified in `config/matrix_params.json`. 
	- NOTE: user_polarities should be run at least once before running `matrices`.
- To create visualizations of user polarities and matrices, run `python run.py visualize`
	- This creates bar charts representing the general user polarity spread, as well as charts showing how users of different types cross into other subreddits. These bar charts will be replaced by heatmaps in a future update for easier visualization. Filepaths are specified in `config/visualize_params.json`. 
	- NOTE: `user_polarities` and `matrices` should be run at least once before running `visualize`.
- To run the full pipeline, run `python run.py all`
	- This will run all the targets. These targets include: `data`, `eda`, `user_polarity`, `matrices`, and `visualize`
- To run the full pipeline on test data, run `python run.py test`
	- This will run all the targets on test data. These targets include: `data`, `eda`, `user_polarity`, `matrices`, and `visualize`
