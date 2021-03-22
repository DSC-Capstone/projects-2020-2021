# Data Science Capstone Project

## Info
* The data being used here is Tweets from various news sources.
## Preqrequisites
* Ensure libraries are installed. (pandas, requests, os, gzip, shutil, json, flatten).
* Download repo: https://github.com/thepanacealab/covid19_twitter.
* Docker container id: tmpankaj/example-docker
## How to Run
* All parameters are of type str unless specified otherwise
* Set twitter API Keys in config/twitter-api-keys.json
#### Test
* run 'python run.py test' in root directory of repo
* look in test/visualizations for the test targets
#### Data 
* Go inside docker container
* Add .txt files with Tweet IDs from https://tweetsets.library.gwu.edu/ to some directory where preprocessed data will be stored. (E.g. cnn.txt in /data/preprocessed directory)
* Use this hydrator https://github.com/DocNow/hydrator to hydrate these tweets and make sure there is a .csv file in the same directory (E.g. cnn.csv in /data/preprocessed)
* Set data parameters in config/data-params.json
   * preprocessed_data_path: path to directory of preprocessed data
   * training_data_path: path to directory to output training data
   * dims (list of str): list of the names of the dimensions that polarities will be eventually calculated on (E.g. ["moderacy", "misinformation"])
   * labels (dict): dictionary with the keys including the news sources and each value being a list with a polarity for each dimension. Every news source that will be used in your data must have a label for every dimension. (E.g. {"cnn": [0, 1], "fox": [1, 0]})
   * user_data_path: path to directory to output user data
   * exclude_replies (bool): If true, will exclude replies when collecting user tweets.
   * include_rts (bool): If true, will include retweets when collecting user tweets.
   * max_recent_tweets (int): maximum recent number of tweets to obtain from a user
   * tweet_ids (list of str): list of tweet IDs to collect to analyze flagged vs unflagged retweeters
* Make sure paths to directories already exist
* run 'python run.py data' in root directory of repo
* This will only collect the data
#### Train
* Go inside docker container and make sure data has been collected
* Set train parameters in config/train-params.json
   * training_data_path: path to directory of the training data (should be same as data-params)
   * model_path: path to directory to output models to be trained
   * dims (list of str): list of the names of the dimensions that polarities should be calculated on (E.g. ["moderacy", "misinformation"])
   * fit_priors (list of bools): hyperparemeter for Naive Bayes classifier (1 for each dimension) - Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
   * max_dfs (list of floats/ints): hyperparameter for CountVectorizer (1 for each dimension) - When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts.
   * min_dfs (list of floats/ints): hyperparameter for CountVectorizer (1 for each dimension) - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. 
   * n_splits (int): number of folds to use for K-fold cross validation
   * outdir: path to directory to output a notebook of the results
* Make sure paths to directories already exist
* run 'python run.py train' in root directory of repo 
* Look in the outdir you specified for an html file of the results
#### Analysis
* Go inside docker container and make sure data has been collected and models have been trained
* Set analysis parameters in config/analysis-params.json
   * model_path: path to directory of trained models (should be same as train-params)
   * user_data_path: path to directory of user data (should be same as data-params)
   * dims (list of str): list of the names of the dimensions that polarities should be calculated on (E.g. ["moderacy", "misinformation"])
   * tweet_ids (list of str): list of tweet IDs to analyze
   * flagged (dict): dictionary should have a key for every tweet to be analyzed and a boolean for whether or not the tweet was flagged (E.g. {"123": true, "456": false})
   * outdir: path to directory to output a notebook of the results
* Make sure paths to directories already exist
* run 'python run.py analysis' in root directory of repo
* Look in the outdir you specified for an html file of the results
#### Results
* Go inside docker container and make sure data has been collected, models have been trained, and analysis has been ran.
* Set results parameters in config/results-params.json
   * user_data_path: path to directory of user data (should be same as data-params)
   * dims (list of str): list of the names of the dimensions that results should be calculated on (E.g. ["moderacy", "misinformation"])
   * outdir: path to directory to output a notebook of the results
* Make sure paths to directories already exist
* run 'python run.py results' in root directory of repo
* Look in the outdir you specified for an html file of the results
