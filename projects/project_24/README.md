Name: Jason Chau, Sung-Lin Chang, Dylan Loe

Welcome to our Stock Predictor. 

IMPORTANT!
For running actual models:

In order to run our stock predictor, just make sure that you are in the current directory and run the command 

python run.py all - this will let you scrape the data, as well as running our model


python run.py test - this will let you run our model on the data pulled from the webscraper, as well as predicting which stocks will
be bullish or bearish in Dow Jones 30.

python run.py fcn - this will let you run our Fully connected network model on the data pulled from the webscraper

python run.py build - This command builds the test portion of the code. Please keep in mind this will pull in data from the yahoo finance api,
so it will require an internet connection to pull in the data and calculate it.


------ CONFIG FILE--------------

"NUM_EPOCHS" : 100, ( this is the number of trials you want to use)
"LEARNING_RATE" : 0.001, ( this is the learning rate)
"NUM_HIDDEN" : 32, ( number of hidden features)
"num_days": 5, (number of lag days you will have, we chose 5 because there are 5 trading days)
"nfeat" : 20, ( this is 4 * num_days)
"nclass" : 1, 
"dataset" : "./data/dowJonescorrelation0.4graph.csv", (this is the correlation graph we use for our model)
"thresh" : 0.4, ( the threshold for the node adjacency)
"filepath" : "./data/12modowJonesData/", (the dataset to use)
"timeframe" : "12mo" ( which time period, there is 12mo and 6 mo

Required packages

yfinance == 0.1.55
pandas-datareader = 0.9.0
beautifulsoup4 4.9.3
tensorflow == 1.12.0
networkx == 2.1
numpy == 1.14.3
scipy == 1.1.0
sklearn == 0.19.1
matplotlib == 2.2.2
