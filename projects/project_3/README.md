# The Sentiment of U.S. Presidential Elections on Twitter
This project investigates the public sentiment on Twitter regarding the 2016 and 2020 U.S. Presidential Elections. Political tensions in the United States came to a head in 2020 as people disputed President Donald Trump's handling of various major events that the year brought such as the COVID-19 pandemic and the killing of George Floyd and subsequent racial protests, and we aim to identify if this was quantifiably reflected in people's behavior on social media. To do this, we perform sentiment analysis on tweets related to the elections and conduct permutation testing to analyze how sentiment may differ between the two years and between and within politically left- and right-leaning groups of users. 


### Running The Project
- All commands specified here are to be run from within this project's root directory
- To install necessary dependencies, run `pip install -r requirements.txt`
- Note: to get the data necessary to replicate this project, access to the Twitter API is needed

### Using `run.py`
- This project uses publicly available 2016 and 2020 presidential election datasets located at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN and https://github.com/echen102/us-pres-elections-2020. Given the 2016 dataset is not uniformly structured, you must manually download the txt files of tweet ids from the dataset's website to the directory `data/raw/2016`. The 2020 dataset can be downloaded programmatically using the `data` target, as follows.

- To get hydrated tweets for the 2016 and 2020 tweet ids, run the command `python run.py data`
    * This samples from the 2016 tweet ids located in `data/raw/2016` and stores them in txt files in `data/temp/2020/tweet_ids/`
    * It also directly downloads tweets for the 2020 election from the dataset's GitHub page falling within the date range specified in `config/etl-params.json`, samples from them, and stores them in txt files in `data/temp/2020/tweet_ids/`
    * It then rehydrates the tweets using `twarc` and saves the them in jsonl format in the `hydrated_tweets/` directory within each year's respective data directory

- To clean the rehydrated tweets, run the command `python run.py clean`
    * This takes the rehydrated tweets obtained from running `python run.py data` and creates a single csv file of tweets for each year with fields for the features of interest specified in `config/clean-params.json`
    * For the purpose of performing sentiment analysis later, tweets in languages other than English are filtered out.
    * The resulting csvs are stored as `clean/clean_tweets.csv` within each year's data directory.

- To run the main analysis, run `python run.py compute` from the project root directory
    * For each year, this subsets the tweets into left and right leaning, classifies the the different types of dialogue, performs sentiment analysis on the various subsets of data, and conducts permutation testing on the subsets of the data between the two years, producing plots of the results.
    
- To run the `clean` and `compute` targets on fake test data, run the command `python run.py test`

