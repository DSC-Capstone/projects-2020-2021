## Where to begin?
### Begin by uploading your twitter API credentials into a json file as under a new .env director. The file path should look like this: .env/twitter_credentials.json

The json file should be structured as

```json
{
   "CONSUMER_KEY":"your-consumer-key-here",
   "CONSUMER_SECRET":"your-consumer-secret-here",
   "ACCESS_TOKEN":"your-access-token-here",
   "ACCESS_TOKEN_SECRET":"your-access-token-secret-here"
}
```

If you do not have twitter API credentials, please visit https://developer.twitter.com/en/docs/twitter-api to apply for a developer account.

To install the dependencies, run the following command from the root directory of the project: pip install ```-r requirements.txt```

## How to use run.py:
run.py takes in one argument, a choice between *data*, *eda*, *test*

## Directories
* A directory titled *data* will be created with 4 subdirectories: *graphs, raw, processed*
   * *graphs* will hold any charts from eda functions
   * *processed* will hold any statistic data from eda functions
   * *raw* will hold raw tweet data
* Each of the above directories will be split into two additional subdirectories, *news* and *election*
   * *news* will hold any data related to news stations
   * *election* will hold any data related to the election dataset

## Description of arguments (targets)

### data
* Your twitter API credentials for use in downloading data to be used in our project.
* The target will download all tweets as specified in the config file *news_params.json*

### eda
* The eda target will generate statistics and visualizations after data has been gathered from the *data* target
* Currently we have built a wordcloud visualization that will be stored in *graphs* and a statistic of most common hashtags per news station stored in *processed*

### compile and embed
* Performs graph embedding calculations as described in the methodology section of the report

### test
* The test target is designed for grading functionality in the DSC180B capstone course and will test three functionalities:
   * *etl_news* checks that test data is available for use
   * *eda* generates visualizations and statistics based on the test data, stores in *test/testreport*
   * *similarity* will generate similarity hashtag vectors to be used in our main analysis *test/testreport*

