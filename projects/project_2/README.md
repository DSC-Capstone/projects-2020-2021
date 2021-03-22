# Political Popularity of Misinformation
- This project looks at the popularity and influence of politicians on Twitter by analyzing the engagement ratios as well as the rolling and cumulative maxes of likes and retweets over time.

### Note
- To get the data necessary to replicate this project, access to the Twitter API is needed. 
- You must have configured twarc using `twarc configure` in the terminal with your API credentials in order to run the data pipeline. Additional information on how to do so can be found here, https://github.com/DocNow/twarc.
- In addition, to run the data pipeline you must obtain a bearer token from Twitter’s API and store it in a config.py file in the root directory. Information on using and generating a bearer token can be found here, https://developer.twitter.com/en/docs/authentication/oauth-2-0/bearer-tokens. bearer_token = “...”

### Obtaining the txt files
- We obtain the tweet IDs that compose our politicians’ timeline found in `src/data` from George Washington University’s TweetSets database found here, https://tweetsets.library.gwu.edu/datasets.
- We chose to focus on politicians who served in the 116th United States Congress, which corresponds to two datasets, Congress: Representatives of the 116th Congress and Congress: Senators of the 116th Congress.
- After identifying our politicians, we gathered the user IDs for their Twitter accounts using Tweepy, which are then used to query the two Congress datasets. The datasets also contain a file of the House and Senate members along with their user IDs which is an alternative way to obtain these IDs. The files can be found here, [Senate](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/MBOJNS/8VQVWT&version=1.0) and [Representative](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/MBOJNS/WXZE5N&version=1.0).
- To query the datasets, for each politician, we selected either the Representative or Senator dataset depending on their position and inputted their user ID in the “Contains any user id” box under the “Posted by” section. This process gives us a txt file of tweet IDs for each politician which is then stored in the `src/data/misinformation` and `src/data/scientific` folders depending on the group the politician is assigned to. 

### Using `run.py`
- To get the data, from the project root directory, run `python run.py data`
    * This downloads the data using the tweet IDs found in `src/data/misinformation` and `src/data/scientific`.
    * This also downloads the engagement metrics needed for ratio analysis using the same tweet IDs.
    * The politicians to analyze can be specified in `config/data-params.json`.
    * The name of the txt file containing the tweet IDs must match the name specified in `config/data-params.json`.
    * The output is a json file for each politician containing their collection of tweets as well as a csv file containing the engagement metrics and text of the tweet ID.

- To calculate the ratios for the tweets, from the project root directory, run `python run.py ratio`
    * This calculates the ratios for the tweets found in the csv files in `src/data/misinformation` and `src/data/scientific`.
    * The politicians to analyze can be specified in `config/data-params.json`.
    * Output is an updated csv file for each politician containing the engagement metrics of their tweets and their ratio values.

- To calculate the popularity estimate metrics, from the projecy root directory, run `python run.py metrics`
    * This will create a JSON file for each estimate metric you wish to analyze. 
    * The outputs are stored in `src/out`.
    * These JSON files will then be used to create visualizations using the visualization target.

- To create the visualizations, from the project root directory, run `python run.py visualization`
    * This creates visualizations from the JSON files created from the metrics target.
    * The outputs are stored in `src/out`.

- To run the permutation test on our groups, from the project root directory, run `python run.py permute`
    * This will run a permutation test within our two groups as well as between our groups.
    * The outputs are stored in `src/out`.

- To run all of the above targets, from the project root directory, run `python run.py all`

- To run the test target, from the project root directory, run `python run.py test`
    * This runs most of the above targets on a small, fake dataset.