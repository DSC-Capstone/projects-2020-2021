# Election-Sentiment
An analysis of views towards the US 2020 Presidential election using Twitter data.

Contained in this repository are a few notebooks that contain our analyses in this investigation on Election Sentiment analysis as well as an investigation into how Twitter impacts election results.

### Building the preoject using `run.py`

Running the test target via the command "python run.py test" will produce images related to the distribution of discussion levels of the two elections. In order to customize the data that this script is run on, replace the data in the test folder with data of your choice.

Provided in the scripts folder, are scripts to donwload tweets from the github repository that we collected tweets on the 2020 election for. The links to the 2020 election and 2016 election tweet ID's are below:

2016:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN

2020:
https://github.com/echen102/us-pres-elections-2020

In order to run the data download scripts, you need to have twarc installed. However, building the project with the provided dockerfile in this repository will download all of the extra packages not native to most Machine Learning and Data Science platforms.


## Running the project
* To get the data from Twitter, create a developer account and get your developer keys
* Configure `twarc`
  - On the terminal, run `twarc configure`
  - Supply keys made earlier


## Groupmate Responsibilities

### Chris

Chris was responsible for the EDA and sentiment analysis and those respective portions of the report. Her work was focused heavily on understanding the sentiment as time progressed and how that related to the individual elections. 

### Prem

Prem was involved heavily in creating the scripts that could be ran by anyone in order to perform ETL on the data. In addition Prem worked on developing the discussion metric that was critical to this investigation.
