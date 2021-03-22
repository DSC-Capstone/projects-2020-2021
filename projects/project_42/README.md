# Restaurant Recommender System
There are multiple factors that go into a rating: wait time, service, quality of food, cleanliness, or even atmosphere - for example, a restaurant could have positive sentiment towards the food but negative sentiment towards the service. In order to solve this problem, our aim is to include such sentiments that can be found in the review text and turn that into data which can be used to further improve business recommendations to users.

This repository is a recommender system with a primary focus on the text reviews analysis through TF-IDF (term frequency-inverse document frequency) and targeted sentiment analysis with AutoPhrase to attach sentiments to aspects of a restaurant. In building the recommender system, we learned that review texts can hold the same importance as the numerical statistics because they contain key phrases that characterize how they felt about the review. The ultimate goal is designing a website for deploying our recommender system and showing its functionality.

Visit our `website` branch to try some queries on a preprocessed Las Vegas / Phoenix dataset!

## Important Things:
* This repository is contains two branches. The `main` branch contains the source code for our methods. The `website` branch contains the code to run our recommender sebsite on Flask.
* In our implementation and analysis, we use the Autophrase as our core NLP analysis method by submoduling into our repository.
* The Docker Image and Tag is `launch.sh -i catherinehou99/yelp-recommender-system:latest -c 8 -m 20 -P Always`
- If you would like to learn more details about the AutoPhrase method, please refer to the original github repository: https://github.com/shangjingbo1226/AutoPhrase. Namely, you will find the system requirements, all the tools used and detailed explanation of the output.
- Jingbo Shang, Jialu Liu, Meng Jiang, Xiang Ren, Clare R Voss, Jiawei Han, "**[Automated Phrase Mining from Massive Text Corpora](https://arxiv.org/abs/1702.04457)**", accepted by IEEE Transactions on Knowledge and Data Engineering, Feb. 2018.

## Before You Run:
* In order to use Yelp's academic dataset, you will need to go to their [Website](https://www.yelp.com/dataset) and agree to the Terms of Use Agreement before you download the dataset. Save the dataset to the directory `data/raw`
* This repo uses AutoPhrase as the git submodule, run the command `git submodule update --init` after cloning this repo.

## How to Use this Repository:
1. Run the test target in the `main` branch if you would like to test the targets.
2. In `config/data-params` you can choose which city you would like to subset. For test target, the city can be Las Vegas or Phoenix.
3. Once you successfully run the targets, the generated files will be saved to `data/tmp`. These files need to be used in the `data` folder of the`website` branch.
4. If you would like to run the website on Flask, head over to the website branch!

## Default Run

```
$ python3 run.py -- all
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- all
This will run all the targets below (data, sentiment, eda, tfidf)

For each of the target:
* data: prepares necessary folders, reads in Yelp json files, and filters the dataset to contain rows relevant to the specified city.
* sentiment: performs sentiment analysis on the reviews. It will take in the reviews dataframe and output the positive/negative sentences counts.
* eda: performs the eda analysis of the dataset and autophrase result.
* test: runs the above targets on a test dataset which runs around 3mins.
* clean: removes all the files generated with keeping the html report in the `data/eda` folder.

```
$ python3 run.py -- data
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- data
* Check if the reference folder exists in the user local drive. If not, create all the necessary folder for projects
* Read the dataframes for further analysis.

```
$ python3 run.py -- sentiment
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- sentiment
* Perform sentiment analysis on the reviews text
* Outputs a city_name.csv in the `data/tmp` folder contains, for each restaurant, the positive phrases and the number of times they were mentioned in a review.

```
$ python3 run.py -- eda
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- eda
* Perform the EDA analysis on the AutoPhrase result of individual user
* Perform the EDA analysis on individual city review dataset such as sentiment analysis, feature exploration
* Convert all the EDA analysis into an HTML report stored under data/eda
* After running this tag go to the data/eda directory to see the report.html

```
$ python3 run.py -- tfidf
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- tfidf
* Run restaurant-restaurant based recommendation methods using TF-IDF and cosine similarity score
* Generate the TF-IDF results CSV file of two cities, Las Vegas and Phoenix and store in the reference folder
* Two TF-IDF results CSV are using in the website building as backend database for generating recommendation


```
$ python3 run.py -- clean
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- clean
* Remove all the generated files, plots, dataframes under the reference folder
* Keep the HTML file in the eda/data folder for report visualization
* Be careful: this will clear all the outputs running so far and can not be reversed!!

```
$ python3 run.py -- test
```
The default running file is run.py and can be simply run through the command line: python3 run.py -- test
* Run all the targets on the test data set we generated 

### Responsibilities
* Catherine Hou developed the sentiment/eda/data tag and the food query on the website.
* Vincent Le created the dockerfile and developed the website (not in main branch).
* Shenghan Liu developed TF-IDF/clean/data tag, user AutoPhrase EDA, and the restaurant query on the website.
