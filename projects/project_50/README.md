## Asnapp

Asnapp is a workout video recommender web application. 

Authors: Amanda Shu, Peter Peng, Najeem Kanishka

### Website URL
The website is now live on: https://workout-recommender.herokuapp.com/

### Video Demonstration
For a demonstration of the project, visit: https://www.youtube.com/watch?v=QJFg0HguGuI

### Data
The data is scraped from https://www.fitnessblender.com/. We are using the data for academic purposes only.

### Code Organization

- `run.py`: Run to get data and model results.
- `app.py`: Runs flask web application.
- `workout_db.sql`: Contains sql statements for creation of tables in database.
- `requirements.txt`: Python packages required to run project
- `wsgi.py & Procfile & runtime.txt`: Entrypoint for Heroku, used in website deployment

**Source**
- The `src/data` folder contains `scrape.py`, the web-scraping script that writes three data files into `data/raw` folder. `fbpreprocessing.py` takes these raw data files and outputs cleaned/transformed data files into `data/preprocessed` folder. `youtube.py` grabs youtube related data from the Youtube API. `model_preprocessing` reads in preprocessed data and transforms the data into what is needed for model inputs.
- `src/models` contains `run_models.py` which trains and evaluates the models. Models are implemented in `lightm_fm.py` and `top_popular.py`
- The `src/utils` folder has `clean.py` which implements the standard target `clean`.
- The `src/app` folder holds files for the web application. `forms.py` contains wtforms classes for registration/login pages. `recommendations.py` holds code for filtering user preferences and building recommendation lists. `register.py` contains helper functions to create the sql insertion/update statements for registering users and updating their workout preferences.

**Static**:
- The`images` folder holds a gif ([source](https://www.pinterest.at/pin/512495632597411529/)) used for the loading page. No copyright intended.
- The `js` folder contains several javascript files. `overlay.js` is for the display of the popup videos on the recommendation page. `workout_info.js` is for registration and update preferences pages. `rec.js` is for the loading page and recommendation engine logic on the recommendations page.
- `libraries/slick` has several files for the carousel, `styles` has a css file, and `favicon.io` is the dumbbell icon
- `vendor` holds several javascript files (Bootstrap, JQuery) for styling/theming of the website

**Config**: `data-params.json` has file paths outputs for data collection/preprocessing and `test-params.json` has the data paths for the test target. To webscrape, this folder should also include `chromedriver.json`. To gather Youtube data, `api_key.json` specifies the api key. To run the app, `db_config.json` has the database configurations.

**Notebook**: `eda.ipynb` is a notebook with exploratory data analysis on scraped data. `param_comparision.ipynb` is a notebook reporting the recommendation models' performance across a couple parameters. `top_popular_extension.ipynb` is a notebook looking into adding Youtube API data into the top popular recommender. `KNN_collab.ipynb` contains results of a KNN collaborative filtering model from surprise package and a pure collaborative filtering from LightFM.

**Templates**: Holds html files for the various endpoints.

**Testdata/raw**: These are fake datasets meant to be used with the test target.

**Docker**: Docker related files. See [here](https://github.com/amandashu/Workout_Recommender/blob/main/docker/README_DOCKER.md)

**Materials**: Contains pdfs for presentation slides and a report detailing our methods/implementations.

### Set Up Project Environment
There are two ways to run this project: a) Docker (preferred) or b) Locally <br>
a) To Run in Docker:<br>
  1) Pull the container with `docker pull nkanishka/workout-recommender`
  2) Run the container using:
  * General Use: `docker run -it -p 5000:5000 workout-recommender`
  * DSMLP Only: `launch.sh -i nkanishka/workout_recommender_dsmlp -c 4 -m 8` <br><br>`kubectl port-forward <Kubernates Cluster Name> 5000`<br><br> `ssh -N -L 5000:127.0.0.1:5000 <AD Name>@dsmlp-login.ucsd.edu`
  3) Inside container/cluster, type `cd Workout_Recommender`. Note that in the DSMLP environment, you will need to manually clone this repo.
  4) If using website, go to [localhost:5000](localhost:5000)
    <br>

b) To run locally, install requiremnents.txt into a virtualenv. Make sure you have Python 3.8+ and Pip installed.

### Run the Project Stages
- To get the data, run `python run.py data`. This scrapes the data and cleans the data and saves these files into `/data/raw` and `data/preprocessed` respectively.
  - Note: for scraping, this assumes that there is a file `config/chromedriver.json` that specifies where the path to the downloaded chromedriver.exe file for your Chrome version lies in the attribute `chromedriver_path`.
  - Note: for making requests to Youtube API, this assumes that there is a file `config/api_key.json` that specifies the api key in the `api_key` attribute.
- To run model results, run `python run.py model`. This takes in the preprocessed data, trains the models, and prints out the NDCG scores for each model.
- Standard target `clean` is implemented, and it will delete the `data` folder.
- Standard target `all` is implemented, and it equivalent to running `python run.py data model`.
- Standard target `test` is implemented, and runs the data preprocessing and modeling results on the test data. The purpose of this target is purely to check the implementation of the code.
- Use `python app.py` to run the app locally.
  - Note: this assumes that there is a file `config/db_config.json`, which has database host, user, password, and name information.
  - And a file `config/flask_keys.json` which has a Flask secret key
