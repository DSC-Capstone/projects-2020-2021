

# OnSight: Outdoor Rock Climbing Recommendations

Recommendations for outdoor rock climbing has historically been limited to word of mouth, guide books, and most popular climbs. With our project OnSight, we believe we can offer personalized recommendations for outdoor rock climbers.

Disclaimer: With rock climbing, especially outdoors, there is an inherent risk that is taken when you decide to climb. Although our recommender tries to offer routes similar to the ones users have done, there is still a risk that the route may be too hard and therefore dangerous. This is not a problem that is solely put on the recommender, but a problem with rock climbing as a whole. There is no standard in climbing grades, but rather it is an agreement among the climbers that have climbed that route. Therefore climbing grades are subjective, and climbs may be harder and more dangerous than a user expects. We realize this, and we encourage everyone to look at the safety information of each climb on its corresponding climbing page on Mountain Project.

## How To Run

We suggest for casual users to simply use our website to run the project. The website URL is https://dsc180b-rc-rec.herokuapp.com/. Note that this project runs on a free dyno, so if you are the first user to open the website in about half an hour, then the website may take a minute to load. Be patient!

However, if you are interested in making changes or diving deep into the code, you can run the project and customize it by either creating your own Heroku project or running the project on the command line.

### Creating your own Heroku project
To have your own version of OnSight running on Heroku, do the following steps:
 1. Fork the OnSight GitHub repository to your own GitHub account
 2. Create a new project on Heroku
 3. In the Heroku app dashboard, go to the "Deploy" tab
 4. Under the "Deployment method" section, select GitHub and connect the Heroku app to your forked repository
 5. In the Heroku app dashboard, go to the "Settings" tab
 6. Under the "Config Vars" section, click on "Reveal Config Vars" and fill in the config variables as shown in the table below. 

|Config Vars|Value|Description|
|-|-|-|
|PROJECT_PATH|mysite/|Since the django webserver is stored in the "mysite/" folder, but the Procfile (tells Heroku how to start the web server) is in the project root, we need to tell Heroku to look in the "mysite/" folder for the webserver code|
|GOOGLE_MAPS_API_KEY|Your API key|Your Google Maps API key needs to have the following APIs enabled on the key: "Maps JavaScript API", "Places API", and "Maps Embed API". This key is not strictly necessary, but without the key the map will not work and location can only be entered by manually typing in a latitude and longitude, which is not very UX friendly.|

7. Make sure you deploy the Heroku app again from the "Deploy" tab on the Heroku dashboard and you should be all set!

### Running the Project on the Command Line

To run the project, every command must start with "python run.py" from the root directory of the project. By default, "python run.py" will do absolutely nothing. You must use at least one of the following flags to actually get some response:

|Flag|Type|Default Value|Description|
|-|-|-|-|
|-d, --data|bool|False|Use this flag to run all data scraping code. This will take a very long time, upwards of one week total to scrape all the data. It is recommended *not* to run this. Be aware that this will only store data locally. |
|-c, --clean|bool|False|Use this flag to run all data cleaning code. It is expected that all files defined in the "state" key of data_params.json are present in the raw data folder.|
|-\-data-config|str|"config/data_params.json"|The location at which data parameters can be found|
|-\-web-config|str|"config/web_params.json"|The location at which web parameters can be found. These parameters simulate a user using the website.|
|-\-top-pop|bool|False|Use this flag to print the top N most popular climbs. This does not use locally saved data, but rather uses saved data in MongoDB. Additionally, the exact climbs and number of climbs are determined by the web_params.json file.|
|-\-cosine|bool|False|Use this flag to print the top N most similar climbs to the users favorite. This does not use locally saved data, but rather uses saved data in MongoDB. Additionally, the exact climbs and number of climbs are determined by the web_params.json file.|
|-\-test|bool|False|Use this flag to run the two implemented models based on default config files. Using the --test flag will override all other present flags and is equivalent to running "python run.py --top-pop --cosine --debug".|
|-\-delete|bool|False|Use this flag to wipe out all data from MongoDB. This will not do anything since the MongoDB login is set to read only.|
|-\-upload|bool|False|Use this flag to upload cleaned data to MongoDB. This will not do anything since the MongoDB login is set to read only.|
|-\-debug|bool|False|Use this flag activate various print statements throughout the project.|

### Description of Parameters

#### Data Parameters

|Parameter Name|Type|Default Value|Description|
|-|-|-|-|
|raw_data_folder|str|"data/raw/"|The location at which raw data will be saved. Note that this path is relative to the project root.|
|clean_data_folder|str|"data/cleaned/"|The location at which clean data will be saved. Note that this path is relative to the project root.|
|states|dict|Too long to copy here...|Although the parameter is called states, this is really just the areas to scrape/clean and the urls at which they can be found. The file will be named based on the key string, and the area url to scrape is the value string. By default this contains all 50 states, with the state name as key and state area url as value.|

#### Web Parameters
Be aware that all these parameters do is simulate a user using the website. Each of the parameters here refer to a form element on the website.

|Parameter Name|Type|Default Value|Description|
|-|-|-|-|
|user_url|str|Too long to copy here...|The "Mountain Project URL" form element on the website. This value is only used if the user requests personalized recommendations. The default value is a user with a lot of climbs rated, about 600 in March 2021. You can find the actual default value in the config file.|
|location|[float, float]|[43.444918, -71.707888]|The "Latitude" and "Longitude" form elements on the website. This location is the center of the circle where climbs will be looked for. The default value is some random location in New Hampshire.|
|max_distance|int|50|The "Max Distance (mi)" form element on the website. This value is the radius of the circle where climbs will be looked for. The default value is 50 miles, which should be sufficient to encompass any climbing area.|
|recommender|str|"top_pop"|The "Recommenders" form element on the website. This is the recommender to use and will be any of "top_pop" or "cosine_rec". There is an additional hidden debug recommender that uses the string of "debug". The debug recommender is not accessible without modifying the "mysite/bootstrap4/forms.py" file|
|num_recs|int|10|The "Number of Recommendations" form element on the website. This is the maximum number of recommendations that will be displayed once the user hits submit.|
|difficulty_range|{"boulder": [int, int], "route: [int, int]}|{"boulder": [0, 3], "route": [11, 16]}|The "Boulder", "V_-V_", "Route", and "5.\_-5.\_" form elements on the website. Due to the way data is cleaned, bouldering V grades and route 5. grades are converted to integers starting at 0 on different scales. You can find the scales defined in code. The two default difficulty ranges correspond to V0-V3 and 5.8-5.10d. Note that if the user does not want boulders or routes, the corresponding difficulty range will be [-1, -1]|

