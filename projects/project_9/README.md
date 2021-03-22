# ResRecovery

Website: https://stephdoan.github.io/ResRecovery/

# Table of Contents

1. [Abstract](#Abstract)
2. [Config Files](#config)
   - [`train-params.json`](#train)
   - [`model-params.json`](#model)
   - [`user-data.json`](#user)
   - [`generate-data.json`](#generate)
3. [Running the Project](#running)

## Abstract

Virtual private networks, or VPNs, have seen a growth in popularity as more of the general population has come to realize the importance of maintaining data privacy and security while browsing the Internet. In previous works, our domain developed robust classifiers that could identify when a user was streaming video. As an extension, our group has developed a Random Forest model that determines the resolution at the time of video streaming. Our final model has an overall accuracy of **87%**.

<a name="config"></a>

## Configuration Files

<a name="train"></a>

### `train-params.json`

Allows users to adjust some parameters of the training data creation process. The main point of focus is the `{interval}` argument. This allows users to adjust how big of a chunk size they would like their model to be trained on. The default is 300 seconds as it allows replication of our project.

| Parameter     | Description                                                                                                     |
| ------------- | --------------------------------------------------------------------------------------------------------------- |
| folder_path   | path to where all of the raw data is stored; please refer to the folder structure below to achieve best results |
| interval      | chunk size                                                                                                      |
| threshold     | minimum megabit value; used in peak feature creation                                                            |
| prominence_fs | sampling rate to find the max peak prominence                                                                   |
| binned_fs     | deprecated parameter                                                                                            |

##### Data Folder Structure

All of the training data should be stored in an accessible `data` folder. CSV files should be categorized into folders according to their resolution below.

```
+-- data
 |
 +-- 144p
 +-- 240p
 +-- 360p
 +-- 480p
 +-- 720p
 +-- 1080p
```

<a name="model"></a>

### `model-params.json`

Allows users to adjust hyperparameters of the random forest classifier. The default values are the values we utilized in our original project.

| Parameter         | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| training_data     | path to where training data is stored; data must be stored as a CSV file |
| n_estimators      | number of trees in the forest model                                      |
| max_depth         | max depth of the tree                                                    |
| min_samples_split | minimum number of samples required to split an internal node             |

<a name="user"></a>

### `user-data.json`

Allows users to input their own data to be classified by the model.

| Parameter     | Description                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| path          | path to where all of the raw user data is stored; must be an output of [network-stats](https://github.com/viasat/network-stats) |
| interval      | chunk size                                                                                                                      |
| threshold     | minimum megabit value; used in peak feature creation                                                                            |
| prominence_fs | sampling rate to find the max peak prominence                                                                                   |
| binned_fs     | deprecated parameter                                                                                                            |

<a name="generate"></a>

### `generate-data.json`

Parameters used by the `generate_data.py` script. Please refer to [Selenium](https://www.selenium.dev/documentation/en/) documentation to install the appropriate `webdriver.exe`. For best use, please configure the `PATH` variable in the `generate-data.py` file to the correct file path of the webdriver. This script was developed using Google Chrome.

| Parameter          | Description                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| network_stats_path | location of network-stats.py                                                                                    |
| interface          | user interface to collect from; refer to [network-stats](https://github.com/viasat/network-stats) documentation |
| playlist           | link to YouTube playlist                                                                                        |
| outdir             | to be implemented                                                                                               |
| resolutions        | list of resolutions to be collected                                                                             |

<a name="running"></a>

## Running the Project

The project is current set to the assumption that users will collect their own training data. There is a repository of available training hosted on the DSMLP server located at `/teams/DSC180A_FA20_A00/b05vpnxray/personal_stdoan/data`. If not accessible, please refer to the [`generate_data.json`](#generate) configurations to automate collection of a training set.

#### Running on the DSMLP Server

The project was mean to be run on the UCSD DSMLP server. Below are instructions if user has access to DSMLP resources.

1. Open up a terminal and run the command below to log onto the server. Users will need to provide appropriate identification when asked.

> `ssh [username]@dsmlp-login.ucsd.edu`

2. Launch a docker container to ensure package dependencies are fulfilled by running the command:

> `launch-180-gid.sh -G 100011652 -P Always stdoan/viasat-q1`

3. Clone this repository.

4. Adjust config files as necessary and then run the targets!

#### Targets

- `python run.py test` will test the various targets to ensure that all methods are running properly.

- `python run.py clean` will delete files created from running various targets. The folder and files are deleted from the local machine.

- `python run.py features` will create features from data specified in `train-params.json`.

- `python run.py predict` will either create training data to create a model or utilize a Pickle'd model that we have included. Output is an array of resolution label for each chunk in the data.
