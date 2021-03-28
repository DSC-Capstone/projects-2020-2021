# Data Science Senior Capstone - Viasat VPN Analysis

**Table of Contents**:
- [Link to Blog page](#blog-page)
- [Abstract](#abstract)
- [Approach](#approach)
- [Running](#running)
  - [Setup](#setup)
  - [Logging](#logging)
  - [Target `data`](#target-data)
  - [Target `features`](#target-features)
  - [Target `train`](#target-train)
- [Report](#report)

## Link to Blog page
https://mhrowlan.github.io/streaming_provider_classifier_inside_vpn/
## Abstract

Whether to access another country's Netflix library or for privacy, more people are using Virtual Private Networks (VPN) to stream videos than ever before. However, many of the different service providers offer different user experiences that can lead to differences in the network transmissions. This repository contains the implementation of our classifying model to determine what streaming service provider was being used over a VPN. The streaming providers that the model identifies are Amazon Prime, Youtube, Netflix, Youtube Live, Twitch, and an other category consiting of Disney+, Discovery+, and Hulu. This is valuable in understanding the differences in the network work patterns for the different streaming service providers. We achieve an average accuracy of 96.5% on our Random Forest model.

## Approach

We utilize Viasat's [`network-stats`](https://github.com/Viasat/network-stats) to collect network traffic on a per-second, per-connection basis while we are connected to a VPN, then engage in either internet browsing or video streaming behavior.

Utilizing the output of network-stats, we extract packet-level measurements and engineer features based on the packet sizes, arrival times, and directions.

We leverage these features in a classification model to determine whether or not a network-stats output contains video streaming activity.

## Running

The following targets can be run by calling `python run.py <target_name>`. These targets perform various aspects of the data collection, cleaning, engineering, training, and predicting pipeline.

### Setup

To leverage the existing dataset, you must be a member of DSMLP and have access to the shared /teams/ directory.

Log on to DSMLP via `ssh <username>@dsmlp-login.ucsd.edu`

Launch a Docker container with the necessary components via `launch-180.sh -i jeq004/streaming_provider_classifier_inside_vpn -G B05_VPN_XRAY -c 8 -g 1 -m 64`

Clone this repository: `git clone https://github.com/mhrowlan/streaming_provider_classifier_inside_vpn.git`

Navigate to this repository `cd streaming_provider_classifier_inside_vpn`

Now, you are ready to configure targets for our project build. Details are specified below.

### Logging

Logging behavior can be configured in `config/logging.json`.
| Key | Description |
| --- | --- |
| produce_logs | Boolean. Whether or not to write to the log file. Default: `true` |
| log_file | Path to the log file. Default: `data/logs/project_run.log` |

### Target `data`

Loads data from a source directory then performs cleaning and preprocessing steps on each file. Saves the preprocessed data to a intermediate directory.

If on DSMLP with the proper group membership, this target will symlink existing data from the shared /teams/ directory.

See `config/data-params.json` for configuration:
| Key | Description |
| --- | --- |
| source_dir | Path to directory containing raw data. Default: `data/raw/` |
| out_dir | Path to store preprocessed data. Default: `data/preprocessed/` |

### Target `features`

Engineers features on the preprocessed data with configurable parameters and saves to an output directory.

See `config/features-params.json` for configuration:
| Key | Description |
| --- | --- |
| source_dir | Path to directory containing preprocessed data. Default: `data/preprocessed/` |
| out_dir | Path to directory to store feature engineered data. Default: `data/features/` |
| chunk_size | Milliseconds. We split our variable length data into smaller chunks with consistent time spans. Default: `90000` |
| rolling_window_1 | Milliseconds. We generate a smoothed mean using rolling windows of multiple lengths, this is the first length. Default `10000` |
| rolling_window_1 | Milliseconds. This is the second length for our smoothed means. Default `60000` |
| resample_rate | Time offset. We resample our packet measurements to produce a consistent sample-rate signal for spectral analysis. Default `500ms` |
| frequency | Hertz. We compute a power spectral density feature on a signal of this sample rate. Default `2.0` |

### Target `train`

Trains a classifier based on the new features and outputs the accuracy between the predicted and true labels. In other words, it prints out the percentage of cases that were correctly classified as streaming.

See `config/train-params.json` for configuration:
| Key | Description |
| --- | --- |
| source | Path to csv containing feature engineered data. Default: `data/features/features.csv` |
| out | Path to .pkl file which will save the trained model. Default: `data/out/model.pkl` |
| validation_size | Proportion. This amount of training data will be withheld to evaluate the performance of the trained classifier. Default: `0.3` |
| classifier | String name of scikit-learn classifier to use. One of 'RandomForest', 'KNN', or 'LogisticRegression'. Default: `RandomForest` |
| model_params | Scikit-Learn hyperparameters for the chosen model. See scikit-learn documentation. |

### Target `all`

Runs `data`, `features`, `train` in order.

### Target `test`

Runs `data`, `features`, `train` with configuration found in `test/config/`.

Can optionally specify targets after test to only run that target. For example `python run.py test data` will only run the data target with the test config.

## Report

An academic report on the exploration and model built in this repository can be found at [`SPICIVPN_report.pdf`](SPICIVPN_report.pdf)
