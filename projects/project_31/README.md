

# Using Epidemiology Model To Predict Case Numbers for COVID-19

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Directions](#directions)
* [Processing](#in_processing)
## General info
- Use covid-19 datasets provided by JHU to fit epidemiology model to U.S.. After figuring out the infection parameter, we can then predict 

## Introduction
Fitting Epidemiology Model with Covid-19 JHU U.S. Data
## Technology
Project is created with:
* Image : https://hub.docker.com/repository/docker/caw062/test
## Setup
- Before running, use `pip install -r requirements.txt` to install all the required packages
- on terminal, run `python run.py data` to retrieve the most current data from JHU & Apple Data

## Directions
`python run.py test` to first download test data, and then build epidemiology model on the test data. 
It will return the beta (infection rate), and D (infection duration) for the entire United States.
It will also return a prediction for counties in Southern California on 1/22/2021 based on case counts on 1/21/2021 (previous day)
