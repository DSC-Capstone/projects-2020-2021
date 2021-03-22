# Opioid-Use Prevalance Analysis Project 
```
### 
* Author: Flory Huang
* Date: 03.19.2021
```

This repository contains code for extraction mentioned drug terms and emotions of drug use discussion in Reddit data. 
The code takes in reddit post that are discussing drugs and ontology list(RxNorm and Mesh),used to extract drug names. 
The emotion in reddit post will be analyzed.

The data_reddit.py takes care of loading data and formating data
The analysis.py conduct the similar matching to extract drug terms.
The emotion.py will analyze emotion in the reddit post
The model.py procude a report of matching result

The code can be excuted by 
    - python run.py test 
    - python run.py data 
    - python run.py analysis

Target explaination 
    0: test: test the whole process of code
    1: data： read in, text, and ontology data. And process them into format (e.g.parse nouns from text and put into a dictionary with ontology terms) that can be used for following analysis steps.
    2: analysis: use similar matching to extract drug terms and perform emotion analysis and store result into output csv files. 

