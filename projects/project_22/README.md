# DSC180B_Project

URL: https://sdevinl.github.io/DSC180B_Project/

**Disclaimer:**   
  We want to make it clear that the graphsage implentation found in this repo is not our own. We have made minor alterations to the code in order to better serve our overall project in regards to NBA team rankings. The original graphsage implementation can be found here https://github.com/williamleif/GraphSAGE . We would also like to cite their paper:
  
     @inproceedings{hamilton2017inductive,
	     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
	     title = {Inductive Representation Learning on Large Graphs},
	     booktitle = {NIPS},
	     year = {2017}
	   }
  For more information on how to run graphsage as well as the requirements for grapshage be sure to checkout the original graphsage's implementation.

**About:**  
  This repository contains an implementation of a GraphSAGE for node classification on an NBA dataset. The goal being able to classify the ranks of NBA teams using player stats and a graph representation of matchups between teams in a season. 
  
**Setting Up Docker Image**  
  The docker image that was created in order to have an environment able to run this project is found on the repo at aubarrio/graphsage . 
    
**Model**  
  data: The data we use in this project is a compound of multiple webscraped data found on https://www.basketball-reference.com we used stats such as player and team stats, along with team schedules for the season. The seasons for which we collected data range from 2011 to the 2019 season.  

**Basic Parameters**  
  Since we are in early stages of developing we only have one parameter of choice and that is [train, test]. This is to distinguish the data being input into the model.  
    train: Parameter train will train the model on all available features (181 different features)  
    test: Parameter test will train the model on 2 features (Rank and Id) which is meant to use as an evaluation of how our data is performing  
    
**Examples run.py**  
  python run.py test  
  python run.py  
  
  The run.py file will run a graphsage model with the mean aggregator on our NBA data. For more information on how to change the overall model or change the parameters of a model refer to the original graphsage implementation.
  
**Output**  
  Direct terminal output outlining the training, validation and test accuracies of our model.  
  In the sage_outputs folder you will find files that contain data on the model:
  	The rawOutputs folder contains a json file that includes the raw output probabilites of our NBA test set.
	The stats folder will contain all accuracies and losses captured whilst training your model.
	
**Acknowledgements**   
  As mentioned in the disclaimer the original version of this code can be found here https://github.com/williamleif/GraphSAGE. An appreciation to the original creators of this code and a thank you for allowing this code to be available and ready to use on projects such as ours. 
