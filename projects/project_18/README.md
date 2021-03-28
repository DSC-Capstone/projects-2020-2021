# NBA-Game-Prediction
Project Group: MengYuan Shi, Austin Le

This repository contains a data science project that discover the NBA Game Prediction. We investigate the social network for individual NBA players and the relationship between each team. We will use team's statistics and players' statistics and analysis for predicting who wins the games by leveraging the team's statistics and players' statistics from 2015 season to 2019 season. We will use GraphSAGE which is a generalizable embedding framework to create a graph classification.


### Warning
Our group has altered and used the graphsage implementation that can be received from https://github.com/williamleif/GraphSAGE . We made minor changes within the model and inputs to align with the goals of our project, but we would like to cite them as a source for the main graphsage implementation.


### Running the project
- `python run.py` can be run from the command line to ingest data, train a model, and present relevant statistics for model performance to the shell
  - Reads in CSV file from eightthirtyfour for play by play data
  - Runs algorithm to create network between each player based on their playing time
  - Appends all player edges from all season onto a single graph 
  - Embedd player categorical statistics onto each node 
  - runs graphSage model to learn over features

### Outputs
  - The outputs printed will be the corresponding accuracies obtained after the training
  - ~5min runtime

### Responsibility 
- Austin Le: Responsible for the data cleaning and data scraping and the coding part as well as the report.
- MengYuan Shi: Responsible for the paper researching and writing the report part as well as the visualization.
