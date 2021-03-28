# Political Analysis on Senatorial Twitter Account Using Machine Learning
 
### Project Description
The modern American political landscape often seems void of bipartisanship. Nowhere is this stark divide between red and blue more evident than in the halls of the US Capitol, where the Senate and House of Representatives convene to carry out the duties of the legislative branch. While us average Americans rarely watch the daily proceedings of the Senate or House, Twitter has given us a unique window into the debates and discourse that shape our democracy. In fact, the 116th Congress, which served from January 3, 2019 to January 3, 2021 broke records by tweeting a total of 2.3 million tweets! As such, it is clear that Twitter is quickly becoming a digital public forum for American politicians. This surplus of tweets from the 116th Congress enables us to analyze the Twitter (following-follower) relationships between politicians on and across the two sides of the aisle. This project’s main inquiry is into whether there is a tangible difference in the way that Democrat members of Congress speak and interact on social media in comparison to Republican members of Congress. If there are such differences, this project will leverage them to train a suitable ML model on this data for node classification. That is to say, this project aims to determine a Senator’s political affiliation based off of a) their Twitter relationships to other Senators b) their speech patterns, and c) other mine-able features on Twitter. In order to truly utilize the complex implicit relationships hidden in the Twitter graph, we can use models such as Graph Convolutional Networks, which apply the concept of “convolutions” from CNNs to a graph network-oriented framework, and GraphSage model.

### run.py
We implement the GCN and GraphSage models as our main models for training and comparison.

- parameters:
  - model: The choice of models. We only implement the GCN and GraphSage. 
  - dataset: The choice of datasets. There are multiple datasets, including data_voting, data_voting_senti. The differences between these datasets are features. The adjacency matrix of each dataset stays the same.
  - output_path: The output of project will be stored in json file.
  - agg_func: The choice of aggregated function in the graphSage model. We only support MEAN aggregator. The default is MEAN.
  - num_neigh: The number of neighbors in the graphSage model. The default is 10.
  - n: The number of hidden layers in the GCN model. This can be tuned to reach higher accuracy.
  - self_weight: The weight of self-loop in the GCN model.
  - hidden_neurons: The number of hidden neurons in the GCN model. The default is 200 and it can be tuned to reach higher accuracy.
  - device: The device for training the model. We only support cuda.
  - epochs: The number of epochs for both models. The default is 200 epochs.
  - lr: The learning rate for both models. The default is 1e-4. This can be tuned for higher accuracy.
  - val_size: The size of testing data. The default is 0.3.
  - test: The parameter for running test data on models.

- some examples for using the project:
  - python run.py
  - python run.py --test
  - python run.py --n_GCN
  - python run.py --dataset data_voting
  - python run.py --model n_GCN --n 2 --self_weight 20

  

# Our Project Website

Please view our website [here](https://anuragpamuru.github.io/dsc-180b-capstone-b03/)


### Contributers: 
Yimei Zhao, Anurag Pamuru, Yueting Wu
