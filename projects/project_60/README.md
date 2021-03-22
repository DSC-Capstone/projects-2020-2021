# 
Capstone Project dsc180B Covid-19 Transmission in Buses

In order to install mesa-geo and rtree, we have copied Johnny Lei's README file from our domain repo. Here it is:

### Install MESA
#### Install ABM package MESA with:

> pip install mesa

#### Install MESA-geo
  If you are using a Mac Machine:
    Install rtree FIRST with conda (there seems to exist a distribution issue with pip specificly to Mac):

> conda install rtree

  Install geospatial-enabled MESA extension mesa-geo with:

> pip install mesa-geo

end of credit to J.L


### How to run

Open terminal, change to the directory of the code, then type in the following command 

> python run.py

or if you want to run the code with your own parameters in config/test.json

> python run.py test


## Logistics

We are using Agent-Based-Modeling to simulate covid-19 transmission in Buses.
In order to run this simulation, you could change the parameters in config/test.json to simulate different scenarios.

This simulation start with an empty bus. The bus is scheduled to stop at a certain number of stops at certain times, picking up certain number of students. 
All these parameters are adjustable. As the bus continues, the sick students in the bus transmit the virus by normal activities like talking and breathing, and also coughing, sneezing, etc. This model stops the simulation at the end of a trip where the bus reaches the school (destination). Then this model creates a graph of the number of healthy and sick(if they received the virus) students every minute. There is also a gif of the simulation that shows the position of each student, the time since start, and the layout of the bus.
