# DSC 180 Autonomous Systems

Neghena Faizyar, Garrett Gibo, Shiyin Liang

## Data
To get sample data: 
[Link to Data](https://drive.google.com/drive/folders/1wh7EtgtrS8Wi8xBIe1VwzFDBnp751XHv?usp=sharing)

Download this data to put into the data/raw folder.

## Usage 

### ROS

A large portion of this project is a series of ROS packages that can be launched
directly or integrated in other ROS packages. This repo has been made in such a
way that it serves as a full ROS workspace, thus to run the packages that are
contained here, simply run:

``` sh
# build project
catkin_make

# source ROS packages
source devel/setup.bash

# launch main node
roslaunch simulation main.launch
```

This will launch a gazebo simulation containing a test vehicle with sensors that
were used for this project. Because the simulation requires both ROS and gazebo
which have large graphical portions, these packages must be run a system that
has ROS setup already and also has some type of grapical interface, for example
X on linux based systems.

### Analysis

The second portion of this project is analysis that is done on the data that
is gathered from both simulation and real sensors. 

To run any of the following targets, the command is:

```sh
python run.py <target>
```

Information on the targets is found below.

#### Targets

* `cep`: Calculates the Circular Error Probable (CEP), and 
2D Root Mean Square (2DRMS), and then plots and creates a graph of the CEP 
and 2DRMS circles with the datapoints. 

* `clean_data`: Extract, transform, and clean the raw GPS data so
that it can be used for anaylsis.

* `get_path`: Takes in CSV of GPS coordinates and cleans/filters points to create
a usable path.

* `ground_truth`: Plot ground truth coordinates against estimated coordinates 
for reported GPS values.

* `robot`: Creates an instance of the
[dronekit-sitl](https://dronekit-python.readthedocs.io/en/latest/develop/sitl_setup.html),
which can be used to generate realistic sensor data that can be used
as a template for the following targets.

* `robot_client` Provides the interface to connect to a specified robot.
The client connects over tcp or udp and uses the
[MAVLink](https://mavlink.io/en/messages/common.html), standard for
the messages.

* `test`: Runs our projects test code by extracting, transforming, and then 
cleaning the raw GPS test data such that it could be used. 

* `visualize`: Create visualizations for all of our data using bokeh. It will 
plot the line the GPS reports it has traveled and uploads it into the vis 
folder of our repository. 
