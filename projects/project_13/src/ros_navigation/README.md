# Autonomous Navigation Using Gazebo Simulator / RViz

## I. INSTALLATION ROS KINETIC & TURTLEBOT (UBUNTU 16.04)

1) sudo apt-get install ros-kinetic-desktop-full
2) sudo apt-get install ros-kinetic-turtlebot*
3) sudo apt-get install ros-kinetic-catkin python-catkin-tools
4) sudo apt-get install ros-kinetic-web-video-server 

## II. EXTRACT & BUILD 

1) mkdir ucsd_sim_ws
2) catkin config --extend /opt/ros/kinetic 
3) paste src folder into ucsd_sim_ws directory.
4) catkin build

## III. SOURCE DIRECTORY

1) echo "source ~/ucsd_sim_ws/devel/setup.bash" >> ~/.bashrc
2) source ~/.bashrc
** delete existing bashrc:  gedit ~/.bashrc

## IV. ROSLAUNCH CODE (GMapping Using RRT* Algorithm)

### 0. RUN ROSCORE 
1) roscore

### 1. CREATING THE MAP:

<b>1) Launch Gazebo Simulator</b> <br />
roslaunch ucsd_f1tenth_gazebo mybot_test.launch

<b>2) Launch GMapping Node</b> <br />
roslaunch ucsd_f1tenth_navigation gmapping_demo.launch

<b>3) Launch RViz with GMapping Node</b> <br />
roslaunch ucsd_f1tenth_description mybot_rviz_gmapping.launch

<b>4) Launch Teleop Node For Control</b> <br />
roslaunch ucsd_f1tenth_navigation mybot_teleop.launch

### 2. SAVING THE MAP:

<b>1) Save Map to ucsd_f1tenth_navigation Directory</b> <br />
rosrun map_server map_saver -f ~/ucsd_sim_ws/src/ucsd_f1tenth_navigation/maps/test_map

### 3. LOADING THE MAP:

<b>1) Launch Gazebo Simulator</b> <br />
roslaunch ucsd_f1tenth_gazebo mybot_test.launch

<b>2) Launch Completd GMapped Node</b> <br />
roslaunch ucsd_f1tenth_navigation amcl_demo.launch

<b>3) Launch RViz To Start Autonomous Navigation</b> <br />
roslaunch ucsd_f1tenth_description mybot_rviz_amcl.launch

## V. DISPLAY IMAGE THROUGH WEB BROWSER

<b>1) Launch Web Video Server</b> <br />
rosrun web_video_server web_video_server

<b>2) Access Localhost:8080</b> <br />
localhost:8080

<b>3) View Image</b> <br />
http://localhost:8080/stream?topic=/mybot/camera1/image_raw
