# Interactive User Interface Using Rosbridge

## I. INSTALLATION ROSBRIDGE (UBUNTU 16.04)

<b>1) sudo apt-get install ros-kinetic-rosbridge-server</b>

## II. LAUNCH SERVER AND HTML

<b>1) Launch Rosbridge Server</b> <br />
roslaunch rosbridge_server rosbridge_websocket.launch

<b>2) Launch Web Browser</b> <br />
drag web.html to internet browser

<b>3) Connect to Rosbridge Server</b> <br />
click "Connect!" button to connect to Rosbridge server.

## III. INTERACT WITH VEHICLE

<b>1) Run Python Visualizer</b> <br />
python generate_rrt_vis.py

<b>2) Navigate Autonomously</b> <br />
click "Preset Value" button, then click "Submit" button
