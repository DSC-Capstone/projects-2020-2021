# f1tenth_racecar_dl_custom

This repository holds several packages used for our car. A lot of them have important dependencies that usually need to be built from source depending on the 
architecture.

Namely:
- CUDA
- Pytorch & Torchvision
- Detectron2
- OpenCV
- ROS (tested on Melodic)
- RTABMAP (sudo apt-get ros-melodic-rtabmap-ros)

For running testcamera.launch you need to make your own configuration file and adjust the path in testcamera.launch and my_robot.launch to point to your 
new configuration file. File format can be seen in the camera tuning repository in the references

For running the navigation module you need to call 
```
roslaunch detectron2_ros detectron2_ros.launch input:=camera_topic detection_threshold:=confidence_of_model_you_want model:=model_config.yaml anns:=anns.json imgs:=imgs/ weights:=weights/
```
You can see sample values in detectron2_ros.launch
Once this model is running you can then run 
```
roslaunch racecar object_avoiding.launch
```

You may have to adjust the sensors that get launched with this command. Currently this command also launches the F1Tenth ackermann steering, VESC, camera and Lidar.
They can be adjusted in f110_system/racecar/racecar/launch/includes/common/sensors.launch.xml. 

In the event you are running this in ROS Melodic or lower a virtual environment is necessary to run detectron2_ros package in Python3. Just create a virtual environment 
and install the aforementioned dependencies.


For running the mapping you just need to adjust the camera, depth and any other sensory information you have in the mapping.launch file. We had to add our own custom
transforms as well. Often running with the base RTABMAP launch file that comes with installing RTABMAP is a great way to start.

References:
This package relies heavily on: 
- For running Detectron2 in ROS: https://github.com/DavidFernandezChaves/Detectron2_ros 
- Our base F1Tenth framework: https://github.com/f1tenth/f1tenth_system
- The models have been trained using https://github.com/UCSDAutonomousVehicles2021Team1/autonomous_navigation_image_segmentation 
- The camera has been tuned using https://github.com/UCSDAutonomousVehicles2021Team1/autonomous_navigation_light_sensitivity
- The mapping file has been tuned using https://github.com/UCSDAutonomousVehicles2021Team1/rtabmap_mapping_tuning

