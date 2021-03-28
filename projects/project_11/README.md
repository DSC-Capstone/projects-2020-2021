# Autonomous Vehicles Capstone: Odometry and IMU 


We are Team 4 in the Autonomous Vehicles Data Science Capstone Project. Our project revolves around the IMU, Odometry efforts while we collectively work to build a 1/5 scale racing autonomous vehicle. 

For a vehicle to successfully navigate istelf and even race autonomously, it is essential for the vehicle to be able localize itself within its environment. This is where Odometry and IMU data can greatly support the robot’s navigational ability. Wheel Odometry provides useful measurements to estimate the position of the car through the use of wheel’s circumference and rotations per second. IMU, which stands for Interial Measurement Unit, is 9 axis sensor that can sense linear acceleration, angular velocity, and magnetic fields. Together, these data sources can provide us crucial information in deriving a Position Estimate (how far our robot has traveled) and a Compass Heading (orientation of the robot/where it’s headed).

Our aim is to calibrate, tune, and analyze Odometry and IMU data to provide most accurate Position Estimate, Heading, and data readings to achieve high performant autonomous navigation and racing ability.

*Developed by: Pranav Deshmane and Sally Poon*

### Usage

```
python run.py <target>
```
The Targets are: 
 
* `conversion` - This will extract the data from the raw ROS bags, clean them, and convert them to csv files to be analyzed
 
* `viz_analysis` - This will run the visualizations used in our analysis for IMU and Odometry calibration, tuning, and testing

* `test` - This will test the conversion and visualization process with sample data chosen from our raw data

### Resources
In the resources folder:

* `Openlog_Artemis_IMU_Guide` - Guide we developed for SparkFun Openlog Artemis IMU to improve future experience and aid in the installation, setup, and integration with Jetson NX and ROS.

* `Calibration_OLA_Artemis` - Calibration guide we developed for SparkFun Openlog Artemis IMU to aid in calibration process, analysis for future students/users

* `Setup for Odometry_IMU` - Guide we developed for the Odometry to aid in tuning process, analysis, and setup of Odometry and VESC interaction to improve experience for future students/users.

* `Apollo3`, `ICM-20948`, `Artemis_Hardware` - Sparkfun Openlog Artemis IMU Hardware Specifications, used to cross reference PIN headers that were needed to be configured correctly during integration process. 




### Additional ROS package 
* `ros_imu_yaw_pkg` 
ROS package we developed to aid in the integration of the OLA Artemis IMU to ROS. It allows the orientation quaternion readings derived from the IMU to be easily converted into Euler angles and Yaw heading. This is to improve the debugging process within ROS and helps to easily visualize the Yaw heading. This package can be run in parallel as a complement to the main ROS package used to interface with the OLA Artemis IMU and can easily integrate with the rest of your current ROS system in place as a separate node. Overall, this is to aid in the development process within ROS when deriving Yaw Heading from the OLA Artemis IMU. 

