cmake_minimum_required(VERSION 3.0.2)
project(navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/create_global_path.py
  scripts/joy_teleop.py
  scripts/waypoint_follower.py
  scripts/waypoint_logger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
