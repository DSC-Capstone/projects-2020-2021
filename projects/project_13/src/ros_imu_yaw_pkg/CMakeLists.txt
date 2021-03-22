cmake_minimum_required(VERSION 3.0.2)
project(ucsd_navigation)

find_package(catkin REQUIRED)

catkin_package()

include_directories( ${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/imu_to_yaw.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
