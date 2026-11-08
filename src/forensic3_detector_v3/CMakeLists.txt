cmake_minimum_required(VERSION 3.0.2)
project(forensic3_detector_v3)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs)
catkin_package()

install(PROGRAMS
  scripts/forensic3_detect_v3
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)