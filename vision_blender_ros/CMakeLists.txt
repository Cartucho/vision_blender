cmake_minimum_required(VERSION 2.8.3)
project(vision_blender_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  cv_bridge
  tf2
)

catkin_python_setup()

catkin_package(
  DEPENDS rospy std_msgs cv_bridge tf2
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   scripts/vision_blender_listener
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/vision_blender_ros.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
