cmake_minimum_required(VERSION 3.0.2)
project(my_vins_loop)

add_compile_options(-std=c++14)

# 1. 查找 ROS 依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
)

# 2. 查找 OpenCV (使用之前验证过的正确路径)
set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# =================================================================
# 3. 【核心修复】查找 RTAB-Map
# =================================================================

# 尝试自动查找
# find_package(RTABMap REQUIRED) 


# set(RTABMap_DIR "/opt/ros/noetic/lib/aarch64-linux-gnu/rtabmap-0.21") 


find_package(RTABMap REQUIRED HINTS 
    /opt/ros/noetic/lib/aarch64-linux-gnu/rtabmap-0.20
    /opt/ros/noetic/lib/aarch64-linux-gnu/rtabmap-0.21
    /opt/ros/noetic/lib/x86_64-linux-gnu/rtabmap-0.20
    /opt/ros/noetic/share/rtabmap/cmake
)

message(STATUS "RTABMap Found: ${RTABMap_FOUND}")
message(STATUS "RTABMap Dir: ${RTABMap_DIR}")

# =================================================================

catkin_package(
  INCLUDE_DIRS
  LIBRARIES my_vins_loop
  CATKIN_DEPENDS roscpp cv_bridge sensor_msgs nav_msgs visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${RTABMap_INCLUDE_DIRS}
)

add_executable(vins_loop src/vins_loop.cpp)

target_link_libraries(vins_loop
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${RTABMap_LIBRARIES}
)