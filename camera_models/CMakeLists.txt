cmake_minimum_required(VERSION 2.8.3)
project(camera_models)

set(CMAKE_BUILD_TYPE "Release")
# 将标准改为 C++14 以匹配 ROS Noetic
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

# 1. 查找 ROS 依赖
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
)

# 2. 查找 Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

# 3. 查找 OpenCV
# 去掉版本号限制，让它找系统默认的，通常更稳
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4   # <--- 【核心修复】加上这一行！
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
# 4. 查找 Ceres
find_package(Ceres REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_models
    CATKIN_DEPENDS roscpp std_msgs
)

# 5. 包含头文件 (核心修复部分)
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv4      # 【关键修复】强制包含这个路径
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# 6. 编译库文件
add_library(camera_models
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

# 7. 编译可执行文件
add_executable(Calibrations 
    src/intrinsic_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

# 8. 链接库
target_link_libraries(camera_models 
    ${Boost_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES} 
    ${catkin_LIBRARIES}
)

target_link_libraries(Calibrations 
    ${Boost_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES} 
    ${catkin_LIBRARIES}
)