cmake_minimum_required(VERSION 3.0.2)
project(my_vins_loop)

add_compile_options(-std=c++14)

# ==========================================
# 1. 设置 TensorRT 路径 (根据你的实际路径)
# ==========================================
set(TRT_DIR "/home/ark/TensorRT/TensorRT-8.6.1.6")

# ==========================================
# 2. 查找依赖
# ==========================================
find_package(catkin REQUIRED COMPONENTS
  roscpp cv_bridge sensor_msgs nav_msgs std_msgs visualization_msgs image_transport
)
find_package(OpenCV REQUIRED)
find_package(RTABMap REQUIRED)

# 查找 CUDA (使用 CMake 内置模块)
find_package(CUDA REQUIRED)

catkin_package()

# ==========================================
# 3. 包含头文件
# ==========================================
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${RTABMap_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TRT_DIR}/include
)

# ==========================================
# 4. 链接库目录 (显式指定)
# ==========================================
link_directories(
  ${TRT_DIR}/lib
  /usr/local/cuda/lib64
)

# ==========================================
# 5. 编译设置
# ==========================================
add_executable(my_vins_loop 
  src/my_vins_loop.cpp 
  src/SuperPoint.cpp
)

# 【重点修正】显式列出 TensorRT 的核心库绝对路径，避免找不到
# 注意：libnvinfer.so 和 libnvparsers.so (没有 ch 结尾)
target_link_libraries(my_vins_loop
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${RTABMap_LIBRARIES}
  ${CUDA_LIBRARIES}      # 这会自动链接 cudart
  ${TRT_DIR}/lib/libnvinfer.so
  ${TRT_DIR}/lib/libnvparsers.so
)

add_executable(vins_loop src/vins_loop.cpp)


target_link_libraries(vins_loop
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${RTABMap_LIBRARIES}
)
