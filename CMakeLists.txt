cmake_minimum_required(VERSION 3.16)
project(gitslam VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# CUDA flags for Jetson
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_72")  # Xavier
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")  # Orin (uncomment for Orin)

#############################################
# Dependencies
#############################################

# Eigen3 (linear algebra)
find_package(Eigen3 3.3 REQUIRED)

# OpenCV with CUDA support (vision)
find_package(OpenCV 4.5 REQUIRED
    COMPONENTS core imgproc highgui features2d calib3d
               cudaimgproc cudawarping cudaoptflow)

# ZED SDK (camera)
find_package(ZED 4.0 REQUIRED)

# TensorRT (deep learning inference)
find_package(TensorRT 8.6 REQUIRED)

# Sophus (Lie algebra)
find_package(Sophus REQUIRED)

# g2o (graph optimization)
find_package(g2o REQUIRED)

# Pangolin (visualization)
find_package(Pangolin REQUIRED)

# yaml-cpp (configuration)
find_package(yaml-cpp REQUIRED)

# glog (logging)
find_package(glog REQUIRED)

# gflags (command-line flags)
find_package(gflags REQUIRED)

# Threads
find_package(Threads REQUIRED)

# CUDA
find_package(CUDA 11.4 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#############################################
# Include directories
#############################################

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

#############################################
# Source files
#############################################

# Sensors
set(SENSOR_SOURCES
    src/sensors/zed_driver.cpp
    src/sensors/mavlink_imu_driver.cpp
    src/sensors/time_sync.cpp
)

# Frontend
set(FRONTEND_SOURCES
    src/frontend/image_preprocessing.cpp
    src/frontend/stereo_rectification.cpp
    src/frontend/optical_flow_tracker.cpp
    src/frontend/superpoint_tracker.cpp
    src/frontend/hybrid_tracker.cpp
    src/frontend/stereo_depth.cpp
    src/frontend/frontend_thread.cpp
)

# IMU
set(IMU_SOURCES
    src/imu/preintegration.cpp
    src/imu/bias_estimator.cpp
)

# Backend
set(BACKEND_SOURCES
    src/backend/keyframe.cpp
    src/backend/map.cpp
    src/backend/visual_inertial_ba.cpp
    src/backend/marginalization.cpp
    src/backend/backend_thread.cpp
)

# Optimization
set(OPTIMIZATION_SOURCES
    src/optimization/cost_functions.cpp
    src/optimization/linearization.cpp
    src/optimization/solver.cpp
)

# Visualization
set(VISUALIZATION_SOURCES
    src/visualization/pangolin_viewer.cpp
    src/visualization/visualizer_thread.cpp
)

# Utils
set(UTILS_SOURCES
    src/utils/math_utils.cpp
    src/utils/calibration.cpp
    src/utils/logging.cpp
)

#############################################
# Library
#############################################

add_library(gitslam_lib SHARED
    ${SENSOR_SOURCES}
    ${FRONTEND_SOURCES}
    ${IMU_SOURCES}
    ${BACKEND_SOURCES}
    ${OPTIMIZATION_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(gitslam_lib
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    Eigen3::Eigen
    Sophus::Sophus
    g2o::core g2o::types_sba g2o::solver_eigen
    ${Pangolin_LIBRARIES}
    yaml-cpp
    glog::glog
    Threads::Threads
)

#############################################
# Executable
#############################################

add_executable(gitslam_node
    src/main.cpp
)

target_link_libraries(gitslam_node
    gitslam_lib
    gflags
)

#############################################
# Install
#############################################

install(TARGETS gitslam_node gitslam_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY config/
    DESTINATION share/gitslam/config
)

#############################################
# Tests (optional)
#############################################

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
