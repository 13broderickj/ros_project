cmake_minimum_required(VERSION 2.8.3)
project(example_tf_listener)

find_package(catkin_simple REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(${EIGEN_DEFINITIONS})

catkin_simple()

# example boost usage
# find_package(Boost REQUIRED COMPONENTS system thread)

# C++0x support - not quite the same as final C++11!
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Libraries
# cs_add_libraries(my_lib src/my_lib.cpp)   

# Executables
cs_add_executable(example_tf_listener src/example_tf_listener.cpp src/example_tf_listener_fncs.cpp)
cs_add_executable(kinect_to_world src/kinect_to_world.cpp src/example_tf_listener_fncs.cpp)
# target_link_library(example my_lib)

cs_install()
cs_export()
    
