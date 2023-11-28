cmake_minimum_required(VERSION 3.0.2)

find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

#chh add
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Build our plugin
# add_library(gazebo_wind_plugin SHARED wind_plugin.cpp)
# target_include_directories( gazebo_wind_plugin PUBLIC 
#         "${PROJECT_SOURCE_DIR}"
#         "/home/firefly/eigen")
# target_link_libraries(gazebo_wind_plugin ${GAZEBO_libraries} ${roscpp_LIBRARIES})

add_library(force_plugin SHARED force_plugin.cc)
target_link_libraries(force_plugin ${GAZEBO_libraries} ${roscpp_LIBRARIES})
