cmake_minimum_required(VERSION 3.0.2)
project(gpcoctomap)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    pcl_ros
    roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gpcoctomap
  #CATKIN_DEPENDS pcl_ros octomap_ros geometry_msgs visualization_msgs roscpp
#  DEPENDS system_lib
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include/gpcoctomap
        include/common
        ${catkin_INCLUDE_DIRS}
)

# Files for GPC source
set(GPC_SOURCE_FILES
  src/gpcoctomap/gpcblock.cpp
  src/gpcoctomap/gpcoctomap.cpp
  src/gpcoctomap/gpcoctree_node.cpp
  src/gpcoctomap/gpcoctree.cpp
)

add_executable(gpcoctomap_server
    ${GPC_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/gpcoctomap/gpcoctomap_server.cpp
)


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(gpcoctomap_server
        ${catkin_LIBRARIES}
)

