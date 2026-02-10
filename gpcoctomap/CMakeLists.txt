cmake_minimum_required(VERSION 3.8)
project(gpcoctomap)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================
# Dependencies
# ======================
find_package(ament_cmake REQUIRED)
# ROS2 Pkgs
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)        
find_package(sensor_msgs REQUIRED)     
find_package(geometry_msgs REQUIRED)  
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# System Pkgs
find_package(PCL REQUIRED)


# ======================
# Source files
# ======================
set(GPC_SOURCE_FILES
  src/gpcoctomap/gpcblock.cpp
  src/gpcoctomap/gpcoctomap.cpp
  src/gpcoctomap/gpcoctree_node.cpp
  src/gpcoctomap/gpcoctree.cpp
)

# ======================
# Executable
# ======================
add_executable(gpcoctomap_server
  ${GPC_SOURCE_FILES}
  src/gpcoctomap/gpcoctomap_server.cpp
)

# ======================
# Include directories
# ======================
#include_directories(
#  include
#  include/gpcoctomap
#  include/common
#)
target_include_directories(gpcoctomap_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}  # CRITICAL: This fixes "pcl/point_cloud.h not found"
)

# ======================
# Link dependencies
# ======================
target_include_directories(gpcoctomap_server PUBLIC
  # Look in the standard include folder
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  
  # Look in the source folder (where your .h files currently are)
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/gpcoctomap>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>

  # Look in the install folder (for when the package is installed)
  $<INSTALL_INTERFACE:include>
  
  # Look for PCL headers
  ${PCL_INCLUDE_DIRS}
)

# ROS2 LIbraries
ament_target_dependencies(gpcoctomap_server
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  pcl_conversions 
)

# ======================
# Install
# ======================
install(
  TARGETS gpcoctomap_server
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ======================
# Testing
# ======================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

