cmake_minimum_required(VERSION 3.0.2)
project(img2ptcloud)

find_package(catkin REQUIRED
  roscpp
  rosbag # for main_bagfile_driver
  cmake_modules
  cv_bridge
  geometry_msgs
  sensor_msgs
  image_geometry
  image_transport
  nodelet
  std_msgs
  tf
  tf2
  tf2_ros
  pcl_ros
)

set(PCL_FIND_QUIETLY ON)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common sample_consensus io)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(CMAKE_VERBOSE_MAKEFILE ON)
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(SOURCE_FILES         
  src/img2ptcloud.cpp
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
) 

catkin_package(
  INCLUDE_DIRS include
#  CATKIN_DEPENDS boost cv_bridge eigen geometry_msgs image_geometry image_transport nodelet pcl_ros std_msgs tf tf2 tf2_ros visualization_msgs
  CATKIN_DEPENDS roscpp roslib cv_bridge sensor_msgs geometry_msgs pcl_ros
#  DEPENDS system_lib
  DEPENDS OpenCV 
  LIBRARIES ${PROJECT_NAME}
)

include_directories(include
        SYSTEM ${catkin_INCLUDE_DIRS}
               ${OPENCV_INCLUDE_DIRS}
               ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} src/img2ptcloud.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES})
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)