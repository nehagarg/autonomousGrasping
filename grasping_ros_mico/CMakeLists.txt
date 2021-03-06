cmake_minimum_required(VERSION 2.4.6)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

project(grasping_ros_mico)
# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
#rosbuild_add_generated_msgs(Belief.msg)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#set(DESPOT_HOME "../despot/")
set(HYPDESPOTALPHA_HOME "../hyp-despot-alpha/")

set(CMAKE_CXX_COMPILER "g++")
#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()


#create despot lib
# search header files in those sub directories
# Maybe a glob could be possible
INCLUDE_DIRECTORIES(
	#src/despot/
        src/Display
        #msg_gen/cpp/include/despot_vrep_glue
)
set(WARNING_FLAGS -Wall -Wextra -Wno-sign-compare)
set(PED_COMPILE_FLAGS -std=c++0x -O3 -c ${WARNING_FLAGS})

#FILE(GLOB LIBDESPOT_SRCS src/despot/*.cpp src/despot/util/*.cpp src/despot/util/pomdpx_parser/*.cpp src/despot/problems/*.cpp)
#rosbuild_add_library(despot ${LIBDESPOT_SRCS})
#rosbuild_add_compile_flags(despot ${PED_COMPILE_FLAGS})

include_directories( 
"${DESPOT_HOME}/include" 
"../despot_ext/include" 
"../graspingV4/src/"
)

find_library(DESPOT_LIB NAMES despot HINTS "${DESPOT_HOME}/build")
find_library(DESPOT_EXT_LIB NAMES despot_ext HINTS "../build/despot_ext")

 find_package( Boost COMPONENTS system REQUIRED )  
 link_directories( ${Boost_LIBRARY_DIRS} )  
 include_directories( ${Boost_INCLUDE_DIRS} "/usr/include/python2.7")  

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_library(YAML-CPP yaml-cpp /usr/local/lib)
FILE(GLOB EXEC_SRCS src/*.cpp src/Display/c*.cpp)
set(WITHOUT_DISPLAY_SRCS ${EXEC_SRCS})
set(GATHER_DATA_SRCS ${EXEC_SRCS})
list(REMOVE_ITEM EXEC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_gather_data.cpp")
list(REMOVE_ITEM EXEC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_without_display.cpp")
list(REMOVE_ITEM WITHOUT_DISPLAY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_gather_data.cpp")
list(REMOVE_ITEM WITHOUT_DISPLAY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM GATHER_DATA_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM GATHER_DATA_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_without_display.cpp")

rosbuild_add_executable(grasping_ros_mico ${EXEC_SRCS})
rosbuild_add_executable(despot_without_display ${WITHOUT_DISPLAY_SRCS})
rosbuild_add_executable(gather_data ${GATHER_DATA_SRCS})

target_link_libraries(grasping_ros_mico ${DESPOT_LIB})
target_link_libraries(grasping_ros_mico ${DESPOT_EXT_LIB})
target_link_libraries(grasping_ros_mico ${YAML-CPP})
target_link_libraries( grasping_ros_mico ${Boost_SYSTEM_LIBRARY} )  
target_link_libraries(grasping_ros_mico ${PYTHON_LIBRARIES})

target_link_libraries(despot_without_display ${DESPOT_LIB})
target_link_libraries(despot_without_display ${DESPOT_EXT_LIB})
target_link_libraries(despot_without_display ${YAML-CPP})
target_link_libraries(despot_without_display ${Boost_SYSTEM_LIBRARY} )  
target_link_libraries(despot_without_display ${PYTHON_LIBRARIES})

target_link_libraries(gather_data ${DESPOT_LIB})
target_link_libraries(gather_data ${DESPOT_EXT_LIB})
target_link_libraries(gather_data ${YAML-CPP})
target_link_libraries( gather_data ${Boost_SYSTEM_LIBRARY} )  
target_link_libraries(gather_data ${PYTHON_LIBRARIES})

rosbuild_add_compile_flags(grasping_ros_mico ${PED_COMPILE_FLAGS})
rosbuild_add_compile_flags(despot_without_display ${PED_COMPILE_FLAGS})
rosbuild_add_compile_flags(gather_data ${PED_COMPILE_FLAGS})

rosbuild_add_executable(display src/Display/mainCupDisplay.cpp src/Display/cupDisplay.cpp)
target_link_libraries(display ${Boost_SYSTEM_LIBRARY} )  
#rosbuild_add_compile_flags(mico_despot_planner ${PED_COMPILE_FLAGS})
