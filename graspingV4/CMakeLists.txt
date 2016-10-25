cmake_minimum_required(VERSION 2.8.3)

project(graspingV4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -Wall -Wno-sign-compare")
#LDFLAGS = -O3 -std=c++0x -Wno-sign-compare
set(DESPOT_HOME "/home/neha/WORK_FOLDER/phd2013/phdTopic/AdaCompNus/despot/")
set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

include_directories("${DESPOT_HOME}/include" "../despot_ext/include")
find_library(DESPOT_LIB NAMES despot HINTS "${DESPOT_HOME}/build")
add_executable("${PROJECT_NAME}"
  src/grasping_v4.cpp
  src/main.cpp
)
target_link_libraries("${PROJECT_NAME}"
  "${DESPOT_LIB}"
  despot_ext
  yaml-cpp
)
install(TARGETS "${PROJECT_NAME}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)
