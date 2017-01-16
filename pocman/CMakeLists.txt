cmake_minimum_required(VERSION 2.8.3)

project(pocman)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -Wall -Wno-sign-compare")
#LDFLAGS = -O3 -std=c++0x -Wno-sign-compare
set(DESPOT_HOME "../despot/")
set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

include_directories("${DESPOT_HOME}/include" "../despot_ext/include" "../graspingV4/src")
find_library(DESPOT_LIB NAMES despot HINTS "${DESPOT_HOME}/build")
find_library(DESPOT_EXT_LIB NAMES despot_ext HINTS "../build/despot_ext")
add_executable("${PROJECT_NAME}"
  src/pocman.cpp
  src/main.cpp
)

target_link_libraries("${PROJECT_NAME}"
  "${DESPOT_LIB}"
  "${DESPOT_EXT_LIB}"
)

install(TARGETS "${PROJECT_NAME}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)