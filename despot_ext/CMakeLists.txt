cmake_minimum_required(VERSION 2.8.3)

project(despot_ext)

set(DESPOT_HOME "../despot/")
set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

include_directories("${DESPOT_HOME}/include" "include")
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


find_library(DESPOT_LIB NAMES despot HINTS "${DESPOT_HOME}/build")

add_library("${PROJECT_NAME}" SHARED
  src/DeepLearningSolver.cpp
  src/LearningModel.cpp
  src/LearningPlanningSolver.cpp
  src/DespotWithLearnedDefaultPolicy.cpp
  #src/DespotWithBeliefTracking.cpp

)
target_link_libraries("${PROJECT_NAME}"
  "${DESPOT_LIB}"
  "${PYTHON_LIBRARIES}"
)

#install(TARGETS "${PROJECT_NAME}"
#  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
#)
