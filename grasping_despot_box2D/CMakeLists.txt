cmake_minimum_required(VERSION 2.8.3)

project(grasping_despot_box2D)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -Wall -Wno-sign-compare")
set(DESPOT_HOME "/home/neha/WORK_FOLDER/phd2013/phdTopic/AdaCompNus/despot/")
set(Box2D_HOME "/home/neha/WORK_FOLDER/phd2013/phdTopic/box2D/Box2D_v2.3.1/Box2D/")
set(Box2D_KENV_HOME "/home/neha/WORK_FOLDER/phd2013/phdTopic/MichaelKoval_github/kenv/box2d_kenv")
set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS cmake_modules)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)
find_package(SFML REQUIRED COMPONENTS graphics system window)

# The FindBox2D.cmake script does not set BOX2D_LIBRARIES.
find_package(Box2D REQUIRED)
set(BOX2D_LIBRARIES Box2D)

include_directories(${Eigen_INCLUDE_DIRS} ${SFML_INCLUDE_DIRS} ${BOOST_INCLUDE_DIRS} "${DESPOT_HOME}/include" "../despot_ext/include" "../graspingV4/src/" "${Box2D_HOME}" "${Box2D_KENV_HOME}/include")
include_directories(${Eigen_INCLUDE_DIRS} 
${SFML_INCLUDE_DIRS} 
${BOOST_INCLUDE_DIRS} 
${BOX2D_INCLUDE_DIRS} 
"${DESPOT_HOME}/include" 
"../despot_ext/include" 
"../graspingV4/src/"
 "${Box2D_KENV_HOME}/include")

find_library(DESPOT_LIB NAMES despot HINTS "${DESPOT_HOME}/build")
#find_library(Box2D_LIB NAMES Box2D HINTS "${Box2D_HOME}/Build/Box2D")
find_library(BOX2D_KENV_LIB NAMES box2d_kenv HINTS "${Box2D_KENV_HOME}/build/devel/lib")
find_library(BOX2D_KENV_VISUALIZER_LIB NAMES box2d_kenv_visualizer HINTS "${Box2D_KENV_HOME}/build/devel/lib")



add_executable("${PROJECT_NAME}"
  src/box2d_grasping_world.cpp
  src/grasping_box2D.cpp
  src/main.cpp
)

target_link_libraries("${PROJECT_NAME}"
  "${DESPOT_LIB}"
   despot_ext
   #graspingV4_lib
   "${Box2D_LIBRARIES}"
   "${BOX2D_KENV_LIB}"
)






add_executable(${PROJECT_NAME}_step_visualize
  src/box2d_grasping_world.cpp
  src/step_visualize_main.cpp
)
target_link_libraries(${PROJECT_NAME}_step_visualize
  ${BOX2D_KENV_LIB}
  ${BOX2D_KENV_VISUALIZER_LIB}
  ${BOX2D_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${SFML_LIBRARIES}
)


install(TARGETS "${PROJECT_NAME}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)
