cmake_minimum_required(VERSION 2.8.3)

add_subdirectory(despot_ext)


#
add_subdirectory(graspingV4)
add_subdirectory(grasping_despot_box2D)


#install(TARGETS "${PROJECT_NAME}"
#  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
#)
