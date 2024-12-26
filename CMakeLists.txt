cmake_minimum_required(VERSION 3.8)
project(mtofalib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Use link path
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
# Supress warnings
add_definitions(-w) 

#######################################################################
############################ Find thigs ###############################
#######################################################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

#######################################################################
########################### ROS Messages ##############################
#######################################################################
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Mtof.msg"
  DEPENDENCIES std_msgs
)

# Export dependencies for generated messages (so other packages can use them)
ament_export_dependencies(rosidl_default_runtime)

#######################################################################
########################## INCL & LIBS ################################
#######################################################################
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

set(LIBS  
)

#######################################################################
########################## CPP EXEC ###################################
#######################################################################
add_executable(mtof_publisher src/mtof_publisher.cpp)
rosidl_target_interfaces(mtof_publisher ${PROJECT_NAME} "rosidl_typesupport_cpp")
ament_target_dependencies(mtof_publisher rclcpp)
target_link_libraries(mtof_publisher
  ${LIBS}
)

#######################################################################
########################## INSTALL ####################################
#######################################################################
# Install C++ executables
install(TARGETS
  mtof_publisher
DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/calibrator.py
  scripts/uvc_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)


#######################################################################
########################### Final #####################################
#######################################################################
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
