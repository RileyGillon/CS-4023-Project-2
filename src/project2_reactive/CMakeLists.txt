cmake_minimum_required(VERSION 3.8)
project(project2_reactive)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(irobot_create_msgs REQUIRED)

add_library(behaviors src/behaviors.cpp)
target_include_directories(behaviors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(behaviors PUBLIC c_std_99 cxx_std_17)

add_executable(reactive_controller src/reactive_controller.cpp)
target_include_directories(reactive_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(reactive_controller PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(reactive_controller
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  irobot_create_msgs)
target_link_libraries(reactive_controller behaviors)

install(TARGETS
  behaviors
  reactive_controller
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_behaviors test/test_behaviors.cpp)
  if(TARGET test_behaviors)
    target_include_directories(test_behaviors PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
    target_compile_features(test_behaviors PUBLIC cxx_std_17)
    target_link_libraries(test_behaviors behaviors)
  endif()
endif()

ament_package()
