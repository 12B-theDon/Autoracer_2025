cmake_minimum_required(VERSION 3.8)
project(mpc_bicycle)

# -----------------------
# Compiler & C++ standard
# -----------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------
# Dependencies
# -------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)                # for MPC math
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
# find_package(tf2_geometry_msgs REQUIRED)   # 필요 시 주석 해제


add_executable(mpc_bicycle_node
  src/mpc_bicycle_node.cpp
)

add_executable(odom_to_csv_node
  src/odom_to_csv.cpp
)


add_executable(csv_trajectory_viz_node
  src/csv_trajectory_visualizer.cpp
)

add_executable(csv_downsampler
  src/csv_downsampler.cpp
)

add_executable(odom_tf_broadcaster
  src/odom_tf_broadcaster.cpp
)

target_include_directories(mpc_bicycle_node PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(mpc_bicycle_node
  Eigen3::Eigen
)

ament_target_dependencies(mpc_bicycle_node
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  # tf2_geometry_msgs
)

ament_target_dependencies(odom_to_csv_node
  rclcpp
  nav_msgs
)

ament_target_dependencies(csv_trajectory_viz_node
  rclcpp
  nav_msgs
  visualization_msgs
  geometry_msgs
)

ament_target_dependencies(odom_tf_broadcaster
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_ros
)

# -------
# Install
# -------
install(TARGETS
  mpc_bicycle_node
  odom_to_csv_node
  csv_trajectory_viz_node
  csv_downsampler
  odom_tf_broadcaster
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
