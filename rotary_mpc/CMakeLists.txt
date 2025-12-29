cmake_minimum_required(VERSION 3.5)
project(object_detector)

# C++ 표준 (필요시 14 이상으로 설정)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# ament_cmake 및 의존 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(Eigen3 REQUIRED)  
find_package(tf2)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rclpy REQUIRED) 

# scan_processor_node: 레이저 스캔 전처리 및 장애물 후보 생성
add_executable(scan_processor_node src/scan_processor_node.cpp)
ament_target_dependencies(scan_processor_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

add_executable(knn_tracker_node
  src/knn_tracker_node.cpp
)

ament_target_dependencies(knn_tracker_node
  rclcpp
  geometry_msgs
  nav_msgs
  visualization_msgs
)

add_executable(publish_opp_odom
  src/publish_opp_odom.cpp
)

ament_target_dependencies(publish_opp_odom
  rclcpp
  nav_msgs
  visualization_msgs
)

add_executable(mpc_bicycle_node
  src/mpc_bicycle_node.cpp
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

# 설치 대상 설정
install(TARGETS
  mpc_bicycle_node
  scan_processor_node
  knn_tracker_node
  publish_opp_odom
  DESTINATION lib/${PROJECT_NAME}
)

# launch 파일이나 기타 리소스 디렉토리도 설치할 경우
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
