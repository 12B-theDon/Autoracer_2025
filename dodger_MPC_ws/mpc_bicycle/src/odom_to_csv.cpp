#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <optional>

class OdomCsvLogger : public rclcpp::Node {
public:
  OdomCsvLogger()
  : rclcpp::Node("odom_csv_logger")
  {
    // ---- Parameters ----
    odom_topic_   = this->declare_parameter<std::string>("odom_topic", "/odom");
    output_path_  = this->declare_parameter<std::string>("output_path", "trajectory.csv");
    qos_depth_    = this->declare_parameter<int>("qos_depth", 100);
    use_wall_time_= this->declare_parameter<bool>("use_wall_time", false);

    // ---- Open CSV & write header ----
    openCsv_(output_path_);
    writeHeader_();

    // ---- Time base ----
    // ROS 시각을 사용할 때는 메시지 stamp 기준 상대시간.
    // 벽시계(use_wall_time=true)일 땐 노드 시작 시각을 t0로.
    if (use_wall_time_) {
      rclcpp::Clock wall_clock(RCL_SYSTEM_TIME);
      t0_wall_sec_ = wall_clock.now().seconds();
    }

    // ---- Subscription ----
    rclcpp::QoS qos(rclcpp::KeepLast(static_cast<size_t>(qos_depth_)));
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&OdomCsvLogger::odomCb_, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(),
      "Writing trajectory CSV to: %s\n- topic: %s\n- use_wall_time: %s",
      output_path_.c_str(), odom_topic_.c_str(), use_wall_time_ ? "true" : "false");
  }

  ~OdomCsvLogger() override {
    std::lock_guard<std::mutex> lk(mtx_);
    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }
  }

private:
  // yaw from quaternion (x,y,z,w), ZYX convention
  static double quatToYaw_(double x, double y, double z, double w) {
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  void openCsv_(const std::string &path) {
    ofs_.open(path, std::ios::out | std::ios::trunc);
    if (!ofs_) {
      throw std::runtime_error("Failed to open CSV file: " + path);
    }
    // 고정 소수점 6자리
    ofs_ << std::fixed << std::setprecision(6);
  }

  void writeHeader_() {
    std::lock_guard<std::mutex> lk(mtx_);
    ofs_ << "timestep,x,y,yaw,velocity\n";
    ofs_.flush();
  }

  void odomCb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
    double timestep = 0.0;

    if (use_wall_time_) {
      rclcpp::Clock wall_clock(RCL_SYSTEM_TIME);
      const double now_sec = wall_clock.now().seconds();
      timestep = now_sec - t0_wall_sec_;
    } else {
      const double t_msg = static_cast<double>(msg->header.stamp.sec)
                         + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
      if (!t0_msg_sec_.has_value()) {
        t0_msg_sec_ = t_msg;
      }
      timestep = t_msg - t0_msg_sec_.value();
    }

    const double px = static_cast<double>(msg->pose.pose.position.x);
    const double py = static_cast<double>(msg->pose.pose.position.y);

    const double qx = static_cast<double>(msg->pose.pose.orientation.x);
    const double qy = static_cast<double>(msg->pose.pose.orientation.y);
    const double qz = static_cast<double>(msg->pose.pose.orientation.z);
    const double qw = static_cast<double>(msg->pose.pose.orientation.w);
    const double yaw = quatToYaw_(qx, qy, qz, qw);

    const double vx = static_cast<double>(msg->twist.twist.linear.x);
    const double vy = static_cast<double>(msg->twist.twist.linear.y);
    const double vz = static_cast<double>(msg->twist.twist.linear.z);
    const double speed = std::sqrt(vx*vx + vy*vy + vz*vz);

    {
      std::lock_guard<std::mutex> lk(mtx_);
      ofs_ << timestep << "," << px << "," << py << "," << yaw << "," << speed << "\n";
      ofs_.flush(); // 빈번 flush가 부담되면 제거 가능
    }
  }

  // Params
  std::string odom_topic_;
  std::string output_path_;
  int qos_depth_{100};
  bool use_wall_time_{false};

  // File
  std::ofstream ofs_;
  std::mutex mtx_;

  // Time bases
  std::optional<double> t0_msg_sec_;
  double t0_wall_sec_{0.0};

  // ROS
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OdomCsvLogger>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
