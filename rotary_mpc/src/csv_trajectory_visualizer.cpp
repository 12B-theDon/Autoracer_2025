#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

struct CsvRow {
  double t, x, y, yaw, v;
};

static bool load_csv(const std::string& path, std::vector<CsvRow>& rows) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    RCLCPP_ERROR(rclcpp::get_logger("CsvLoader"), "Cannot open CSV file: %s", path.c_str());
    return false;
  }

  std::string line;
  if (!std::getline(ifs, line)) return false;  // header skip

  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string tok;
    CsvRow r{};

    // timestep,x,y,yaw,velocity
    if (!std::getline(ss, tok, ',')) continue;
    r.t = std::stod(tok);

    if (!std::getline(ss, tok, ',')) continue;
    r.x = std::stod(tok);

    if (!std::getline(ss, tok, ',')) continue;
    r.y = std::stod(tok);

    if (!std::getline(ss, tok, ',')) continue;
    r.yaw = std::stod(tok);

    if (!std::getline(ss, tok, ',')) continue;
    r.v = std::stod(tok);

    rows.push_back(r);
  }
  return true;
}

class CsvTrajectoryViz : public rclcpp::Node {
public:
  CsvTrajectoryViz() : Node("csv_trajectory_viz") {
    csv_path_ = this->declare_parameter<std::string>("csv_path", "straight_trajectory_100.csv");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");
    path_topic_ = this->declare_parameter<std::string>("path_topic", "/csv_path");
    marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/csv_marker");
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 1.0);

    if (!load_csv(csv_path_, data_)) {
      RCLCPP_FATAL(get_logger(), "Failed to read CSV: %s", csv_path_.c_str());
      rclcpp::shutdown();
      return;
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu points.", data_.size());

    // QoS: RViz가 뒤늦게 붙어도 한 번에 전체 마커를 받을 수 있도록 depth를 크게 설정
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1024)).transient_local().reliable();

    path_pub_   = this->create_publisher<nav_msgs::msg::Path>(path_topic_, qos);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(marker_topic_, qos);

    build_path_msg_();
    build_line_marker_msg_();
    build_arrows_();

    // 화살표는 한 번만 발행
    publish_arrows_once_();
    RCLCPP_INFO(get_logger(), "Published %zu arrows (once).", arrows_.size());

    // 주기적으로 Path/Line만 재게시 (RViz 새로 켰을 때도 보이도록)
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1e-3, publish_rate_hz_))),
      std::bind(&CsvTrajectoryViz::tick, this));
  }

private:
  void build_path_msg_() {
    path_msg_.header.frame_id = frame_id_;
    path_msg_.poses.clear();
    path_msg_.poses.reserve(data_.size());

    double t0 = data_.empty() ? 0.0 : data_.front().t;
    for (const auto& r : data_) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = frame_id_;
      ps.header.stamp = rclcpp::Time((r.t - t0) * 1e9);
      ps.pose.position.x = r.x;
      ps.pose.position.y = r.y;
      ps.pose.position.z = 0.0;

      // yaw -> quaternion (Z축 회전)
      double cy = std::cos(r.yaw * 0.5);
      double sy = std::sin(r.yaw * 0.5);
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = sy;
      ps.pose.orientation.w = cy;

      path_msg_.poses.push_back(ps);
    }
  }

  void build_line_marker_msg_() {
    marker_line_.header.frame_id = frame_id_;
    marker_line_.ns = "csv_traj";
    marker_line_.id = 0;
    marker_line_.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker_line_.action = visualization_msgs::msg::Marker::ADD;
    marker_line_.scale.x = 0.05; // 선 두께
    marker_line_.color.r = 0.0f;
    marker_line_.color.g = 1.0f;
    marker_line_.color.b = 1.0f;
    marker_line_.color.a = 1.0f;

    marker_line_.points.clear();
    marker_line_.points.reserve(data_.size());
    for (const auto& r : data_) {
      geometry_msgs::msg::Point p;
      p.x = r.x; p.y = r.y; p.z = 0.0;
      marker_line_.points.push_back(p);
    }
  }

  void build_arrows_() {
    // 과부하 방지용 샘플링(최대 약 50개)
    arrow_step_ = std::max<size_t>(1, data_.size() / 50);
    arrows_.clear();
    arrows_.reserve((data_.size() + arrow_step_ - 1) / arrow_step_);

    auto now = this->now();
    int id = arrow_id_base_;
    for (size_t i = 0; i < data_.size(); i += arrow_step_) {
      const auto& r = data_[i];

      visualization_msgs::msg::Marker arrow;
      arrow.header.frame_id = frame_id_;
      arrow.header.stamp = now;                // 초기 발행 시각
      arrow.ns = "csv_traj_dir";
      arrow.id = id++;                         // 고유 ID (tick마다 바뀌지 않음)
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;

      // 속도로 길이 스케일링(0.3~2.0 m)
      double len = std::max(0.3, std::min(2.0, r.v));
      arrow.scale.x = len;
      arrow.scale.y = 0.08;
      arrow.scale.z = 0.08;

      arrow.color.r = 1.0f;
      arrow.color.g = 0.6f;
      arrow.color.b = 0.0f;
      arrow.color.a = 0.9f;

      geometry_msgs::msg::Pose p;
      p.position.x = r.x;
      p.position.y = r.y;
      p.position.z = 0.0;

      double cy = std::cos(r.yaw * 0.5);
      double sy = std::sin(r.yaw * 0.5);
      p.orientation.x = 0.0;
      p.orientation.y = 0.0;
      p.orientation.z = sy;
      p.orientation.w = cy;

      arrow.pose = p;

      // lifetime=0 -> 계속 유지 (transient_local + 충분한 depth로 late-join도 보장)
      arrow.lifetime = rclcpp::Duration(0, 0);

      arrows_.push_back(std::move(arrow));
    }
  }

  void publish_arrows_once_() {
    for (const auto& m : arrows_) {
      marker_pub_->publish(m);
    }
  }

  void tick() {
    auto now = this->now();
    path_msg_.header.stamp = now;
    marker_line_.header.stamp = now;

    path_pub_->publish(path_msg_);
    marker_pub_->publish(marker_line_);
    // (요청: 최소 출력) 주기 로그 생략
  }

  // params
  std::string csv_path_, frame_id_, path_topic_, marker_topic_;
  double publish_rate_hz_{1.0};

  // data & msgs
  std::vector<CsvRow> data_;
  nav_msgs::msg::Path path_msg_;
  visualization_msgs::msg::Marker marker_line_;
  std::vector<visualization_msgs::msg::Marker> arrows_;

  // arrow config
  size_t arrow_step_{10};
  int arrow_id_base_{100};

  // pubs & timer
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CsvTrajectoryViz>());
  rclcpp::shutdown();
  return 0;
}
