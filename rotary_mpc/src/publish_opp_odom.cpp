#include <algorithm>
#include <cmath>
#include <deque>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker.hpp"

class PublishOppOdomNode : public rclcpp::Node
{
public:
  PublishOppOdomNode()
  : Node("publish_opp_odom"),
    marker_topic_(this->declare_parameter<std::string>("marker_topic", "moving_center_marker")),
    odom_topic_(this->declare_parameter<std::string>("odom_topic", "opponent_odometry")),
    child_frame_id_(this->declare_parameter<std::string>("odom_child_frame", "opponent")),
    closest_distance_threshold_(this->declare_parameter<double>("closest_distance_threshold", 0.8)),
    noise_distance_threshold_(this->declare_parameter<double>("noise_distance_threshold", 0.20)),
    new_position_reset_threshold_(this->declare_parameter<double>("new_position_reset_threshold", 1.0)),
    trajectory_capacity_(static_cast<size_t>(this->declare_parameter<int>("trajectory_capacity", 10))),
    jump_limit_(static_cast<size_t>(this->declare_parameter<int>("trajectory_jump_limit", 3))),
    closest_distance_capacity_(static_cast<size_t>(this->declare_parameter<int>("closest_distance_capacity", 30))),
    initial_distance_requirement_(static_cast<size_t>(this->declare_parameter<int>("initial_distance_requirement", 30)))
  {
    marker_sub_ = this->create_subscription<visualization_msgs::msg::Marker>(
      marker_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PublishOppOdomNode::markerCallback, this, std::placeholders::_1));
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    rotary_pub_ = this->create_publisher<std_msgs::msg::Bool>("mpc_start_rotary", 1);
  }

private:
  struct TrajectoryPoint
  {
    geometry_msgs::msg::Point position;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };

  void markerCallback(const visualization_msgs::msg::Marker::SharedPtr msg)
  {
    const geometry_msgs::msg::Point raw_position = msg->pose.position;
    const rclcpp::Time stamp(msg->header.stamp);

    const bool new_segment = trajectory_.empty() || isNewPosition(raw_position);
    if (new_segment)
      resetTrajectory();

    geometry_msgs::msg::Point processed_position =
      new_segment ? raw_position : handleNoise(raw_position, stamp);

    geometry_msgs::msg::Point delta{};
    geometry_msgs::msg::Vector3 velocity{};
    if (has_previous_)
    {
      delta.x = processed_position.x - previous_position_.x;
      delta.y = processed_position.y - previous_position_.y;
      delta.z = processed_position.z - previous_position_.z;

      const double dt = (stamp - previous_stamp_).seconds();
      if (dt > 1e-6)
      {
        velocity.x = delta.x / dt;
        velocity.y = delta.y / dt;
        velocity.z = delta.z / dt;
      }

      const double planar_mag = std::hypot(delta.x, delta.y);
      if (planar_mag > 1e-6)
        heading_ = std::atan2(delta.y, delta.x);
    }

    previous_position_ = processed_position;
    previous_stamp_ = stamp;
    has_previous_ = true;

    addTrajectoryPoint(processed_position, stamp);
    updateClosestDistance(processed_position);
    publishOdometry(msg->header, processed_position, velocity);
  }

  geometry_msgs::msg::Point handleNoise(
    const geometry_msgs::msg::Point &candidate,
    const rclcpp::Time &stamp)
  {
    const double displacement = planarDistance(candidate, trajectory_.back().position);
    if (displacement > noise_distance_threshold_)
      ++consecutive_large_jumps_;
    else
      consecutive_large_jumps_ = 0;

    const bool drop_noise = trajectory_.size() < trajectory_capacity_ &&
      consecutive_large_jumps_ >= jump_limit_ && trajectory_.size() >= 2;
    if (!drop_noise)
      return candidate;

    consecutive_large_jumps_ = 0;
    return interpolateFromHistory(stamp);
  }

  geometry_msgs::msg::Point interpolateFromHistory(const rclcpp::Time &stamp) const
  {
    geometry_msgs::msg::Point result = trajectory_.back().position;
    const auto &last = trajectory_.back();
    const auto &prev = *(trajectory_.rbegin() + 1);
    const double dt = (last.stamp - prev.stamp).seconds();
    const double future_dt = (stamp - last.stamp).seconds();

    if (dt > 1e-6 && future_dt > 0.0)
    {
      const double scale = future_dt / dt;
      result.x += (last.position.x - prev.position.x) * scale;
      result.y += (last.position.y - prev.position.y) * scale;
      result.z += (last.position.z - prev.position.z) * scale;
    }

    return result;
  }

  void addTrajectoryPoint(const geometry_msgs::msg::Point &position, const rclcpp::Time &stamp)
  {
    trajectory_.push_back({position, stamp});
    if (trajectory_.size() > trajectory_capacity_)
      trajectory_.pop_front();
  }

  void resetTrajectory()
  {
    trajectory_.clear();
    consecutive_large_jumps_ = 0;
    has_previous_ = false;
    heading_ = 0.0;
    closest_distance_history_.clear();
    rotary_flag_sent_ = false;
  }

  bool isNewPosition(const geometry_msgs::msg::Point &candidate) const
  {
    if (trajectory_.empty())
      return true;
    return planarDistance(candidate, trajectory_.back().position) > new_position_reset_threshold_;
  }

  double planarDistance(
    const geometry_msgs::msg::Point &a,
    const geometry_msgs::msg::Point &b) const
  {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
  }

  void updateClosestDistance(const geometry_msgs::msg::Point &position)
  {
    const double distance = planarDistanceFromVehicle(position);
    closest_distance_history_.push_back(distance);
    if (closest_distance_history_.size() > closest_distance_capacity_)
      closest_distance_history_.pop_front();

    const auto min_it = std::min_element(
      closest_distance_history_.begin(), closest_distance_history_.end());
    if (min_it == closest_distance_history_.end())
      return;

    const double min_distance = *min_it;
    //RCLCPP_INFO(
    //  this->get_logger(),
    //  "Latest marker distance: %.3fm, closest over last %zu samples: %.3fm",
    //  distance, closest_distance_history_.size(), min_distance);

    const bool queue_ready = closest_distance_history_.size() >= initial_distance_requirement_;
    if (!rotary_flag_sent_ && queue_ready && min_distance <= closest_distance_threshold_ &&
      std::fabs(distance - min_distance) < 1e-6)
    {
      rotary_flag_sent_ = true;
      RCLCPP_INFO(
        this->get_logger(),
        "mpc_start_rotary triggered at %.3fm (threshold %.3fm)",
        min_distance, closest_distance_threshold_);
      publishRotaryTrigger();
    }
  }

  double planarDistanceFromVehicle(const geometry_msgs::msg::Point &position) const
  {
    return std::sqrt(position.x * position.x + position.y * position.y);
  }

  void publishOdometry(
    const std_msgs::msg::Header &header,
    const geometry_msgs::msg::Point &position,
    const geometry_msgs::msg::Vector3 &velocity)
  {
    nav_msgs::msg::Odometry odom;
    odom.header = header;
    odom.child_frame_id = child_frame_id_;
    odom.pose.pose.position = position;
    const double half_yaw = heading_ * 0.5;
    odom.pose.pose.orientation.x = 0.0;
    odom.pose.pose.orientation.y = 0.0;
    odom.pose.pose.orientation.z = std::sin(half_yaw);
    odom.pose.pose.orientation.w = std::cos(half_yaw);
    odom.twist.twist.linear = velocity;
    odom_pub_->publish(odom);
  }

  void publishRotaryTrigger()
  {
    std_msgs::msg::Bool msg;
    msg.data = true;
    rotary_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Published mpc_start_rotary flag");
  }

  std::string marker_topic_;
  std::string odom_topic_;
  std::string child_frame_id_;
  double closest_distance_threshold_;
  double noise_distance_threshold_;
  double new_position_reset_threshold_;
  size_t trajectory_capacity_;
  size_t jump_limit_;
  size_t closest_distance_capacity_;
  size_t initial_distance_requirement_;

  std::deque<TrajectoryPoint> trajectory_;
  size_t consecutive_large_jumps_ = 0;

  bool has_previous_ = false;
  geometry_msgs::msg::Point previous_position_{};
  rclcpp::Time previous_stamp_{0, 0, RCL_ROS_TIME};
  double heading_ = 0.0;

  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr marker_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr rotary_pub_;

  std::deque<double> closest_distance_history_;
  bool rotary_flag_sent_ = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PublishOppOdomNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
