#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"

enum class ClusterColor
{
  RED = 0,
  BLUE = 1
};

struct CenterState
{
  bool valid = false;
  geometry_msgs::msg::Point current;
  geometry_msgs::msg::Point previous;
  double last_movement = 0.0;
};

class KNNTrackerNode : public rclcpp::Node
{
public:
  KNNTrackerNode()
  : Node("knn_tracker_node"),
    cluster_topic_(this->declare_parameter<std::string>("cluster_topic", "cluster_centers")),
    movement_threshold_(this->declare_parameter<double>("movement_threshold", 0.01)),
    switch_margin_(this->declare_parameter<double>("movement_switch_margin", 0.005)),
    range_window_size_(static_cast<size_t>(std::max<int>(1, this->declare_parameter<int>("range_window_size", 30)))),
    approach_threshold_(this->declare_parameter<double>("approach_threshold", 0.01)),
    center_window_size_(static_cast<size_t>(std::max<int>(1, this->declare_parameter<int>("center_window_size", 5)))),
    center_box_limit_(this->declare_parameter<double>("center_box_limit", 0.3)),
    center_proximity_threshold_(this->declare_parameter<double>("center_proximity_threshold", 0.2)),
    trajectory_window_size_(static_cast<size_t>(std::max<int>(1, this->declare_parameter<int>("trajectory_window_size", 20)))),
    trajectory_far_threshold_(this->declare_parameter<double>("trajectory_far_threshold", 0.3)),
    trajectory_persistence_frames_(std::max<int>(1, this->declare_parameter<int>("trajectory_persistence_frames", 20)))
  {
    cluster_sub_ = this->create_subscription<visualization_msgs::msg::Marker>(
      cluster_topic_, rclcpp::SensorDataQoS(),
      std::bind(&KNNTrackerNode::clusterCallback, this, std::placeholders::_1));

    component_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "moving_center_marker", 10);
  }

private:
  void clusterCallback(const visualization_msgs::msg::Marker::SharedPtr msg)
  {
    if (msg->type != visualization_msgs::msg::Marker::POINTS)
      return;

    if (msg->ns == "centers")
    {
      handleCenters(*msg);
      publishMovingCenter();
    }
    else if (msg->ns == "clusters")
    {
      handleClusterComponents(*msg);
      publishMovingCenter();
    }
  }

  void handleCenters(const visualization_msgs::msg::Marker &marker)
  {
    if (marker.points.empty())
      return;

    for (size_t i = 0; i < marker.points.size(); ++i)
    {
      ClusterColor color = inferColor(marker.colors, i);
      updateCenterState(color, marker.points[i]);
    }

    updateRangeAndCluster();
  }

  void handleClusterComponents(const visualization_msgs::msg::Marker &marker)
  {
    if (marker.points.empty() || marker.colors.size() != marker.points.size())
      return;

    latest_cluster_header_ = marker.header;
    for (auto &vec : cluster_points_)
      vec.clear();

    for (size_t i = 0; i < marker.points.size(); ++i)
    {
      ClusterColor color = inferColor(marker.colors, i);
      cluster_points_[static_cast<size_t>(color)].push_back(marker.points[i]);
    }
  }

  void publishMovingCenter()
  {
    if (!moving_cluster_.has_value())
      return;
    if (!isCenterWithinBounds(*moving_cluster_))
      return;
    if (centersTooClose())
      return;
    if (!latest_cluster_header_.has_value())
      return;

    const auto &state = centers_[static_cast<size_t>(*moving_cluster_)];
    if (!state.valid)
      return;

    visualization_msgs::msg::Marker marker;
    marker.header = *latest_cluster_header_;
    marker.ns = "moving_center";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.1;
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    marker.pose.position = state.current;
    component_pub_->publish(marker);
  }

  ClusterColor inferColor(const std::vector<std_msgs::msg::ColorRGBA> &colors, size_t idx) const
  {
    if (idx >= colors.size())
      return ClusterColor::RED;
    const auto &c = colors[idx];
    return (c.r >= c.b) ? ClusterColor::RED : ClusterColor::BLUE;
  }

  void updateCenterState(ClusterColor color, const geometry_msgs::msg::Point &p)
  {
    if (!acceptTrajectorySample(color, p))
      return;

    auto &state = centers_[static_cast<size_t>(color)];
    if (state.valid)
    {
      double jump = std::hypot(p.x - state.current.x, p.y - state.current.y);
      state.last_movement = jump;
    }
    else
    {
      state.last_movement = 0.0;
    }
    state.previous = state.current;
    state.current = p;
    state.valid = true;

    auto &history = center_histories_[static_cast<size_t>(color)];
    history.push_back(state.current);
    if (history.size() > center_window_size_)
      history.pop_front();
  }

  void updateRangeAndCluster()
  {
    for (ClusterColor color : {ClusterColor::RED, ClusterColor::BLUE})
    {
      const auto &state = centers_[static_cast<size_t>(color)];
      if (!state.valid)
        continue;
      double dist = std::hypot(state.current.x, state.current.y);
      range_histories_[static_cast<size_t>(color)].push_back(dist);
      if (range_histories_[static_cast<size_t>(color)].size() > range_window_size_)
        range_histories_[static_cast<size_t>(color)].pop_front();
    }

    double red_trend = rangeTrend(ClusterColor::RED);
    double blue_trend = rangeTrend(ClusterColor::BLUE);

    if (red_trend < -approach_threshold_)
    {
      moving_cluster_ = ClusterColor::RED;
    }
    else if (blue_trend > approach_threshold_)
    {
      moving_cluster_ = ClusterColor::BLUE;
    }
  }

  bool isCenterWithinBounds(ClusterColor color) const
  {
    const auto &history = center_histories_[static_cast<size_t>(color)];
    if (history.size() < center_window_size_)
      return true;

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    for (const auto &p : history)
    {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }

    double span_x = max_x - min_x;
    double span_y = max_y - min_y;
    if (span_x > center_box_limit_ || span_y > center_box_limit_)
    {
      //RCLCPP_WARN(
      //  this->get_logger(),
      //  "Center span exceeded (span_x=%.3f, span_y=%.3f); dropping visualization",
      //  span_x, span_y);
      return false;
    }
    return true;
  }

  bool centersTooClose() const
  {
    const auto &red = centers_[static_cast<size_t>(ClusterColor::RED)];
    const auto &blue = centers_[static_cast<size_t>(ClusterColor::BLUE)];
    if (!red.valid || !blue.valid)
      return false;
    double dist = std::hypot(red.current.x - blue.current.x, red.current.y - blue.current.y);
    if (dist < center_proximity_threshold_)
    {
      //RCLCPP_INFO(
      //  this->get_logger(),
      //  "Centers too close (%.3fm); suppressing marker",
      //  dist);
      return true;
    }
    return false;
  }

  double rangeTrend(ClusterColor color) const
  {
    const auto &hist = range_histories_[static_cast<size_t>(color)];
    if (hist.size() < 2)
      return 0.0;
    return hist.back() - hist.front();
  }

  bool acceptTrajectorySample(ClusterColor color, const geometry_msgs::msg::Point &p)
  {
    size_t idx = static_cast<size_t>(color);
    auto &history = trajectory_histories_[idx];
    auto &far_count = trajectory_far_counts_[idx];

    if (history.empty())
    {
      pushTrajectory(history, p);
      far_count = 0;
      return true;
    }

    double min_dist = std::numeric_limits<double>::max();
    for (const auto &past : history)
      min_dist = std::min(min_dist, std::hypot(p.x - past.x, p.y - past.y));

    if (min_dist <= trajectory_far_threshold_)
    {
      pushTrajectory(history, p);
      far_count = 0;
      return true;
    }

    ++far_count;
    if (far_count >= trajectory_persistence_frames_)
    {
      history.clear();
      pushTrajectory(history, p);
      far_count = 0;
      return true;
    }

    return false;
  }

  void pushTrajectory(std::deque<geometry_msgs::msg::Point> &history,
                      const geometry_msgs::msg::Point &p)
  {
    history.push_back(p);
    if (history.size() > trajectory_window_size_)
      history.pop_front();
  }

  std_msgs::msg::ColorRGBA colorFor(ClusterColor color) const
  {
    std_msgs::msg::ColorRGBA c;
    c.a = 1.0f;
    if (color == ClusterColor::RED)
    {
      c.r = 1.0f;
      c.g = 0.0f;
      c.b = 0.0f;
    }
    else
    {
      c.r = 0.0f;
      c.g = 0.0f;
      c.b = 1.0f;
    }
    return c;
  }

  std::string cluster_topic_;
  double movement_threshold_;
  double switch_margin_;
  std::array<std::deque<double>, 2> range_histories_;
  size_t range_window_size_;
  double approach_threshold_;
  size_t center_window_size_;
  double center_box_limit_;
  double center_proximity_threshold_;
  size_t trajectory_window_size_;
  double trajectory_far_threshold_;
  int trajectory_persistence_frames_;

  std::array<std::vector<geometry_msgs::msg::Point>, 2> cluster_points_;
  std::array<CenterState, 2> centers_;
  std::array<std::deque<geometry_msgs::msg::Point>, 2> center_histories_;
  std::array<std::deque<geometry_msgs::msg::Point>, 2> trajectory_histories_;
  std::array<int, 2> trajectory_far_counts_{};
  std::optional<ClusterColor> moving_cluster_;
  std::optional<std_msgs::msg::Header> latest_cluster_header_;

  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr cluster_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr component_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<KNNTrackerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
