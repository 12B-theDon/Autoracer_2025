#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <vector>
#include <deque>
#include <iostream>
#include <utility>
#include <iomanip>

struct Point2D
{
  double x;
  double y;
};

class ScanProcessor : public rclcpp::Node
{
public:
  ScanProcessor()
  : Node("scan_processor"),
    window_size_(20),
    merge_threshold_(0.18),
    speed_window_size_(10),
    speed_swap_margin_(0.02),
    swap_cooldown_(0),
    swap_cooldown_max_(15)
  {
    this->declare_parameter<double>("scan_range_min", 0.0);
    this->declare_parameter<double>("scan_range_max", 10.0);
    this->declare_parameter<double>("scan_angle_min", -M_PI / 3.0);
    this->declare_parameter<double>("scan_angle_max",  M_PI / 3.0);

    this->get_parameter("scan_range_min",  scan_range_min_);
    this->get_parameter("scan_range_max",  scan_range_max_);
    this->get_parameter("scan_angle_min",  scan_angle_min_);
    this->get_parameter("scan_angle_max",  scan_angle_max_);

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ScanProcessor::scanCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "cluster_centers", 10);
  }

private:
  double dist(const Point2D &a, const Point2D &b) const
  {
    return std::hypot(a.x - b.x, a.y - b.y);
  }

  Point2D add(const Point2D &a, const Point2D &b) const
  {
    return {a.x + b.x, a.y + b.y};
  }

  Point2D sub(const Point2D &a, const Point2D &b) const
  {
    return {a.x - b.x, a.y - b.y};
  }

  std::pair<Point2D, Point2D> kmeans2(const std::vector<Point2D> &pts) const
  {
    Point2D c1 = pts.front();
    Point2D c2 = pts.back();

    const int max_iter = 10;
    std::vector<int> labels(pts.size(), 0);

    for (int iter = 0; iter < max_iter; ++iter)
    {
      for (size_t i = 0; i < pts.size(); ++i)
      {
        double d1 = dist(pts[i], c1);
        double d2 = dist(pts[i], c2);
        labels[i] = (d1 <= d2) ? 0 : 1;
      }

      Point2D new_c1{0.0, 0.0}, new_c2{0.0, 0.0};
      int count1 = 0, count2 = 0;

      for (size_t i = 0; i < pts.size(); ++i)
      {
        if (labels[i] == 0)
        {
          new_c1.x += pts[i].x;
          new_c1.y += pts[i].y;
          ++count1;
        }
        else
        {
          new_c2.x += pts[i].x;
          new_c2.y += pts[i].y;
          ++count2;
        }
      }

      if (count1 > 0)
      {
        new_c1.x /= static_cast<double>(count1);
        new_c1.y /= static_cast<double>(count1);
      }
      if (count2 > 0)
      {
        new_c2.x /= static_cast<double>(count2);
        new_c2.y /= static_cast<double>(count2);
      }

      double move1 = dist(c1, new_c1);
      double move2 = dist(c2, new_c2);
      c1 = new_c1;
      c2 = new_c2;
      if (move1 < 1e-4 && move2 < 1e-4)
        break;
    }

    return {c1, c2};
  }

  Point2D pushAndSmooth(std::deque<Point2D> &hist, const Point2D &p)
  {
    hist.push_back(p);
    if (hist.size() > window_size_)
      hist.pop_front();

    double sx = 0.0, sy = 0.0;
    for (const auto &h : hist)
    {
      sx += h.x;
      sy += h.y;
    }
    double n = static_cast<double>(hist.size());
    return {sx / n, sy / n};
  }

  double avgSpeed(const std::deque<double> &hist) const
  {
    if (hist.empty())
      return 0.0;
    double s = 0.0;
    for (double v : hist)
      s += v;
    return s / static_cast<double>(hist.size());
  }

  void associateClusters(const Point2D &c1,
                         const Point2D &c2,
                         Point2D &outA,
                         Point2D &outB)
  {
    if (hist_A_.empty() || hist_B_.empty())
    {
      outA = c1;
      outB = c2;
      return;
    }

    if (hist_A_.size() < 5 || hist_B_.size() < 5)
    {
      Point2D lastA = hist_A_.back();
      Point2D lastB = hist_B_.back();

      double cost1 = dist(c1, lastA) + dist(c2, lastB);
      double cost2 = dist(c2, lastA) + dist(c1, lastB);

      if (cost1 <= cost2)
      {
        outA = c1;
        outB = c2;
      }
      else
      {
        outA = c2;
        outB = c1;
      }
      return;
    }

    Point2D prevA = hist_A_[hist_A_.size() - 2];
    Point2D lastA = hist_A_.back();
    Point2D prevB = hist_B_[hist_B_.size() - 2];
    Point2D lastB = hist_B_.back();

    Point2D vA_prev = sub(lastA, prevA);
    Point2D vB_prev = sub(lastB, prevB);

    Point2D predA = add(lastA, vA_prev);
    Point2D predB = add(lastB, vB_prev);

    double cost1 = dist(c1, predA) + dist(c2, predB);
    double cost2 = dist(c2, predA) + dist(c1, predB);

    if (cost1 <= cost2)
    {
      outA = c1;
      outB = c2;
    }
    else
    {
      outA = c2;
      outB = c1;
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
  {
    std::vector<Point2D> points;
    points.reserve(scan_msg->ranges.size());

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
    {
      double r = scan_msg->ranges[i];
      if (std::isnan(r) || std::isinf(r))
        continue;

      double angle = scan_msg->angle_min + static_cast<double>(i) * scan_msg->angle_increment;

      if (r < scan_range_min_ || r > scan_range_max_ ||
          angle < scan_angle_min_ || angle > scan_angle_max_)
      {
        continue;
      }

      Point2D p{r * std::cos(angle), r * std::sin(angle)};
      points.push_back(p);
    }

    if (points.size() < 2)
      return;

    auto centers = kmeans2(points);
    Point2D c1 = centers.first;
    Point2D c2 = centers.second;

    Point2D rawA, rawB;
    associateClusters(c1, c2, rawA, rawB);

    Point2D smoothA = pushAndSmooth(hist_A_, rawA);
    Point2D smoothB = pushAndSmooth(hist_B_, rawB);

    bool swapped_by_velocity = false;
    if (hist_A_.size() >= 2 && hist_B_.size() >= 2)
    {
      Point2D lastA_prev = hist_A_[hist_A_.size() - 2];
      Point2D lastA      = hist_A_.back();
      Point2D lastB_prev = hist_B_[hist_B_.size() - 2];
      Point2D lastB      = hist_B_.back();

      Point2D vA_raw = sub(lastA, lastA_prev);
      Point2D vB_raw = sub(lastB, lastB_prev);

      double speedA = std::hypot(vA_raw.x, vA_raw.y);
      double speedB = std::hypot(vB_raw.x, vB_raw.y);

      speed_hist_A_.push_back(speedA);
      if (speed_hist_A_.size() > speed_window_size_)
        speed_hist_A_.pop_front();

      speed_hist_B_.push_back(speedB);
      if (speed_hist_B_.size() > speed_window_size_)
        speed_hist_B_.pop_front();

      double avgA = avgSpeed(speed_hist_A_);
      double avgB = avgSpeed(speed_hist_B_);

      if (swap_cooldown_ > 0)
        --swap_cooldown_;

      if (avgB > avgA + speed_swap_margin_ && swap_cooldown_ == 0)
      {
        std::swap(hist_A_, hist_B_);
        std::swap(speed_hist_A_, speed_hist_B_);
        std::swap(smoothA, smoothB);
        swap_cooldown_ = swap_cooldown_max_;
        swapped_by_velocity = true;
      }
    }

    if (swapped_by_velocity)
      std::swap(rawA, rawB);

    visualization_msgs::msg::Marker cluster_marker;
    cluster_marker.header.frame_id = scan_msg->header.frame_id;
    cluster_marker.header.stamp = scan_msg->header.stamp;
    cluster_marker.ns = "clusters";
    cluster_marker.id = 0;
    cluster_marker.type = visualization_msgs::msg::Marker::POINTS;
    cluster_marker.action = visualization_msgs::msg::Marker::ADD;
    cluster_marker.scale.x = 0.03;
    cluster_marker.scale.y = 0.03;
    cluster_marker.color.a = 1.0;

    std_msgs::msg::ColorRGBA colorA;
    colorA.r = 1.0f; colorA.g = 0.0f; colorA.b = 0.0f; colorA.a = 1.0f;
    std_msgs::msg::ColorRGBA colorB;
    colorB.r = 0.0f; colorB.g = 0.0f; colorB.b = 1.0f; colorB.a = 1.0f;
    std_msgs::msg::ColorRGBA colorArc;
    colorArc.r = 0.0f; colorArc.g = 1.0f; colorArc.b = 1.0f; colorArc.a = 1.0f;

    visualization_msgs::msg::Marker arc_marker;
    arc_marker.header = cluster_marker.header;
    arc_marker.ns = "arc_candidates";
    arc_marker.id = 2;
    arc_marker.type = visualization_msgs::msg::Marker::POINTS;
    arc_marker.action = visualization_msgs::msg::Marker::ADD;
    arc_marker.scale.x = 0.035;
    arc_marker.scale.y = 0.035;
    arc_marker.color.a = 1.0;

    auto isArcPoint = [&](const Point2D &p, const Point2D &center, double threshold) {
      double dx = p.x - center.x;
      double dy = p.y - center.y;
      double r = std::hypot(center.x, center.y);
      if (r < 1e-3)
        return false;
      double dist_line = std::fabs(dx * (-center.y) + dy * center.x) / r;
      return dist_line > threshold;
    };

    for (const auto &pt : points)
    {
      double dA = dist(pt, rawA);
      double dB = dist(pt, rawB);
      geometry_msgs::msg::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = 0.0;
      bool belongs_to_A = (dA <= dB);
      cluster_marker.points.push_back(p);
      cluster_marker.colors.push_back(belongs_to_A ? colorA : colorB);

      Point2D center_ref = belongs_to_A ? rawA : rawB;
      if (isArcPoint(pt, center_ref, 0.03))
      {
        arc_marker.points.push_back(p);
        arc_marker.colors.push_back(colorArc);
      }
    }

    marker_pub_->publish(cluster_marker);
    if (!arc_marker.points.empty())
      marker_pub_->publish(arc_marker);

    double dAB = dist(smoothA, smoothB);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = scan_msg->header.frame_id;
    marker.header.stamp = scan_msg->header.stamp;
    marker.ns = "centers";
    marker.id = 1;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.15;
    marker.scale.y = 0.15;
    marker.color.a = 1.0;

    geometry_msgs::msg::Point pA, pB;
    pA.x = smoothA.x;
    pA.y = smoothA.y;
    pA.z = 0.0;
    pB.x = smoothB.x;
    pB.y = smoothB.y;
    pB.z = 0.0;

    std::cout << std::fixed << std::setprecision(3);

    if (dAB < merge_threshold_)
    {
      marker.points.push_back(pA);
      marker.colors.push_back(colorA);
    }
    else
    {
      marker.points.push_back(pA);
      marker.colors.push_back(colorA);
      marker.points.push_back(pB);
      marker.colors.push_back(colorB);
    }

    marker_pub_->publish(marker);
  }

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  std::deque<Point2D> hist_A_;
  std::deque<Point2D> hist_B_;

  const std::size_t window_size_;
  const double merge_threshold_;

  std::deque<double> speed_hist_A_;
  std::deque<double> speed_hist_B_;
  const std::size_t speed_window_size_;
  const double speed_swap_margin_;
  int swap_cooldown_;
  const int swap_cooldown_max_;

  double scan_range_min_;
  double scan_range_max_;
  double scan_angle_min_;
  double scan_angle_max_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
