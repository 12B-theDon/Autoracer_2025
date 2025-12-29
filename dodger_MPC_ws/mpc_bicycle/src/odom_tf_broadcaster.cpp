// -----------------------------------------------------------------------------
// Odom TF Broadcaster
// This helper node mirrors the pose published on an odometry topic (default
// /ego_racecar/odom) into the TF tree by sending map -> ego_racecar/base_link
// transforms. It allows RViz and other consumers to visualize the robot frame
// even when the odom stream itself does not provide tf.
// -----------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

class OdomTfBroadcaster : public rclcpp::Node {
public:
  OdomTfBroadcaster()
  : rclcpp::Node("odom_tf_broadcaster"),
    tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this))
  {
    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/ego_racecar/odom");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");
    child_frame_id_ = this->declare_parameter<std::string>("child_frame_id", "ego_racecar/base_link");

    auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).best_effort();
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&OdomTfBroadcaster::odomCallback_, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "Broadcasting TF %s -> %s from odom topic: %s",
      frame_id_.c_str(), child_frame_id_.c_str(), odom_topic_.c_str());
  }

private:
  void odomCallback_(const nav_msgs::msg::Odometry::SharedPtr msg) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = frame_id_.empty() ? msg->header.frame_id : frame_id_;
    tf_msg.child_frame_id = child_frame_id_.empty()
                              ? (msg->child_frame_id.empty() ? "ego_racecar/base_link" : msg->child_frame_id)
                              : child_frame_id_;

    tf_msg.transform.translation.x = msg->pose.pose.position.x;
    tf_msg.transform.translation.y = msg->pose.pose.position.y;
    tf_msg.transform.translation.z = msg->pose.pose.position.z;
    tf_msg.transform.rotation = msg->pose.pose.orientation;

    tf_broadcaster_->sendTransform(tf_msg);
  }

  std::string odom_topic_;
  std::string frame_id_;
  std::string child_frame_id_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomTfBroadcaster>());
  rclcpp::shutdown();
  return 0;
}
