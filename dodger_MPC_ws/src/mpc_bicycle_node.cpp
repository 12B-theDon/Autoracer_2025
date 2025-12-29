// -----------------------------------------------------------------------------
// MPC Bicycle Controller
// - Waypoints/CSV coordinates are expressed in the same frame as the odometry
//   topic we subscribe to (currently /ego_racecar/odom, whose frame_id is "map").
//   As long as recording and tracking use the same odom frame, no extra tf
//   transforms are required.
// - The commanded output is AckermannDriveStamped on /drive, not cmd_vel, because
//   this vehicle has steerable front wheels; we must explicitly send steering
//   angles instead of Twist velocities.
// -----------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <numeric>
#include <optional>

using std::placeholders::_1;

struct Waypoint {
  double x;
  double y;
  double v;
};

struct RefGeom {
  double psi_ref; // 경로 탄젠트(rad)
  double ey;      // 경로 좌표계 기준 횡오차(m)
};

class MPCBicycleNode : public rclcpp::Node {
public:
  MPCBicycleNode() : Node("mpc_bicycle_node")
  {
    // ---- Params ----
    wheelbase_       = declare_parameter<double>("wheelbase", 0.33);
    dt_              = declare_parameter<double>("dt", 0.1);
    N_               = declare_parameter<int>("horizon", 10);

    v_ref_param_     = declare_parameter<double>("v_ref", 1.0);
    use_csv_speed_   = declare_parameter<bool>("use_csv_speed", false);
    speed_mode_      = declare_parameter<std::string>("speed_mode", "vref"); // "vref" or "integrate"

    delta_max_deg_   = declare_parameter<double>("delta_max_deg", 25.0);
    a_max_           = declare_parameter<double>("a_max", 2.0);

    q_y_             = declare_parameter<double>("q_y",   3.0);
    q_psi_           = declare_parameter<double>("q_psi", 1.0);
    q_v_             = declare_parameter<double>("q_v",   0.5);
    r_k_             = declare_parameter<double>("r_kappa", 0.1);
    r_a_             = declare_parameter<double>("r_a",     0.01);

    use_curvature_ff_= declare_parameter<bool>("use_curvature_ff", true);
    debug_            = declare_parameter<bool>("debug", false);
    debug_dump_rows_  = declare_parameter<int>("debug_dump_rows", 5);
    debug_lookahead_  = declare_parameter<int>("debug_lookahead", 5);
    trace_flow_       = declare_parameter<bool>("trace_flow", false);

    cmd_topic_        = declare_parameter<std::string>("cmd_topic", "/drive");
    path_csv_         = declare_parameter<std::string>("path_csv", "/dodger_ws/mpc_bicycle/global_path/straight_trajectory_100.csv");

    steering_sign_ = declare_parameter<int>("steering_sign", +1); // +1 or -1
    ey_sign_       = declare_parameter<int>("ey_sign",       +1); // +1 or -1

    // ---- Load path ----
    if (!loadGlobalPath(path_csv_)) {
      RCLCPP_FATAL(get_logger(), "Failed to load global path CSV: %s", path_csv_.c_str());
      rclcpp::shutdown();
      return;
    }
    summarizePathOnce_();

    // ---- Limits ----
    const double delta_max = delta_max_deg_ * M_PI / 180.0;
    kappa_max_ = std::tan(delta_max) / wheelbase_;

    RCLCPP_INFO(get_logger(),
      "Params: L=%.3f, dt=%.3f, N=%d, v_ref=%.2f, use_csv_speed=%s, speed_mode=%s, cmd_topic=%s",
      wheelbase_, dt_, N_, v_ref_param_, use_csv_speed_ ? "true" : "false",
      speed_mode_.c_str(), cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(),
      "Limits: delta_max=%.1f deg, kappa_max=%.3f 1/m, a_max=%.2f | Weights: qy=%.2f, qpsi=%.2f, qv=%.2f, rk=%.3f, ra=%.3f",
      delta_max_deg_, kappa_max_, a_max_, q_y_, q_psi_, q_v_, r_k_, r_a_);

    // ---- IO ----
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/ego_racecar/odom", 10, std::bind(&MPCBicycleNode::odomCb, this, _1));

    cmd_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(cmd_topic_, 10);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt_),
      std::bind(&MPCBicycleNode::controlLoop, this));

    RCLCPP_INFO(get_logger(), "MPCBicycleNode initialized.");
  }

private:
  // -------------------------- CSV loader with line index logs --------------------------
  bool loadGlobalPath(const std::string& filepath, bool has_header=true) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
      RCLCPP_ERROR(get_logger(), "Cannot open CSV: %s", filepath.c_str());
      return false;
    }
    std::string line;
    if (has_header && std::getline(ifs, line)) { /* skip header */ }

    std::vector<Waypoint> buf;
    buf.reserve(10000);

    size_t line_num = 0;
    while (std::getline(ifs, line)) {
      line_num++;
      if (line.empty()) continue;

      std::vector<std::string> tokens;
      tokens.reserve(5);
      std::stringstream ss(line);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        tokens.push_back(tok);
      }
      if (tokens.size() < 3) continue;

      Waypoint w;
      try {
        if (tokens.size() >= 5) {
          // Format: timestep,x,y,yaw,velocity
          w.x = std::stod(tokens[1]);
          w.y = std::stod(tokens[2]);
          w.v = std::stod(tokens[4]);
        } else {
          // Format: x,y,v
          w.x = std::stod(tokens[0]);
          w.y = std::stod(tokens[1]);
          w.v = std::stod(tokens[2]);
        }
      } catch (...) {
        RCLCPP_WARN(get_logger(), "Line %zu skipped (parse error): %s", line_num, line.c_str());
        continue;
      }

      buf.push_back(w);
      if (line_num <= 5) {
        RCLCPP_INFO(get_logger(), "CSV[%zu] -> x=%.3f, y=%.3f, v=%.3f", line_num, w.x, w.y, w.v);
      }
    }

    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "CSV has no valid data rows: %s", filepath.c_str());
      return false;
    }

    global_path_.swap(buf);
    RCLCPP_INFO(get_logger(), "CSV loaded: %zu points (last CSV line=%zu)", global_path_.size(), line_num);
    return true;
  }

  void summarizePathOnce_() {
    const size_t n = global_path_.size();
    RCLCPP_INFO(get_logger(), "Path summary: %zu points", n);
    double v_min = std::numeric_limits<double>::infinity();
    double v_max = -std::numeric_limits<double>::infinity();
    double v_sum = 0.0;
    for (const auto &w : global_path_) {
      v_min = std::min(v_min, w.v);
      v_max = std::max(v_max, w.v);
      v_sum += w.v;
    }
    RCLCPP_INFO(get_logger(), "v min=%.3f, max=%.3f, mean=%.3f", v_min, v_max, v_sum / static_cast<double>(n));
    if (use_csv_speed_ && std::abs(v_max) < 1e-6) {
      RCLCPP_WARN(get_logger(), "CSV v are near all zeros while use_csv_speed=true. Vehicle may not move.");
    }
  }

  // -------------------------- Geometry helpers --------------------------
  size_t findClosestIndex(double x, double y) const {
    size_t idx_min = 0;
    double d2_min = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < global_path_.size(); ++i) {
      double dx = x - global_path_[i].x;
      double dy = y - global_path_[i].y;
      double d2 = dx*dx + dy*dy;
      if (d2 < d2_min) { d2_min = d2; idx_min = i; }
    }
    return idx_min;
  }

  RefGeom computeRefGeom(size_t i, double X, double Y) const {
    size_t j = std::min(i + 1, global_path_.size() - 1);
    const auto &Pi = global_path_[i];
    const auto &Pj = global_path_[j];

    double dx = Pj.x - Pi.x;
    double dy = Pj.y - Pi.y;
    double psi_ref = std::atan2(dy, dx);  // 탄젠트 방향

    // 월드 → 경로(psi_ref) 좌표계로 회전
    double ex =  std::cos(psi_ref) * (X - Pi.x) + std::sin(psi_ref) * (Y - Pi.y);
    double ey = -std::sin(psi_ref) * (X - Pi.x) + std::cos(psi_ref) * (Y - Pi.y);

    (void)ex; // 현재 모델에서는 ex 미사용
    return {psi_ref, ey};
  }

  double curvatureFrom3pts(size_t a, size_t b, size_t c) const {
    const auto &A = global_path_[a], &B = global_path_[b], &C = global_path_[c];
    double x1=A.x, y1=A.y, x2=B.x, y2=B.y, x3=C.x, y3=C.y;
    double num = 2.0 * ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1));
    double d12 = std::hypot(x2-x1, y2-y1);
    double d23 = std::hypot(x3-x2, y3-y2);
    double d31 = std::hypot(x1-x3, y1-y3);
    double denom = d12 * d23 * d31;
    if (std::abs(num) < 1e-9 || denom < 1e-9) return 0.0;
    return num / denom; // 부호 있는 곡률
  }

  static double yawFromQuat(const geometry_msgs::msg::Quaternion &q) {
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  // -------------------------- Callbacks --------------------------
  void odomCb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    last_odom_ = *msg;
    have_odom_ = true;

    if (!logged_start_pose_) {
      const auto &p = msg->pose.pose.position;
      const double psi = yawFromQuat(msg->pose.pose.orientation);
      RCLCPP_INFO(get_logger(),
        "Start pose snapshot -> X=%.3f, Y=%.3f, psi=%.3f rad (%.1f deg)",
        p.x, p.y, psi, psi * 180.0 / M_PI);
      logged_start_pose_ = true;
    }

    RCLCPP_INFO_ONCE(get_logger(), "First odom received.");
  }

  void traceStage_(const char* stage) const {
    if (!trace_flow_) return;
    RCLCPP_INFO(get_logger(), "[TRACE] %s", stage);
  }

  void controlLoop() {
    if (!have_odom_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "Waiting for odom...");
      return;
    }

    // --- State ---
    const auto &o = last_odom_;
    double X = o.pose.pose.position.x;
    double Y = o.pose.pose.position.y;
    double psi = yawFromQuat(o.pose.pose.orientation);
    double vx  = o.twist.twist.linear.x;
    traceStage_("state snapshot ready");

    // --- Reference from CSV ---
    size_t idx = findClosestIndex(X, Y);
    const auto &wp = global_path_[idx];
    double v_ref_csv = wp.v;
    double v_ref = use_csv_speed_ ? v_ref_csv : v_ref_param_;
    traceStage_("closest waypoint + v_ref computed");

    // --- Path-frame errors ---
    auto geom = computeRefGeom(idx, X, Y);
    double psi_ref = geom.psi_ref;
    double e_y     = geom.ey;
    double e_psi   = std::atan2(std::sin(psi - psi_ref), std::cos(psi - psi_ref));
    double e_v     = vx - v_ref;

    if (debug_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 1000,
        "idx=%zu/%zu | ref: psi=%.3f rad, v_ref=%.3f (%s) | err: ey=%.3f, epsi=%.3f, ev=%.3f | state: X=%.2f Y=%.2f psi=%.2f v=%.2f",
        idx, global_path_.size(), psi_ref, v_ref, use_csv_speed_ ? "CSV" : "param",
        e_y, e_psi, e_v, X, Y, psi, vx);
      if (debug_lookahead_ > 0) {
        int L = std::min<int>(debug_lookahead_, static_cast<int>(global_path_.size() - idx - 1));
        if (L > 0) {
          std::ostringstream oss;
          oss << "Next v: ";
          for (int k = 1; k <= L; ++k) {
            oss << global_path_[idx + k].v;
            if (k < L) oss << ", ";
          }
          RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 1000, "%s", oss.str().c_str());
        }
      }
    }

    // --- Linear discrete model in path frame ---
    // x = [e_y, e_psi, e_v], u=[kappa, a]
    Eigen::Vector3d x;
    x << e_y, e_psi, e_v;

    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    A(0,1) = dt_ * std::max(0.1, v_ref); // v_ref=0인 경우 수치안정 보완

    Eigen::Matrix<double,3,2> B = Eigen::Matrix<double,3,2>::Zero();
    B(1,0) = dt_ * std::max(0.1, v_ref);
    B(2,1) = dt_;

    double solver_cost = 0.0;
    double grad_norm = 0.0;
    int solver_iters = 0;
    Eigen::Vector2d u_cmd = solveNominalMpc(A, B, x, solver_cost, grad_norm, solver_iters);
    double kappa_cmd = u_cmd(0);
    double a_cmd     = u_cmd(1);
    traceStage_("MPC solver completed");

    if (debug_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 500,
        "MPC solver: cost=%.4f, grad=%.4e, iters=%d, kappa=%.4f, a=%.4f",
        solver_cost, grad_norm, solver_iters, kappa_cmd, a_cmd);
    }

    // --- Optional curvature feedforward ---
    if (use_curvature_ff_) {
      double kappa_ff = 0.0;
      if (idx > 0 && idx + 1 < global_path_.size()) {
        kappa_ff = curvatureFrom3pts(idx - 1, idx, idx + 1);
      }
      double kappa_before_ff = kappa_cmd;
      kappa_cmd += kappa_ff;
      if (debug_) {
        RCLCPP_DEBUG(get_logger(), "kappa: fb=%.4f, ff=%.4f, sum=%.4f", kappa_before_ff, kappa_ff, kappa_cmd);
      }
    }

    // --- Clamp ---
    double kappa_raw = kappa_cmd, a_raw = a_cmd;
    kappa_cmd = std::clamp(kappa_cmd, -kappa_max_, kappa_max_);
    a_cmd     = std::clamp(a_cmd, -a_max_, a_max_);
    traceStage_("control inputs clamped");
    if (kappa_cmd != kappa_raw) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 1500,
        "kappa saturated: raw=%.4f -> %.4f (|kappa|max=%.4f)", kappa_raw, kappa_cmd, kappa_max_);
    }
    if (a_cmd != a_raw) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 1500,
        "accel saturated: raw=%.3f -> %.3f (|a|max=%.3f)", a_raw, a_cmd, a_max_);
    }

    // --- Output mapping ---
    double delta_cmd = std::atan(wheelbase_ * kappa_cmd);

    // speed 명령: "vref"=참조속도 직접, "integrate"=현재속도 적분
    double speed_cmd = 0.0;
    if (speed_mode_ == "integrate") {
      speed_integrator_ = (speed_integrator_.has_value() ? speed_integrator_.value() : vx);
      speed_integrator_ = std::clamp(speed_integrator_.value() + a_cmd * dt_, 0.0, std::max(0.1, v_ref_param_ * 3.0));
      speed_cmd = speed_integrator_.value();
    } else { // "vref" (default & most compatible with many simulators)
      speed_cmd = std::max(0.0, v_ref);
    }

    // --- Publish ---
    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = this->now();
    cmd.drive.steering_angle = delta_cmd;
    cmd.drive.speed          = speed_cmd;
    cmd.drive.acceleration   = a_cmd;
    cmd_pub_->publish(cmd);

    bool steer_saturated = std::abs(kappa_cmd) >= (kappa_max_ - 1e-6);
    if (steer_saturated && !steer_sat_reported_) {
      RCLCPP_WARN(get_logger(),
        "Steer saturation hit (path=%s): idx=%zu raw=%.4f clamped=%.4f | err: ey=%.3f epsi=%.3f ev=%.3f | pose: X=%.2f Y=%.2f psi=%.2f",
        path_csv_.c_str(), idx, kappa_raw, kappa_cmd, e_y, e_psi, e_v, X, Y, psi);
      steer_sat_reported_ = true;
    } else if (!steer_saturated) {
      steer_sat_reported_ = false;
    }
    traceStage_("command published");

    // subscriber check
    auto subs = cmd_pub_->get_subscription_count();
    if (subs == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
        "No subscribers on %s. Commands may be ignored.", cmd_topic_.c_str());
    }

    if (debug_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 1000,
        "cmd: steer=%.3f rad (%.1f deg), speed=%.3f, accel=%.3f",
        delta_cmd, delta_cmd * 180.0 / M_PI, speed_cmd, a_cmd);
    }
  }

  Eigen::Vector2d solveNominalMpc(
    const Eigen::Matrix3d &A,
    const Eigen::Matrix<double,3,2> &B,
    const Eigen::Vector3d &x0,
    double &cost_out,
    double &grad_norm_out,
    int &iters_out) const
  {
    const int nx = 3;
    const int nu = 2;
    const int N = std::max(1, N_);

    Eigen::MatrixXd Sx = Eigen::MatrixXd::Zero(nx * N, nx);
    Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(nx * N, nu * N);

    std::vector<Eigen::Matrix3d> A_pows(N + 1, Eigen::Matrix3d::Identity());
    for (int k = 1; k <= N; ++k) {
      A_pows[k] = A * A_pows[k - 1];
    }

    for (int k = 0; k < N; ++k) {
      Sx.block(k * nx, 0, nx, nx) = A_pows[k + 1];
      for (int j = 0; j <= k; ++j) {
        Su.block(k * nx, j * nu, nx, nu) = A_pows[k - j] * B;
      }
    }

    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Q(0,0) = q_y_;
    Q(1,1) = q_psi_;
    Q(2,2) = q_v_;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0,0) = r_k_;
    R(1,1) = r_a_;

    Eigen::MatrixXd Qbar = Eigen::MatrixXd::Zero(nx * N, nx * N);
    for (int k = 0; k < N; ++k) {
      Qbar.block(k * nx, k * nx, nx, nx) = Q;
    }
    // terminal weight (same as Q for simplicity)
    Qbar.block((N - 1) * nx, (N - 1) * nx, nx, nx) = Q;

    Eigen::MatrixXd Rbar = Eigen::MatrixXd::Zero(nu * N, nu * N);
    for (int k = 0; k < N; ++k) {
      Rbar.block(k * nu, k * nu, nu, nu) = R;
    }

    Eigen::MatrixXd H = Su.transpose() * Qbar * Su + Rbar;
    Eigen::VectorXd f = Su.transpose() * Qbar * (Sx * x0);

    Eigen::VectorXd u = Eigen::VectorXd::Zero(nu * N);
    Eigen::VectorXd u_min = Eigen::VectorXd::Zero(nu * N);
    Eigen::VectorXd u_max = Eigen::VectorXd::Zero(nu * N);
    for (int k = 0; k < N; ++k) {
      u_min(k * nu + 0) = -kappa_max_;
      u_max(k * nu + 0) =  kappa_max_;
      u_min(k * nu + 1) = -a_max_;
      u_max(k * nu + 1) =  a_max_;
    }

    double diag_max = H.diagonal().maxCoeff();
    if (diag_max < 1e-6) {
      diag_max = 1.0;
    }
    const double step = 1.0 / diag_max;
    const int max_iters = 60;
    const double grad_tol = 1e-4;

    grad_norm_out = 0.0;
    Eigen::VectorXd grad(u.size());
    for (iters_out = 0; iters_out < max_iters; ++iters_out) {
      grad = H * u + f;
      grad_norm_out = grad.norm();
      if (grad_norm_out < grad_tol) {
        break;
      }
      u -= step * grad;
      for (int i = 0; i < u.size(); ++i) {
        u[i] = std::clamp(u[i], u_min[i], u_max[i]);
      }
    }
    grad = H * u + f;
    grad_norm_out = grad.norm();
    const double quad_term = (0.5 * u.transpose() * H * u).value();
    const double lin_term  = (f.transpose() * u).value();
    cost_out = quad_term + lin_term;

    Eigen::Vector2d u0;
    u0 << u(0), u(1);
    u0(0) = std::clamp(u0(0), -kappa_max_, kappa_max_);
    u0(1) = std::clamp(u0(1), -a_max_, a_max_);

    if (!std::isfinite(cost_out) || !std::isfinite(u0.norm())) {
      cost_out = 0.0;
      u0.setZero();
    }
    return u0;
  }

  // -------------------------- Members --------------------------
  std::vector<Waypoint> global_path_;
  std::string path_csv_;
  std::string cmd_topic_;

  // Params
  double wheelbase_{0.33}, dt_{0.1}; int N_{10};
  double v_ref_param_{1.0};
  bool   use_csv_speed_{false};
  std::string speed_mode_{"vref"}; // "vref" or "integrate"
  double delta_max_deg_{25.0}, a_max_{2.0};
  double q_y_{3.0}, q_psi_{1.0}, q_v_{0.5};
  double r_k_{0.1}, r_a_{0.01};
  bool   use_curvature_ff_{true};
  bool   debug_{false};
  int    debug_dump_rows_{5}, debug_lookahead_{5};
  int    steering_sign_{+1}, ey_sign_{+1};
  double kappa_max_{0.0};
  bool   trace_flow_{false};

  // IO & state
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  nav_msgs::msg::Odometry last_odom_;
  bool have_odom_{false};
  bool logged_start_pose_{false};
  bool steer_sat_reported_{false};

  // speed integrator (optional mode)
  std::optional<double> speed_integrator_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCBicycleNode>());
  rclcpp::shutdown();
  return 0;
}
