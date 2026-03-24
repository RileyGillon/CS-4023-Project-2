#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "irobot_create_msgs/msg/hazard_detection.hpp"
#include "irobot_create_msgs/msg/hazard_detection_vector.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "project2_reactive/behaviors.hpp"

namespace project2_reactive
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

}  // namespace

class ReactiveController : public rclcpp::Node
{
public:
  ReactiveController()
  : Node("reactive_controller"), rng_(std::random_device{}())
  {
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ReactiveController::scan_callback, this, std::placeholders::_1));

    hazard_sub_ = create_subscription<irobot_create_msgs::msg::HazardDetectionVector>(
      "/hazard_detection", 10,
      std::bind(&ReactiveController::hazard_callback, this, std::placeholders::_1));

    key_vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/key_vel", 10,
      std::bind(&ReactiveController::key_vel_callback, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&ReactiveController::odom_callback, this, std::placeholders::_1));

    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&ReactiveController::control_loop, this));

    RCLCPP_INFO(get_logger(), "ReactiveController initialized");
  }

private:
  geometry_msgs::msg::TwistStamped make_twist(double linear_x, double angular_z) const
  {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = "base_link";
    cmd.twist.linear.x = linear_x;
    cmd.twist.angular.z = angular_z;
    return cmd;
  }

  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    scan_ = msg;

    auto front = behaviors::get_front_distances(
      msg->ranges,
      msg->angle_min,
      msg->angle_increment,
      msg->range_min,
      msg->range_max);

    front_left_ = std::move(front.left);
    front_right_ = std::move(front.right);

    laser_collision_ = false;
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
      const double r = static_cast<double>(msg->ranges[i]);
      if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) {
        continue;
      }
      const double raw = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
      double angle = std::fmod(raw + kPi, 2.0 * kPi);
      if (angle < 0.0) {
        angle += 2.0 * kPi;
      }
      angle -= kPi;
      if (std::abs(angle) < behaviors::LASER_HALT_SECTOR_RAD &&
        r < behaviors::LASER_HALT_DISTANCE_M)
      {
        laser_collision_ = true;
        break;
      }
    }
  }

  void hazard_callback(const irobot_create_msgs::msg::HazardDetectionVector::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    collision_detected_ = std::any_of(
      msg->detections.begin(), msg->detections.end(),
      [](const auto & d) {
        return d.type == irobot_create_msgs::msg::HazardDetection::BUMP;
      });
  }

  void key_vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    keyboard_cmd_ = msg;
    last_key_time_ = std::chrono::steady_clock::now();
    has_key_time_ = true;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;

    current_yaw_ = behaviors::yaw_from_quaternion(q.x, q.y, q.z, q.w);

    if (has_last_odom_) {
      const double dx = x - last_odom_x_;
      const double dy = y - last_odom_y_;
      const double fwd = behaviors::forward_displacement(dx, dy, current_yaw_);
      if (fwd > 0.0) {
        distance_since_turn_ += fwd;
      }
    }

    last_odom_x_ = x;
    last_odom_y_ = y;
    has_last_odom_ = true;
  }

  std::optional<geometry_msgs::msg::TwistStamped> halt_behavior()
  {
    if (collision_detected_ || laser_collision_) {
      return make_twist(0.0, 0.0);
    }
    return std::nullopt;
  }

  std::optional<geometry_msgs::msg::TwistStamped> keyboard_behavior()
  {
    if (!keyboard_cmd_ || !has_key_time_) {
      return std::nullopt;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::steady_clock::now() - last_key_time_).count();
    if (elapsed > behaviors::KEYBOARD_TIMEOUT_S) {
      return std::nullopt;
    }

    const auto & t = keyboard_cmd_->twist;
    if (std::abs(t.linear.x) > 1e-3 ||
      std::abs(t.linear.y) > 1e-3 ||
      std::abs(t.angular.z) > 1e-3)
    {
      return *keyboard_cmd_;
    }

    return std::nullopt;
  }

  std::optional<geometry_msgs::msg::TwistStamped> escape_behavior()
  {
    if (!scan_) {
      return std::nullopt;
    }

    const auto now_steady = std::chrono::steady_clock::now();

    if (escaping_) {
      if (!behaviors::obstacle_in_range(front_left_, front_right_)) {
        escaping_ = false;
        escape_cooldown_until_ = now_steady +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(behaviors::ESCAPE_COOLDOWN_S));
        return std::nullopt;
      }

      return make_twist(
        behaviors::ESCAPE_BACKUP_SPEED_MPS,
        escape_turn_sign_ * behaviors::TURN_SPEED_RADS);
    }

    if (now_steady < escape_cooldown_until_) {
      return std::nullopt;
    }

    if (behaviors::is_symmetric_obstacle(front_left_, front_right_)) {
      escaping_ = true;
      std::uniform_real_distribution<double> coin(0.0, 1.0);
      escape_turn_sign_ = (coin(rng_) > 0.5) ? 1.0 : -1.0;
      return make_twist(
        behaviors::ESCAPE_BACKUP_SPEED_MPS,
        escape_turn_sign_ * behaviors::TURN_SPEED_RADS);
    }

    return std::nullopt;
  }

  std::optional<geometry_msgs::msg::TwistStamped> avoid_behavior() const
  {
    if (!scan_) {
      return std::nullopt;
    }

    if (!behaviors::obstacle_in_range(front_left_, front_right_)) {
      return std::nullopt;
    }

    if (behaviors::is_symmetric_obstacle(front_left_, front_right_)) {
      return std::nullopt;
    }

    const double angular_z = behaviors::get_avoid_direction(front_left_, front_right_);
    return make_twist(0.0, angular_z);
  }

  std::optional<geometry_msgs::msg::TwistStamped> turn_behavior()
  {
    const auto now_steady = std::chrono::steady_clock::now();

    if (turning_) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        now_steady - turn_start_time_).count();
      if (elapsed < turn_duration_s_) {
        return make_twist(0.0, turn_sign_ * behaviors::TURN_SPEED_RADS);
      }
      turning_ = false;
    }

    if (distance_since_turn_ >= behaviors::TURN_DISTANCE_M) {
      const double angle = behaviors::sample_random_turn_angle(rng_);
      turn_sign_ = (angle >= 0.0) ? 1.0 : -1.0;
      turn_duration_s_ = std::abs(angle) / behaviors::TURN_SPEED_RADS;
      turn_start_time_ = now_steady;
      distance_since_turn_ = 0.0;
      turning_ = true;
      return make_twist(0.0, turn_sign_ * behaviors::TURN_SPEED_RADS);
    }

    return std::nullopt;
  }

  geometry_msgs::msg::TwistStamped drive_behavior() const
  {
    return make_twist(behaviors::FORWARD_SPEED_MPS, 0.0);
  }

  void control_loop()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (auto cmd = halt_behavior()) {
      cmd_vel_pub_->publish(*cmd);
      return;
    }
    if (auto cmd = keyboard_behavior()) {
      cmd_vel_pub_->publish(*cmd);
      return;
    }
    if (auto cmd = escape_behavior()) {
      cmd_vel_pub_->publish(*cmd);
      return;
    }
    if (auto cmd = avoid_behavior()) {
      cmd_vel_pub_->publish(*cmd);
      return;
    }
    if (auto cmd = turn_behavior()) {
      cmd_vel_pub_->publish(*cmd);
      return;
    }

    cmd_vel_pub_->publish(drive_behavior());
  }

  std::mutex mutex_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<irobot_create_msgs::msg::HazardDetectionVector>::SharedPtr hazard_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr key_vel_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  sensor_msgs::msg::LaserScan::SharedPtr scan_;
  std::vector<double> front_left_;
  std::vector<double> front_right_;

  bool collision_detected_{false};
  bool laser_collision_{false};

  geometry_msgs::msg::TwistStamped::SharedPtr keyboard_cmd_;
  std::chrono::steady_clock::time_point last_key_time_{};
  bool has_key_time_{false};

  double last_odom_x_{0.0};
  double last_odom_y_{0.0};
  bool has_last_odom_{false};
  double current_yaw_{0.0};
  double distance_since_turn_{0.0};

  bool turning_{false};
  double turn_sign_{1.0};
  std::chrono::steady_clock::time_point turn_start_time_{};
  double turn_duration_s_{0.0};

  bool escaping_{false};
  double escape_turn_sign_{1.0};
  std::chrono::steady_clock::time_point escape_cooldown_until_{};

  std::mt19937 rng_;
};

}  // namespace project2_reactive

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<project2_reactive::ReactiveController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
