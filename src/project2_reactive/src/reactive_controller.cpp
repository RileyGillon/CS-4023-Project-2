/**
 * reactive_controller.cpp
 *
 * Reactive robot controller for TurtleBot 4 (Project 2).
 * Implements a pseudo-subsumption architecture with the following
 * behaviors in priority order (highest to lowest):
 *
 *   1. Halt     — stop if bumper collision or laser proximity detected
 *   2. Keyboard — pass through human teleop commands from /key_vel
 *   3. Escape   — turn ~180° away from symmetric close obstacles (fixed action pattern)
 *   4. Avoid    — reflexively steer away from asymmetric close obstacles
 *   5. Turn     — random ±15° turn after every 1ft of forward travel
 *   6. Drive    — drive forward at constant speed
 *
 * Priority arbitration is implemented as a top-down if/return chain in
 * control_loop(). Each behavior returns std::optional<TwistStamped>:
 * std::nullopt means "not active", a value means "take control". The
 * first behavior that returns a value wins and suppresses all lower ones.
 *
 * Halt is bypassed during escape and the post-escape forward push so the
 * robot can complete its 180° turn and clear the wall uninterrupted.
 */

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

/**
 * Wraps an angle (radians) into the range [-pi, pi].
 * Used when computing the angular error for yaw-tracking in escape_behavior.
 */
double normalize_angle(double a)
{
  while (a >  kPi) { a -= 2.0 * kPi; }
  while (a < -kPi) { a += 2.0 * kPi; }
  return a;
}

}  // namespace

class ReactiveController : public rclcpp::Node
{
public:
  ReactiveController()
  : Node("reactive_controller"), rng_(std::random_device{}())
  {
    // Publishes stamped velocity commands to the TurtleBot 4 drive system.
    // TwistStamped (not plain Twist) is required by the physical robot's
    // Create 3 base firmware.
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);

    // Laser scanner — used for obstacle detection (escape/avoid) and the
    // laser-proximity fallback for the halt behavior.
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&ReactiveController::scan_callback, this, std::placeholders::_1));

    // Hazard detection — used for bumper-based halt. The Create 3 base
    // publishes BUMP events on this topic when the physical bumpers are hit.
    hazard_sub_ = create_subscription<irobot_create_msgs::msg::HazardDetectionVector>(
      "/hazard_detection", 10,
      std::bind(&ReactiveController::hazard_callback, this, std::placeholders::_1));

    // Keyboard teleop input — published by teleop_twist_keyboard remapped to
    // /key_vel so it does not directly drive /cmd_vel, allowing the controller
    // to arbitrate its priority against autonomous behaviors.
    key_vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/key_vel", 10,
      std::bind(&ReactiveController::key_vel_callback, this, std::placeholders::_1));

    // Odometry — provides current heading (yaw) for escape yaw-tracking and
    // forward displacement for the 1ft turn-trigger accumulator.
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&ReactiveController::odom_callback, this, std::placeholders::_1));

    // Control loop at 10 Hz. All behavior arbitration happens here.
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&ReactiveController::control_loop, this));

    RCLCPP_INFO(get_logger(), "ReactiveController initialized");
  }

private:
  /**
   * Builds a TwistStamped message with the given linear and angular velocities.
   * All behaviors use this helper so the header stamp and frame_id are always
   * set consistently.
   */
  geometry_msgs::msg::TwistStamped make_twist(double linear_x, double angular_z) const
  {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = "base_link";
    cmd.twist.linear.x = linear_x;
    cmd.twist.angular.z = angular_z;
    return cmd;
  }

  /**
   * Laser scan callback — called every time a new /scan message arrives.
   *
   * Does two things:
   *   1. Partitions beams within the forward ±30° sector into front_left_ and
   *      front_right_ distance vectors. These are consumed by escape_behavior
   *      and avoid_behavior to detect and classify obstacles.
   *   2. Sets laser_collision_ if any beam within the forward ±45° wedge is
   *      closer than LASER_HALT_DISTANCE_M. This provides a laser-proximity
   *      fallback for halt, complementing the bumper-based collision_detected_.
   *
   * A mutex is held for the entire callback because the timer-driven
   * control_loop reads the same shared state.
   */
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    scan_ = msg;

    // Populate left/right front distance vectors for escape and avoid
    auto front = behaviors::get_front_distances(
      msg->ranges,
      msg->angle_min,
      msg->angle_increment,
      msg->range_min,
      msg->range_max);

    front_left_ = std::move(front.left);
    front_right_ = std::move(front.right);

    // Check for a laser-proximity collision in the narrow forward wedge
    laser_collision_ = false;
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
      const double r = static_cast<double>(msg->ranges[i]);
      if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) {
        continue;
      }
      const double raw = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
      const double angle = normalize_angle(raw);
      if (std::abs(angle) < behaviors::LASER_HALT_SECTOR_RAD &&
        r < behaviors::LASER_HALT_DISTANCE_M)
      {
        laser_collision_ = true;
        break;
      }
    }
  }

  /**
   * Hazard detection callback — called when the Create 3 base reports a hazard.
   *
   * Sets collision_detected_ if any detection in the vector is a BUMP event,
   * meaning the physical bumper ring made contact with an obstacle. This is the
   * primary trigger for the Halt behavior as specified in the assignment.
   * laser_collision_ serves as a secondary proximity-based fallback.
   */
  void hazard_callback(const irobot_create_msgs::msg::HazardDetectionVector::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    collision_detected_ = std::any_of(
      msg->detections.begin(), msg->detections.end(),
      [](const auto & d) {
        return d.type == irobot_create_msgs::msg::HazardDetection::BUMP;
      });
  }

  /**
   * Keyboard velocity callback — stores the latest teleop command and records
   * the time it arrived. keyboard_behavior uses the timestamp to expire commands
   * after KEYBOARD_TIMEOUT_S seconds, preventing stale inputs from keeping the
   * keyboard behavior active when the user has stopped typing.
   */
  void key_vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    keyboard_cmd_ = msg;
    last_key_time_ = std::chrono::steady_clock::now();
    has_key_time_ = true;
  }

  /**
   * Odometry callback — extracts yaw and accumulates forward travel distance.
   *
   * Yaw (current_yaw_) is converted from the quaternion orientation in the
   * odometry message and used by escape_behavior for yaw-tracking.
   *
   * Forward displacement is computed by projecting the raw (dx, dy) step onto
   * the robot's current heading direction using a dot product. Only positive
   * (forward) displacement is accumulated so backing up during escape does not
   * count toward the 1ft turn trigger.
   */
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;

    // Extract yaw from quaternion for escape heading control
    current_yaw_ = behaviors::yaw_from_quaternion(q.x, q.y, q.z, q.w);

    if (has_last_odom_) {
      const double dx = x - last_odom_x_;
      const double dy = y - last_odom_y_;
      // Project displacement onto heading — only count forward motion
      const double fwd = behaviors::forward_displacement(dx, dy, current_yaw_);
      if (fwd > 0.0) {
        distance_since_turn_ += fwd;
      }
    }

    last_odom_x_ = x;
    last_odom_y_ = y;
    has_last_odom_ = true;
  }

  /**
   * Priority 1 — Halt if collision detected by bumper or laser proximity.
   *
   * Fires when either the physical bumper (collision_detected_) or the
   * laser proximity check (laser_collision_) is active. Publishes a zero
   * velocity to stop the robot immediately.
   *
   * Bypassed while escaping_ is true so the robot can complete its 180°
   * turn in place without interruption, and during post_escape_ticks_ so
   * the robot can drive away from the wall after the turn finishes.
   */
  std::optional<geometry_msgs::msg::TwistStamped> halt_behavior()
  {
    if (escaping_ || post_escape_ticks_ > 0) {
      return std::nullopt;
    }
    if (collision_detected_ || laser_collision_) {
      return make_twist(0.0, 0.0);
    }
    return std::nullopt;
  }

  /**
   * Priority 2 — Accept keyboard movement commands from a human user.
   *
   * Passes the most recent /key_vel command straight through to /cmd_vel
   * as long as it arrived within KEYBOARD_TIMEOUT_S seconds and contains
   * a nonzero velocity. The timeout prevents a single keypress from keeping
   * the keyboard behavior active indefinitely when the user stops typing.
   */
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

  /**
   * Priority 3 — Escape from symmetric obstacles within 1ft in front of the robot.
   *
   * This is a FIXED ACTION PATTERN (not a reflex): once triggered, the robot
   * turns until it actually reaches the target heading via odometry feedback,
   * regardless of whether the obstacle is still detected mid-turn. This is
   * more reliable than a timer on a physical robot where wheel slip or wall
   * contact can cause the turn to fall short.
   *
   * On trigger:
   *   - A target yaw 180° ± 30° away from the current heading is sampled.
   *   - Turn direction (left/right) is chosen randomly.
   *   - The robot turns at ESCAPE_TURN_SPEED_RADS until the yaw error drops
   *     within ESCAPE_TOLERANCE_RAD (±30°).
   *
   * After the turn completes, a POST_ESCAPE_TICKS forward push drives the
   * robot physically away from the wall before a cooldown prevents immediate
   * re-triggering.
   */
  std::optional<geometry_msgs::msg::TwistStamped> escape_behavior()
  {
    if (!scan_) {
      return std::nullopt;
    }

    const auto now_steady = std::chrono::steady_clock::now();

    // Post-escape forward push — drives away from the wall for ~2 seconds
    // before re-enabling normal behaviors. Halt is bypassed during this phase.
    if (post_escape_ticks_ > 0) {
      post_escape_ticks_--;
      if (post_escape_ticks_ == 0) {
        // Begin cooldown once the forward push finishes
        escape_cooldown_until_ = now_steady +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(behaviors::ESCAPE_COOLDOWN_S));
      }
      return make_twist(behaviors::FORWARD_SPEED_MPS, 0.0);
    }

    // Cooldown — prevents immediate re-trigger after a completed escape
    if (now_steady < escape_cooldown_until_) {
      return std::nullopt;
    }

    if (escaping_) {
      // Yaw-tracking: compute angular error to target heading and keep turning
      // until within ±30° of the target. normalize_angle ensures the error is
      // always in [-pi, pi] so the sign correctly indicates turn direction.
      const double err = normalize_angle(escape_target_yaw_ - current_yaw_);

      if (std::abs(err) <= behaviors::ESCAPE_TOLERANCE_RAD) {
        // Turn complete — begin post-escape forward push
        escaping_ = false;
        post_escape_ticks_ = POST_ESCAPE_TICKS;
        RCLCPP_INFO(get_logger(), "Escape turn complete, pushing forward.");
        return make_twist(behaviors::FORWARD_SPEED_MPS, 0.0);
      }

      // Still turning — direction set by sign of remaining angular error
      const double turn_dir = (err > 0.0) ? 1.0 : -1.0;
      return make_twist(0.0, turn_dir * behaviors::ESCAPE_TURN_SPEED_RADS);
    }

    // Trigger: symmetric obstacle detected within 1ft in the forward sector
    if (behaviors::is_symmetric_obstacle(front_left_, front_right_)) {
      escaping_ = true;

      // Sample target rotation uniformly in [150°, 210°] = [pi - pi/6, pi + pi/6]
      std::uniform_real_distribution<double> angle_dist(
        kPi - kPi / 6.0, kPi + kPi / 6.0);
      const double delta = angle_dist(rng_);

      // Randomly choose to turn left (+) or right (-)
      std::uniform_real_distribution<double> coin(0.0, 1.0);
      const double sign = (coin(rng_) > 0.5) ? 1.0 : -1.0;
      escape_target_yaw_ = normalize_angle(current_yaw_ + sign * delta);

      RCLCPP_INFO(
        get_logger(), "Escape triggered. Target yaw: %.2f", escape_target_yaw_);

      return make_twist(0.0, sign * behaviors::ESCAPE_TURN_SPEED_RADS);
    }

    return std::nullopt;
  }

  /**
   * Priority 4 — Avoid asymmetric obstacles within 1ft in front of the robot.
   *
   * This IS a reflex (unlike escape): it fires only while an asymmetric obstacle
   * is present and stops as soon as the obstacle clears. The robot turns in place
   * away from the closer side. Symmetric obstacles are intentionally ignored here
   * because escape_behavior (higher priority) handles them.
   */
  std::optional<geometry_msgs::msg::TwistStamped> avoid_behavior() const
  {
    if (!scan_) {
      return std::nullopt;
    }

    // No obstacle within 1ft — nothing to avoid
    if (!behaviors::obstacle_in_range(front_left_, front_right_)) {
      return std::nullopt;
    }

    // Symmetric obstacles are handled by escape (higher priority)
    if (behaviors::is_symmetric_obstacle(front_left_, front_right_)) {
      return std::nullopt;
    }

    // Turn away from the side with the closer obstacle
    const double angular_z = behaviors::get_avoid_direction(front_left_, front_right_);
    return make_twist(0.0, angular_z);
  }

  /**
   * Priority 5 — Turn randomly (±15°) after every 1ft of forward movement.
   *
   * distance_since_turn_ accumulates forward-projected odometry displacement.
   * When it reaches TURN_DISTANCE_M (1ft = 0.3048m), a random turn angle is
   * sampled uniformly from [-15°, +15°] using std::uniform_real_distribution.
   * The required turn duration is computed as angle / TURN_SPEED_RADS and
   * executed via a timer so the behavior persists across multiple control ticks.
   */
  std::optional<geometry_msgs::msg::TwistStamped> turn_behavior()
  {
    const auto now_steady = std::chrono::steady_clock::now();

    // If a turn is in progress, keep turning until the duration elapses
    if (turning_) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        now_steady - turn_start_time_).count();
      if (elapsed < turn_duration_s_) {
        return make_twist(0.0, turn_sign_ * behaviors::TURN_SPEED_RADS);
      }
      turning_ = false;
    }

    // Trigger a new turn once 1ft of forward travel has accumulated
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

  /**
   * Priority 6 (lowest) — Drive forward at constant speed.
   *
   * Default behavior — active whenever no higher priority behavior fires.
   * Always returns a value (never nullopt) so the robot is never left with
   * no command.
   */
  geometry_msgs::msg::TwistStamped drive_behavior() const
  {
    return make_twist(behaviors::FORWARD_SPEED_MPS, 0.0);
  }

  /**
   * Main control loop — runs at 10 Hz via the wall timer.
   *
   * Implements the pseudo-subsumption priority chain. Each behavior returns
   * either a TwistStamped command or std::nullopt. The first behavior that
   * returns a command wins; all lower-priority behaviors are suppressed.
   * drive_behavior() is the unconditional fallback at the bottom.
   */
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

  // Number of 100ms control ticks to drive forward after escape turn completes.
  // 20 ticks * 100ms = 2 seconds of forward push to clear the wall.
  static constexpr int POST_ESCAPE_TICKS = 20;

  // Mutex protecting all shared state accessed by both subscription callbacks
  // and the timer-driven control loop.
  std::mutex mutex_;

  // ROS publishers and subscribers
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<irobot_create_msgs::msg::HazardDetectionVector>::SharedPtr hazard_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr key_vel_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Most recent laser scan message; null until first scan arrives
  sensor_msgs::msg::LaserScan::SharedPtr scan_;
  // Laser ranges within the forward ±30° sector, partitioned left/right
  std::vector<double> front_left_;
  std::vector<double> front_right_;

  // True when the Create 3 bumper ring has detected a physical collision
  bool collision_detected_{false};
  // True when a laser beam within ±45° is closer than LASER_HALT_DISTANCE_M
  bool laser_collision_{false};

  // Most recent keyboard command and the time it was received
  geometry_msgs::msg::TwistStamped::SharedPtr keyboard_cmd_;
  std::chrono::steady_clock::time_point last_key_time_{};
  bool has_key_time_{false};  // false until the first keyboard message arrives

  // Odometry state for yaw tracking and distance accumulation
  double last_odom_x_{0.0};
  double last_odom_y_{0.0};
  bool has_last_odom_{false};   // false until the first odom message arrives
  double current_yaw_{0.0};     // current robot heading in radians
  double distance_since_turn_{0.0};  // forward travel since last random turn

  // State for the random turn behavior (Priority 5)
  bool turning_{false};
  double turn_sign_{1.0};                              // +1 = left, -1 = right
  std::chrono::steady_clock::time_point turn_start_time_{};
  double turn_duration_s_{0.0};                        // pre-computed turn duration

  // State for the escape behavior (Priority 3)
  bool escaping_{false};
  double escape_target_yaw_{0.0};   // absolute yaw the robot must reach to exit escape
  int post_escape_ticks_{0};        // counts down the post-escape forward push
  std::chrono::steady_clock::time_point escape_cooldown_until_{};  // re-trigger lockout

  // Mersenne Twister RNG — seeded from hardware entropy at startup.
  // Used by escape (turn direction and angle) and turn behavior (random angle).
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
