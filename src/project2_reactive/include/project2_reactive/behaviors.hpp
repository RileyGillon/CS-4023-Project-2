#ifndef PROJECT2_REACTIVE__BEHAVIORS_HPP_
#define PROJECT2_REACTIVE__BEHAVIORS_HPP_

#include <random>
#include <utility>
#include <vector>

namespace project2_reactive::behaviors
{

// 1ft = 0.3048m — obstacle detection threshold (spec: "within 1ft in front")
constexpr double OBSTACLE_DISTANCE_M = 0.3048;
// 1ft forward movement triggers a random turn
constexpr double TURN_DISTANCE_M = 0.3048;

constexpr double FORWARD_SPEED_MPS = 0.15;
constexpr double TURN_SPEED_RADS = 0.5;
constexpr double ESCAPE_BACKUP_SPEED_MPS = -0.10;
// Escape turn speed — faster than normal turns so 180° completes quickly
constexpr double ESCAPE_TURN_SPEED_RADS = 1.0;
constexpr double AVOID_TURN_SPEED_RADS = 0.5;

constexpr double FRONT_SECTOR_ANGLE_RAD = 0.5235987755982988;  // 30 deg
constexpr double SYMMETRY_RATIO_THRESHOLD = 0.15;
constexpr double TURN_ANGLE_RANGE_RAD = 0.2617993877991494;    // 15 deg
constexpr double KEYBOARD_TIMEOUT_S = 0.5;

constexpr double ESCAPE_COOLDOWN_S = 3.0;
// Escape turn complete when within ±30° of target yaw (matches spec's ±30° tolerance)
constexpr double ESCAPE_TOLERANCE_RAD = 0.5235987755982988;    // 30 deg

// 0.22m — slightly above robot radius to account for physical sensor noise/latency
constexpr double LASER_HALT_DISTANCE_M = 0.22;
constexpr double LASER_HALT_SECTOR_RAD = 0.7853981633974483;   // 45 deg

struct FrontDistances
{
  std::vector<double> left;
  std::vector<double> right;
};

FrontDistances get_front_distances(
  const std::vector<float> & ranges,
  double angle_min,
  double angle_increment,
  double range_min,
  double range_max,
  double front_sector_angle = FRONT_SECTOR_ANGLE_RAD);

bool obstacle_in_range(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance = OBSTACLE_DISTANCE_M);

bool is_symmetric_obstacle(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance = OBSTACLE_DISTANCE_M,
  double symmetry_threshold = SYMMETRY_RATIO_THRESHOLD);

double get_avoid_direction(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances);

double compute_distance(double x1, double y1, double x2, double y2);

double yaw_from_quaternion(double x, double y, double z, double w);

double forward_displacement(double dx, double dy, double yaw);

double sample_random_turn_angle(
  std::mt19937 & rng,
  double turn_range = TURN_ANGLE_RANGE_RAD);

}  // namespace project2_reactive::behaviors

#endif  // PROJECT2_REACTIVE__BEHAVIORS_HPP_
