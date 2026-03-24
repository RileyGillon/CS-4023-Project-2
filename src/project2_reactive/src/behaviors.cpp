#include "project2_reactive/behaviors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

constexpr double kPi = 3.14159265358979323846;

double normalize_angle(double angle)
{
  const double two_pi = 2.0 * kPi;
  double wrapped = std::fmod(angle + kPi, two_pi);
  if (wrapped < 0.0) {
    wrapped += two_pi;
  }
  return wrapped - kPi;
}

}  // namespace

namespace project2_reactive::behaviors
{

FrontDistances get_front_distances(
  const std::vector<float> & ranges,
  double angle_min,
  double angle_increment,
  double range_min,
  double range_max,
  double front_sector_angle)
{
  FrontDistances result;
  result.left.reserve(ranges.size() / 8);
  result.right.reserve(ranges.size() / 8);

  for (size_t i = 0; i < ranges.size(); ++i) {
    const double r = static_cast<double>(ranges[i]);
    if (!std::isfinite(r) || r < range_min || r > range_max) {
      continue;
    }

    const double raw_angle = angle_min + static_cast<double>(i) * angle_increment;
    const double angle = normalize_angle(raw_angle);
    if (std::abs(angle) <= front_sector_angle) {
      if (angle >= 0.0) {
        result.left.push_back(r);
      } else {
        result.right.push_back(r);
      }
    }
  }

  return result;
}

bool obstacle_in_range(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance)
{
  double min_d = std::numeric_limits<double>::infinity();
  for (const auto d : left_distances) {
    min_d = std::min(min_d, d);
  }
  for (const auto d : right_distances) {
    min_d = std::min(min_d, d);
  }
  return std::isfinite(min_d) && min_d < obstacle_distance;
}

bool is_symmetric_obstacle(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance,
  double symmetry_threshold)
{
  if (left_distances.empty() || right_distances.empty()) {
    return false;
  }

  const double left_min = *std::min_element(left_distances.begin(), left_distances.end());
  const double right_min = *std::min_element(right_distances.begin(), right_distances.end());

  if (left_min >= obstacle_distance && right_min >= obstacle_distance) {
    return false;
  }

  const double avg = (left_min + right_min) / 2.0;
  if (avg < 1e-9) {
    return true;
  }

  const double asymmetry = std::abs(left_min - right_min) / avg;
  return asymmetry <= symmetry_threshold;
}

double get_avoid_direction(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances)
{
  const double left_min = left_distances.empty() ?
    std::numeric_limits<double>::infinity() :
    *std::min_element(left_distances.begin(), left_distances.end());
  const double right_min = right_distances.empty() ?
    std::numeric_limits<double>::infinity() :
    *std::min_element(right_distances.begin(), right_distances.end());

  return (left_min <= right_min) ? -AVOID_TURN_SPEED_RADS : AVOID_TURN_SPEED_RADS;
}

double compute_distance(double x1, double y1, double x2, double y2)
{
  const double dx = x2 - x1;
  const double dy = y2 - y1;
  return std::sqrt(dx * dx + dy * dy);
}

double yaw_from_quaternion(double x, double y, double z, double w)
{
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double forward_displacement(double dx, double dy, double yaw)
{
  return dx * std::cos(yaw) + dy * std::sin(yaw);
}

double sample_random_turn_angle(std::mt19937 & rng, double turn_range)
{
  std::uniform_real_distribution<double> dist(-turn_range, turn_range);
  return dist(rng);
}

}  // namespace project2_reactive::behaviors
