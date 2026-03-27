#ifndef PROJECT2_REACTIVE__BEHAVIORS_HPP_
#define PROJECT2_REACTIVE__BEHAVIORS_HPP_

#include <random>
#include <utility>
#include <vector>

namespace project2_reactive::behaviors
{

// Distance threshold for obstacle detection (1 ft = 0.3048 m)
constexpr double OBSTACLE_DISTANCE_M = 0.3048;

// Distance the robot must travel forward before triggering a random turn (1 ft)
constexpr double TURN_DISTANCE_M = 0.3048;

// Linear speed for normal forward driving (m/s)
constexpr double FORWARD_SPEED_MPS = 0.15;

// Angular speed used for the random turn behavior (rad/s)
constexpr double TURN_SPEED_RADS = 0.5;

// Linear speed during escape backup (negative = reverse)
constexpr double ESCAPE_BACKUP_SPEED_MPS = -0.10;

// Angular speed used during the escape fixed action pattern (rad/s)
// Chosen so that 180 deg takes roughly 3 s at this rate
constexpr double ESCAPE_TURN_SPEED_RADS = 1.0;

// Angular speed used during the avoid reflex (rad/s)
constexpr double AVOID_TURN_SPEED_RADS = 0.5;

// Half-angle of the forward sector examined for obstacles (30 deg in radians)
constexpr double FRONT_SECTOR_ANGLE_RAD = 0.5235987755982988;

// Ratio threshold below which left/right min distances are considered symmetric
constexpr double SYMMETRY_RATIO_THRESHOLD = 0.15;

// Half-range for the random turn sample (15 deg in radians)
constexpr double TURN_ANGLE_RANGE_RAD = 0.2617993877991494;

// Time window after the last keyboard message before keyboard behavior deactivates (s)
constexpr double KEYBOARD_TIMEOUT_S = 0.5;

// Cooldown after an escape action before escape can trigger again (s)
constexpr double ESCAPE_COOLDOWN_S = 3.0;

// Laser halt distance: stop if anything this close appears in the halt sector (m).
// Set to 0.22 m to account for physical sensor noise on the real TurtleBot 4.
constexpr double LASER_HALT_DISTANCE_M = 0.22;

// Half-angle of the sector checked for the laser-based halt (45 deg in radians)
constexpr double LASER_HALT_SECTOR_RAD = 0.7853981633974483;

/// Holds the filtered front-sector laser distances split into left and right halves.
struct FrontDistances
{
  std::vector<double> left;
  std::vector<double> right;
};

/// Filter laser scan ranges to only those inside the forward sector and split
/// them into left (angle >= 0) and right (angle < 0) halves.
FrontDistances get_front_distances(
  const std::vector<float> & ranges,
  double angle_min,
  double angle_increment,
  double range_min,
  double range_max,
  double front_sector_angle = FRONT_SECTOR_ANGLE_RAD);

/// Return true if any reading in the front sector is closer than obstacle_distance.
bool obstacle_in_range(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance = OBSTACLE_DISTANCE_M);

/// Return true if the closest obstacles on each side are roughly equal in distance,
/// indicating a symmetric (head-on) obstacle requiring the escape behavior.
bool is_symmetric_obstacle(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances,
  double obstacle_distance = OBSTACLE_DISTANCE_M,
  double symmetry_threshold = SYMMETRY_RATIO_THRESHOLD);

/// Return the signed angular velocity the robot should apply to turn away from
/// the closer obstacle side (negative = right, positive = left).
double get_avoid_direction(
  const std::vector<double> & left_distances,
  const std::vector<double> & right_distances);

/// Euclidean distance between two 2-D points.
double compute_distance(double x1, double y1, double x2, double y2);

/// Extract yaw angle from a quaternion (x, y, z, w).
double yaw_from_quaternion(double x, double y, double z, double w);

/// Project a 2-D displacement (dx, dy) onto the robot's forward axis given its yaw.
double forward_displacement(double dx, double dy, double yaw);

/// Sample a uniformly random turn angle in [-turn_range, +turn_range] radians.
double sample_random_turn_angle(
  std::mt19937 & rng,
  double turn_range = TURN_ANGLE_RANGE_RAD);

}  // namespace project2_reactive::behaviors

#endif  // PROJECT2_REACTIVE__BEHAVIORS_HPP_
