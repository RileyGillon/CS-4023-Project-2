#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "project2_reactive/behaviors.hpp"

namespace b = project2_reactive::behaviors;

namespace
{

constexpr double kPi = 3.14159265358979323846;

std::vector<float> make_full_scan(float distance, size_t n = 360)
{
  return std::vector<float>(n, distance);
}

}  // namespace

TEST(Behaviors, FrontDistancesIncludeFrontSector)
{
  const auto ranges = make_full_scan(1.0F);
  const auto front = b::get_front_distances(
    ranges, -kPi, 2.0 * kPi / 360.0, 0.1, 10.0);

  EXPECT_GT(front.left.size(), 0u);
  EXPECT_GT(front.right.size(), 0u);
  EXPECT_EQ(front.left.size() + front.right.size(), 61u);
}

TEST(Behaviors, FrontDistancesFilterInvalidReadings)
{
  std::vector<float> ranges(360, std::numeric_limits<float>::infinity());
  const auto front = b::get_front_distances(
    ranges, -kPi, 2.0 * kPi / 360.0, 0.1, 10.0);

  EXPECT_TRUE(front.left.empty());
  EXPECT_TRUE(front.right.empty());
}

TEST(Behaviors, ObstacleInRangeWorks)
{
  EXPECT_FALSE(b::obstacle_in_range({1.0, 1.2}, {1.1, 1.3}));
  EXPECT_TRUE(b::obstacle_in_range({0.2}, {1.2}));
}

TEST(Behaviors, SymmetricObstacleDetection)
{
  EXPECT_TRUE(b::is_symmetric_obstacle({0.25}, {0.26}));
  EXPECT_FALSE(b::is_symmetric_obstacle({0.20}, {0.45}));
}

TEST(Behaviors, AvoidDirectionTurnsAway)
{
  EXPECT_LT(b::get_avoid_direction({0.15}, {0.8}), 0.0);
  EXPECT_GT(b::get_avoid_direction({0.8}, {0.15}), 0.0);
}

TEST(Behaviors, DistanceIsEuclidean)
{
  EXPECT_NEAR(b::compute_distance(0.0, 0.0, 3.0, 4.0), 5.0, 1e-9);
}

TEST(Behaviors, YawFromQuaternion)
{
  const double angle = kPi / 2.0;
  const double s = std::sin(angle / 2.0);
  const double c = std::cos(angle / 2.0);
  EXPECT_NEAR(b::yaw_from_quaternion(0.0, 0.0, s, c), angle, 1e-9);
}

TEST(Behaviors, ForwardDisplacementProjection)
{
  EXPECT_NEAR(b::forward_displacement(1.0, 0.0, 0.0), 1.0, 1e-9);
  EXPECT_NEAR(b::forward_displacement(0.0, 1.0, 0.0), 0.0, 1e-9);
}

TEST(Behaviors, RandomTurnSampleWithinRange)
{
  std::mt19937 rng(42);
  for (int i = 0; i < 1000; ++i) {
    const double angle = b::sample_random_turn_angle(rng);
    EXPECT_LE(angle, b::TURN_ANGLE_RANGE_RAD);
    EXPECT_GE(angle, -b::TURN_ANGLE_RANGE_RAD);
  }
}

TEST(Behaviors, ConstantsMatchAssignment)
{
  EXPECT_NEAR(b::TURN_DISTANCE_M, 0.3048, 1e-9);
  EXPECT_NEAR(b::OBSTACLE_DISTANCE_M, 0.6096, 1e-9);
}
