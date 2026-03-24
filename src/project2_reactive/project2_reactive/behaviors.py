# Copyright 2026 Riley
#
# Licensed under the MIT License.
#
# behaviors.py
# ------------
# Pure behavior logic for the reactive TurtleBot 4 controller.
#
# All functions in this module are free of ROS 2 dependencies so they can
# be unit-tested directly with plain pytest, without spinning up a ROS
# graph.  The reactive_controller module imports these functions and wraps
# their outputs in ROS 2 message types.
#
# Reactive architecture: priority-based arbitration (subsumption-inspired).
# Six behaviors are evaluated in order from highest to lowest priority:
#   1. Halt     – stop on bumper collision
#   2. Keyboard – forward human tele-op commands
#   3. Escape   – back away from symmetric frontal obstacles
#   4. Avoid    – turn away from asymmetric frontal obstacles
#   5. Turn     – randomly re-orient after every 1 ft of forward travel
#   6. Drive    – default: drive straight ahead

import math
import random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Obstacle detection threshold (2 ft in metres).
#: A 2 ft lookahead gives the robot enough space to react before contact.
OBSTACLE_DISTANCE_M: float = 0.6096

#: Distance of forward travel (in metres) after which a random turn fires.
TURN_DISTANCE_M: float = 0.3048

# ---------------------------------------------------------------------------
# Speed constants
# ---------------------------------------------------------------------------

#: Default forward driving speed (m/s).
FORWARD_SPEED_MPS: float = 0.15

#: Angular speed used for turns and escape manoeuvres (rad/s).
TURN_SPEED_RADS: float = 0.5

#: Linear speed during an escape backup (negative = backwards, m/s).
ESCAPE_BACKUP_SPEED_MPS: float = -0.10

#: Angular speed used while turning away from an obstacle (rad/s).
AVOID_TURN_SPEED_RADS: float = 0.5

# ---------------------------------------------------------------------------
# Sensor / behaviour-tuning constants
# ---------------------------------------------------------------------------

#: Half-angle of the front sector used for obstacle detection (radians).
#: Rays within ±FRONT_SECTOR_ANGLE_RAD of the robot's forward axis are
#: considered "front" readings.
FRONT_SECTOR_ANGLE_RAD: float = math.radians(30.0)

#: Maximum normalised left/right imbalance still considered *symmetric*.
#: asymmetry = |left_min - right_min| / mean(left_min, right_min)
#: Values at or below this threshold trigger Escape; values above trigger Avoid.
SYMMETRY_RATIO_THRESHOLD: float = 0.15

#: Maximum half-angle for a random re-orientation turn (radians).
TURN_ANGLE_RANGE_RAD: float = math.radians(15.0)

#: Seconds without a keyboard message before the Keyboard behaviour deactivates.
KEYBOARD_TIMEOUT_S: float = 0.5

#: Escape re-trigger cooldown in seconds after leaving an obstacle.
ESCAPE_COOLDOWN_S: float = 3.0

#: Laser pre-collision halt distance in metres.
LASER_HALT_DISTANCE_M: float = 0.18

#: Half-angle of the laser halt sector around the forward axis.
LASER_HALT_SECTOR_RAD: float = math.radians(45.0)


# ---------------------------------------------------------------------------
# Laser-scan helpers
# ---------------------------------------------------------------------------

def get_front_distances(
    ranges: List[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    front_sector_angle: float = FRONT_SECTOR_ANGLE_RAD,
) -> Tuple[List[float], List[float]]:
    """Return valid range readings inside the robot's front sector.

    The front sector spans ``[-front_sector_angle, +front_sector_angle]``
    around the robot's forward axis (angle = 0 in the laser frame).  Angles
    are normalised to ``[-π, π]`` so the function works regardless of
    whether ``angle_min`` starts at 0 or -π.

    Parameters
    ----------
    ranges:
        Raw range array from ``sensor_msgs/LaserScan``.
    angle_min:
        Starting angle of the first ray (radians).
    angle_increment:
        Angular step between consecutive rays (radians).
    range_min:
        Minimum valid range reported by the sensor (metres).
    range_max:
        Maximum valid range reported by the sensor (metres).
    front_sector_angle:
        Half-width of the front sector (radians).

    Returns
    -------
    left_distances:
        Valid ranges for rays with a normalised angle in ``[0, front_sector_angle]``
        (left side of the robot when viewed from above).
    right_distances:
        Valid ranges for rays with a normalised angle in
        ``(-front_sector_angle, 0)`` (right side).
    """
    left_distances: List[float] = []
    right_distances: List[float] = []

    for i, r in enumerate(ranges):
        # Discard non-finite or out-of-range readings.
        if not math.isfinite(r) or not (range_min <= r <= range_max):
            continue

        # Normalise angle to [-π, π] to handle scan frames that start at 0.
        raw_angle = angle_min + i * angle_increment
        angle = (raw_angle + math.pi) % (2.0 * math.pi) - math.pi

        if abs(angle) <= front_sector_angle:
            if angle >= 0.0:
                left_distances.append(r)
            else:
                right_distances.append(r)

    return left_distances, right_distances


# ---------------------------------------------------------------------------
# Obstacle-classification helpers
# ---------------------------------------------------------------------------

def obstacle_in_range(
    left_distances: List[float],
    right_distances: List[float],
    obstacle_distance: float = OBSTACLE_DISTANCE_M,
) -> bool:
    """Return ``True`` if any front reading is closer than *obstacle_distance*.

    Parameters
    ----------
    left_distances, right_distances:
        Outputs of :func:`get_front_distances`.
    obstacle_distance:
        Detection threshold in metres.
    """
    all_front = left_distances + right_distances
    if not all_front:
        return False
    return min(all_front) < obstacle_distance


def is_symmetric_obstacle(
    left_distances: List[float],
    right_distances: List[float],
    obstacle_distance: float = OBSTACLE_DISTANCE_M,
    symmetry_threshold: float = SYMMETRY_RATIO_THRESHOLD,
) -> bool:
    """Return ``True`` when the front obstacle is roughly symmetric.

    An obstacle is symmetric (triggering Escape) when *both* left and right
    minimum distances are below *obstacle_distance* and their normalised
    difference does not exceed *symmetry_threshold*.

    Parameters
    ----------
    left_distances, right_distances:
        Outputs of :func:`get_front_distances`.
    obstacle_distance:
        Detection threshold in metres.
    symmetry_threshold:
        Maximum allowed normalised asymmetry to still be considered symmetric.
    """
    if not left_distances or not right_distances:
        return False

    left_min = min(left_distances)
    right_min = min(right_distances)

    # Both sides must be within the detection range.
    if left_min >= obstacle_distance and right_min >= obstacle_distance:
        return False

    avg = (left_min + right_min) / 2.0
    if avg < 1e-9:
        # Both values are essentially zero – symmetric by definition.
        return True

    asymmetry = abs(left_min - right_min) / avg
    return asymmetry <= symmetry_threshold


def get_avoid_direction(
    left_distances: List[float],
    right_distances: List[float],
) -> float:
    """Return the angular-z velocity needed to turn away from an obstacle.

    Returns positive (counter-clockwise / left turn) when the obstacle is
    on the right, and negative (clockwise / right turn) when it is on the
    left.  The magnitude is ``AVOID_TURN_SPEED_RADS``.

    Parameters
    ----------
    left_distances, right_distances:
        Outputs of :func:`get_front_distances`.  An empty list is treated
        as infinite distance on that side.
    """
    left_min = min(left_distances) if left_distances else float('inf')
    right_min = min(right_distances) if right_distances else float('inf')

    if left_min <= right_min:
        # Closer obstacle is on the left – turn right (negative angular-z).
        return -AVOID_TURN_SPEED_RADS
    else:
        # Closer obstacle is on the right – turn left (positive angular-z).
        return AVOID_TURN_SPEED_RADS


# ---------------------------------------------------------------------------
# Odometry helpers
# ---------------------------------------------------------------------------

def compute_distance(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Euclidean distance between two 2-D points."""
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def yaw_from_quaternion(
    x: float, y: float, z: float, w: float
) -> float:
    """Extract the yaw component from a unit quaternion.

    Parameters
    ----------
    x, y, z, w:
        Quaternion components (``geometry_msgs/Quaternion`` ordering).

    Returns
    -------
    float
        Yaw angle in radians, range ``[-π, π]``.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def forward_displacement(
    dx: float, dy: float, yaw: float
) -> float:
    """Project a displacement vector onto the robot's forward axis.

    Returns the signed projection; positive means the robot moved forward.

    Parameters
    ----------
    dx, dy:
        World-frame displacement since the last odometry reading.
    yaw:
        Current robot heading (radians).
    """
    return dx * math.cos(yaw) + dy * math.sin(yaw)


# ---------------------------------------------------------------------------
# Turn-angle sampling
# ---------------------------------------------------------------------------

def sample_random_turn_angle(
    turn_range: float = TURN_ANGLE_RANGE_RAD,
) -> float:
    """Sample a turn angle uniformly from ``[-turn_range, +turn_range]``.

    Parameters
    ----------
    turn_range:
        Maximum magnitude of the sampled angle (radians).  Must be >= 0.
    """
    return random.uniform(-turn_range, turn_range)
