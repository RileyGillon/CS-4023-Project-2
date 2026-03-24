# Copyright 2026 Riley
#
# Licensed under the MIT License.
#
# test_reactive_controller.py
# ---------------------------
# Unit tests for ReactiveController behaviour methods.
#
# ROS 2 runtime packages are mocked at the module level so these tests
# can be run without a ROS 2 installation:
#
#   cd src/project2_reactive
#   python -m pytest test/test_reactive_controller.py -v

import math
import sys
import threading
import time
from typing import List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock all ROS 2 packages before importing the controller
# ---------------------------------------------------------------------------
# We replace every ROS 2 module in sys.modules with a MagicMock so that
# the `import rclpy` and `from geometry_msgs.msg import TwistStamped` calls
# inside reactive_controller.py resolve without error.

_ROS_MODULES = [
    'rclpy',
    'rclpy.node',
    'rclpy.clock',
    'rclpy.time',
    'geometry_msgs',
    'geometry_msgs.msg',
    'sensor_msgs',
    'sensor_msgs.msg',
    'nav_msgs',
    'nav_msgs.msg',
    'irobot_create_msgs',
    'irobot_create_msgs.msg',
]

for _mod in _ROS_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Set up HazardDetection constants so the controller's
# `HazardDetection.BUMP` comparison resolves to the integer 1.
_hd_cls = MagicMock()
_hd_cls.BUMP = 1
sys.modules['irobot_create_msgs.msg'].HazardDetection = _hd_cls
sys.modules['irobot_create_msgs.msg'].HazardDetectionVector = MagicMock()

# Now import the module under test.
from project2_reactive import behaviors  # noqa: E402
from project2_reactive.reactive_controller import ReactiveController  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a mock LaserScan
# ---------------------------------------------------------------------------

def _make_mock_scan(
    close_both: float = None,
    left_dist: float = None,
    right_dist: float = None,
    background: float = 5.0,
    n_rays: int = 360,
) -> MagicMock:
    """Return a mock LaserScan whose ranges list is a real Python list.

    Parameters
    ----------
    close_both:
        If given, every ray in the front sector is set to this value.
    left_dist:
        Distance for left-sector (positive-angle) front rays.
    right_dist:
        Distance for right-sector (negative-angle) front rays.
    background:
        Distance for all other rays (default 5 m – clear).
    """
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / n_rays
    front_angle = behaviors.FRONT_SECTOR_ANGLE_RAD
    ranges: List[float] = [background] * n_rays

    for i in range(n_rays):
        raw = angle_min + i * angle_inc
        angle = (raw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(angle) <= front_angle:
            if close_both is not None:
                ranges[i] = close_both
            elif angle >= 0.0 and left_dist is not None:
                ranges[i] = left_dist
            elif angle < 0.0 and right_dist is not None:
                ranges[i] = right_dist

    scan = MagicMock()
    scan.ranges = ranges
    scan.angle_min = angle_min
    scan.angle_increment = angle_inc
    scan.range_min = 0.1
    scan.range_max = 10.0
    return scan


# ---------------------------------------------------------------------------
# Fixture: a ReactiveController with all ROS infrastructure mocked out
# ---------------------------------------------------------------------------

@pytest.fixture()
def ctrl() -> ReactiveController:
    """Instantiate ReactiveController without running ROS __init__.

    Uses __new__ to bypass the Node/ROS set-up and then manually
    initialises every attribute that the behaviour methods access.
    """
    c: ReactiveController = ReactiveController.__new__(ReactiveController)

    # Sensor / collision state.
    c._collision_detected = False
    c._scan = None
    c._keyboard_cmd = None
    c._last_key_time = None

    # Odometry / distance state.
    c._last_odom_x = None
    c._last_odom_y = None
    c._current_yaw = 0.0
    c._distance_since_turn = 0.0

    # Turn state.
    c._turning = False
    c._turn_sign = 1.0
    c._turn_start_wall = 0.0
    c._turn_duration = 0.0

    # Thread safety.
    c._lock = threading.Lock()

    # Mock the ROS clock used inside _make_twist().
    mock_ts = MagicMock()
    mock_clock = MagicMock()
    mock_clock.now.return_value.to_msg.return_value = mock_ts
    c.get_clock = MagicMock(return_value=mock_clock)

    # Mock the publisher used in _control_loop().
    c._cmd_vel_pub = MagicMock()

    # Mock the logger so warn/info calls don't raise.
    c.get_logger = MagicMock(return_value=MagicMock())

    return c


# ---------------------------------------------------------------------------
# TestHaltBehavior
# ---------------------------------------------------------------------------


class TestHaltBehavior:
    """Priority 1: _halt_behavior()."""

    def test_returns_command_when_collision_detected(self, ctrl) -> None:
        ctrl._collision_detected = True
        result = ctrl._halt_behavior()
        assert result is not None

    def test_returns_none_when_no_collision(self, ctrl) -> None:
        ctrl._collision_detected = False
        assert ctrl._halt_behavior() is None

    def test_stop_command_has_zero_velocities(self, ctrl) -> None:
        ctrl._collision_detected = True
        result = ctrl._halt_behavior()
        assert result.twist.linear.x == pytest.approx(0.0)
        assert result.twist.angular.z == pytest.approx(0.0)

    def test_returns_none_after_collision_cleared(self, ctrl) -> None:
        ctrl._collision_detected = True
        assert ctrl._halt_behavior() is not None
        ctrl._collision_detected = False
        assert ctrl._halt_behavior() is None


# ---------------------------------------------------------------------------
# TestKeyboardBehavior
# ---------------------------------------------------------------------------


class TestKeyboardBehavior:
    """Priority 2: _keyboard_behavior()."""

    def test_returns_none_when_no_cmd_received(self, ctrl) -> None:
        assert ctrl._keyboard_behavior() is None

    def test_returns_none_when_cmd_expired(self, ctrl) -> None:
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.5
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.0
        ctrl._keyboard_cmd = mock_cmd
        # Timestamp 10 seconds ago – well past the timeout.
        ctrl._last_key_time = time.monotonic() - 10.0
        assert ctrl._keyboard_behavior() is None

    def test_returns_cmd_when_recent_and_nonzero(self, ctrl) -> None:
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.2
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.0
        ctrl._keyboard_cmd = mock_cmd
        ctrl._last_key_time = time.monotonic()
        assert ctrl._keyboard_behavior() is not None

    def test_returns_none_when_recent_but_all_zero(self, ctrl) -> None:
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.0
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.0
        ctrl._keyboard_cmd = mock_cmd
        ctrl._last_key_time = time.monotonic()
        assert ctrl._keyboard_behavior() is None

    def test_returns_cmd_for_angular_only_input(self, ctrl) -> None:
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.0
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.5
        ctrl._keyboard_cmd = mock_cmd
        ctrl._last_key_time = time.monotonic()
        assert ctrl._keyboard_behavior() is not None

    def test_cmd_is_the_original_message(self, ctrl) -> None:
        """The keyboard behavior should pass through the original message."""
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.3
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.0
        ctrl._keyboard_cmd = mock_cmd
        ctrl._last_key_time = time.monotonic()
        assert ctrl._keyboard_behavior() is mock_cmd


# ---------------------------------------------------------------------------
# TestEscapeBehavior
# ---------------------------------------------------------------------------


class TestEscapeBehavior:
    """Priority 3: _escape_behavior()."""

    def test_returns_none_when_no_scan(self, ctrl) -> None:
        ctrl._scan = None
        assert ctrl._escape_behavior() is None

    def test_returns_none_on_clear_path(self, ctrl) -> None:
        ctrl._scan = _make_mock_scan(background=5.0)
        assert ctrl._escape_behavior() is None

    def test_returns_command_on_symmetric_obstacle(self, ctrl) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)
        result = ctrl._escape_behavior()
        assert result is not None

    def test_escape_command_has_negative_linear_x(self, ctrl) -> None:
        """Escape should reverse: linear.x must be negative."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)
        result = ctrl._escape_behavior()
        assert result.twist.linear.x < 0.0

    def test_escape_command_has_nonzero_angular_z(self, ctrl) -> None:
        """Escape also turns to pivot away."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)
        result = ctrl._escape_behavior()
        assert abs(result.twist.angular.z) > 0.0

    def test_returns_none_on_asymmetric_obstacle(self, ctrl) -> None:
        """Asymmetric obstacle should NOT trigger escape."""
        ctrl._scan = _make_mock_scan(left_dist=0.1, right_dist=5.0)
        assert ctrl._escape_behavior() is None


# ---------------------------------------------------------------------------
# TestAvoidBehavior
# ---------------------------------------------------------------------------


class TestAvoidBehavior:
    """Priority 4: _avoid_behavior()."""

    def test_returns_none_when_no_scan(self, ctrl) -> None:
        assert ctrl._avoid_behavior() is None

    def test_returns_none_when_path_clear(self, ctrl) -> None:
        ctrl._scan = _make_mock_scan(background=5.0)
        assert ctrl._avoid_behavior() is None

    def test_returns_command_on_left_obstacle(self, ctrl) -> None:
        ctrl._scan = _make_mock_scan(left_dist=0.1, right_dist=5.0)
        assert ctrl._avoid_behavior() is not None

    def test_returns_command_on_right_obstacle(self, ctrl) -> None:
        ctrl._scan = _make_mock_scan(left_dist=5.0, right_dist=0.1)
        assert ctrl._avoid_behavior() is not None

    def test_returns_none_on_symmetric_obstacle(self, ctrl) -> None:
        """Symmetric obstacle is handled by escape, not avoid."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)
        assert ctrl._avoid_behavior() is None

    def test_left_obstacle_gives_right_turn(self, ctrl) -> None:
        """Left obstacle → negative angular-z (right / CW turn)."""
        ctrl._scan = _make_mock_scan(left_dist=0.1, right_dist=5.0)
        result = ctrl._avoid_behavior()
        assert result.twist.angular.z < 0.0

    def test_right_obstacle_gives_left_turn(self, ctrl) -> None:
        """Right obstacle → positive angular-z (left / CCW turn)."""
        ctrl._scan = _make_mock_scan(left_dist=5.0, right_dist=0.1)
        result = ctrl._avoid_behavior()
        assert result.twist.angular.z > 0.0

    def test_avoid_command_zero_linear_x(self, ctrl) -> None:
        """Avoid turns in place – linear.x should be zero."""
        ctrl._scan = _make_mock_scan(left_dist=0.1, right_dist=5.0)
        result = ctrl._avoid_behavior()
        assert result.twist.linear.x == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestTurnBehavior
# ---------------------------------------------------------------------------


class TestTurnBehavior:
    """Priority 5: _turn_behavior()."""

    def test_returns_none_when_no_distance_accumulated(self, ctrl) -> None:
        ctrl._distance_since_turn = 0.0
        ctrl._turning = False
        assert ctrl._turn_behavior() is None

    def test_returns_none_below_threshold(self, ctrl) -> None:
        ctrl._distance_since_turn = behaviors.TURN_DISTANCE_M * 0.5
        ctrl._turning = False
        assert ctrl._turn_behavior() is None

    def test_triggers_at_threshold(self, ctrl) -> None:
        ctrl._distance_since_turn = behaviors.TURN_DISTANCE_M + 0.01
        ctrl._turning = False
        result = ctrl._turn_behavior()
        assert result is not None

    def test_sets_turning_flag_when_triggered(self, ctrl) -> None:
        ctrl._distance_since_turn = behaviors.TURN_DISTANCE_M + 0.01
        ctrl._turning = False
        ctrl._turn_behavior()
        assert ctrl._turning is True

    def test_resets_distance_when_turn_starts(self, ctrl) -> None:
        ctrl._distance_since_turn = behaviors.TURN_DISTANCE_M + 0.01
        ctrl._turning = False
        ctrl._turn_behavior()
        assert ctrl._distance_since_turn == pytest.approx(0.0)

    def test_continues_turning_during_active_turn(self, ctrl) -> None:
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic()
        ctrl._turn_duration = 10.0   # turn lasts 10 s
        ctrl._turn_sign = 1.0
        result = ctrl._turn_behavior()
        assert result is not None

    def test_turn_command_zero_linear_x(self, ctrl) -> None:
        """Turns should be in place (no forward motion)."""
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic()
        ctrl._turn_duration = 10.0
        ctrl._turn_sign = 1.0
        result = ctrl._turn_behavior()
        assert result.twist.linear.x == pytest.approx(0.0)

    def test_turn_sign_positive_gives_positive_angular_z(self, ctrl) -> None:
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic()
        ctrl._turn_duration = 10.0
        ctrl._turn_sign = 1.0
        result = ctrl._turn_behavior()
        assert result.twist.angular.z > 0.0

    def test_turn_sign_negative_gives_negative_angular_z(self, ctrl) -> None:
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic()
        ctrl._turn_duration = 10.0
        ctrl._turn_sign = -1.0
        result = ctrl._turn_behavior()
        assert result.twist.angular.z < 0.0

    def test_clears_turning_flag_after_duration(self, ctrl) -> None:
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic() - 5.0  # 5 s ago
        ctrl._turn_duration = 1.0                        # only 1 s needed
        ctrl._distance_since_turn = 0.0
        ctrl._turn_behavior()
        assert ctrl._turning is False

    def test_returns_none_immediately_after_turn_ends(self, ctrl) -> None:
        """After a turn completes and no new trigger, should return None."""
        ctrl._turning = True
        ctrl._turn_start_wall = time.monotonic() - 5.0
        ctrl._turn_duration = 1.0
        ctrl._distance_since_turn = 0.0
        result = ctrl._turn_behavior()
        assert result is None


# ---------------------------------------------------------------------------
# TestDriveBehavior
# ---------------------------------------------------------------------------


class TestDriveBehavior:
    """Priority 6 (default): _drive_behavior()."""

    def test_always_returns_a_command(self, ctrl) -> None:
        assert ctrl._drive_behavior() is not None

    def test_forward_speed_is_positive(self, ctrl) -> None:
        result = ctrl._drive_behavior()
        assert result.twist.linear.x > 0.0

    def test_forward_speed_matches_constant(self, ctrl) -> None:
        result = ctrl._drive_behavior()
        assert result.twist.linear.x == pytest.approx(behaviors.FORWARD_SPEED_MPS)

    def test_angular_z_is_zero(self, ctrl) -> None:
        result = ctrl._drive_behavior()
        assert result.twist.angular.z == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestControlLoopPriority
# ---------------------------------------------------------------------------


class TestControlLoopPriority:
    """Verify the priority order enforced by _control_loop()."""

    def test_halt_supersedes_all_when_collision(self, ctrl) -> None:
        """Collision → Halt fires; publisher receives exactly one stop command."""
        ctrl._collision_detected = True
        # Make Escape also active.
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)
        # Make Keyboard also active.
        mock_cmd = MagicMock()
        mock_cmd.twist.linear.x = 0.5
        mock_cmd.twist.linear.y = 0.0
        mock_cmd.twist.angular.z = 0.0
        ctrl._keyboard_cmd = mock_cmd
        ctrl._last_key_time = time.monotonic()

        ctrl._control_loop()

        ctrl._cmd_vel_pub.publish.assert_called_once()
        published = ctrl._cmd_vel_pub.publish.call_args[0][0]
        assert published.twist.linear.x == pytest.approx(0.0)
        assert published.twist.angular.z == pytest.approx(0.0)

    def test_drive_fires_when_all_clear(self, ctrl) -> None:
        """No collision, no keyboard, no obstacles → Drive fires."""
        ctrl._collision_detected = False
        ctrl._scan = _make_mock_scan(background=5.0)

        ctrl._control_loop()

        ctrl._cmd_vel_pub.publish.assert_called_once()
        published = ctrl._cmd_vel_pub.publish.call_args[0][0]
        assert published.twist.linear.x == pytest.approx(behaviors.FORWARD_SPEED_MPS)

    def test_keyboard_supersedes_autonomous_behaviors(self, ctrl) -> None:
        """Active keyboard → keyboard command published, not autonomous."""
        ctrl._collision_detected = False
        ctrl._scan = _make_mock_scan(background=5.0)

        original_msg = MagicMock()
        original_msg.twist.linear.x = 0.3
        original_msg.twist.linear.y = 0.0
        original_msg.twist.angular.z = 0.0
        ctrl._keyboard_cmd = original_msg
        ctrl._last_key_time = time.monotonic()

        ctrl._control_loop()

        ctrl._cmd_vel_pub.publish.assert_called_once()
        published = ctrl._cmd_vel_pub.publish.call_args[0][0]
        assert published is original_msg

    def test_exactly_one_command_published_per_cycle(self, ctrl) -> None:
        """The loop must publish exactly one command per control cycle."""
        ctrl._collision_detected = False
        ctrl._scan = _make_mock_scan(background=5.0)

        ctrl._control_loop()

        assert ctrl._cmd_vel_pub.publish.call_count == 1

    def test_escape_supersedes_avoid(self, ctrl) -> None:
        """Symmetric obstacle → Escape fires (not Avoid, not Drive)."""
        ctrl._collision_detected = False
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        ctrl._scan = _make_mock_scan(close_both=close)

        ctrl._control_loop()

        ctrl._cmd_vel_pub.publish.assert_called_once()
        published = ctrl._cmd_vel_pub.publish.call_args[0][0]
        # Escape commands backward motion.
        assert published.twist.linear.x < 0.0


# ---------------------------------------------------------------------------
# TestOdomCallback (state-level tests, no ROS spin needed)
# ---------------------------------------------------------------------------


class TestOdomCallback:
    """Tests for the internal state transitions driven by odometry."""

    def _mock_odom(self, x: float, y: float, yaw: float = 0.0) -> MagicMock:
        """Build a minimal Odometry mock."""
        q_z = math.sin(yaw / 2)
        q_w = math.cos(yaw / 2)
        msg = MagicMock()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = q_z
        msg.pose.pose.orientation.w = q_w
        return msg

    def test_first_callback_sets_position(self, ctrl) -> None:
        ctrl._odom_callback(self._mock_odom(1.0, 2.0))
        assert ctrl._last_odom_x == pytest.approx(1.0)
        assert ctrl._last_odom_y == pytest.approx(2.0)

    def test_first_callback_does_not_accumulate_distance(self, ctrl) -> None:
        ctrl._odom_callback(self._mock_odom(1.0, 2.0))
        assert ctrl._distance_since_turn == pytest.approx(0.0)

    def test_forward_motion_accumulates_distance(self, ctrl) -> None:
        ctrl._odom_callback(self._mock_odom(0.0, 0.0, yaw=0.0))
        ctrl._odom_callback(self._mock_odom(0.2, 0.0, yaw=0.0))
        assert ctrl._distance_since_turn == pytest.approx(0.2)

    def test_backward_motion_does_not_accumulate(self, ctrl) -> None:
        ctrl._odom_callback(self._mock_odom(0.0, 0.0, yaw=0.0))
        ctrl._odom_callback(self._mock_odom(-0.2, 0.0, yaw=0.0))
        assert ctrl._distance_since_turn == pytest.approx(0.0)

    def test_lateral_motion_does_not_accumulate(self, ctrl) -> None:
        # Heading is along +x (yaw=0); moving along +y is purely lateral.
        ctrl._odom_callback(self._mock_odom(0.0, 0.0, yaw=0.0))
        ctrl._odom_callback(self._mock_odom(0.0, 0.2, yaw=0.0))
        assert ctrl._distance_since_turn == pytest.approx(0.0, abs=1e-9)

    def test_multiple_steps_accumulate_correctly(self, ctrl) -> None:
        for i in range(1, 4):
            ctrl._odom_callback(self._mock_odom(i * 0.1, 0.0, yaw=0.0))
        # Three steps of 0.1 m each = 0.3 m total.
        assert ctrl._distance_since_turn == pytest.approx(0.3)

    def test_yaw_extracted_correctly(self, ctrl) -> None:
        yaw_expected = math.pi / 4
        ctrl._odom_callback(self._mock_odom(0.0, 0.0, yaw=yaw_expected))
        assert ctrl._current_yaw == pytest.approx(yaw_expected)
