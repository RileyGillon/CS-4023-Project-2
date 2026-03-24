# Copyright 2026 Riley
#
# Licensed under the MIT License.
#
# reactive_controller.py
# ----------------------
# ROS 2 node that implements priority-based reactive control for the
# TurtleBot 4.
#
# Architecture: subsumption-inspired priority arbitration.
# Behaviours are tested in order from highest to lowest priority.  The
# first behaviour that returns a non-None command is published to /cmd_vel;
# lower-priority behaviours are suppressed for that control cycle.
#
# Priority order (highest → lowest):
#   1. Halt     – stop immediately when a bumper collision is detected
#   2. Keyboard – forward human tele-op commands arriving on /key_vel
#   3. Escape   – back away from a symmetric frontal obstacle
#   4. Avoid    – turn away from an asymmetric frontal obstacle
#   5. Turn     – randomly re-orient after every 1 ft of forward travel
#   6. Drive    – default: drive straight forward

import math
import random
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from project2_reactive import behaviors


class ReactiveController(Node):
    """Priority-based reactive controller for a TurtleBot 4.

    Subscribes to:
        /scan               – LaserScan from the onboard LiDAR
        /hazard_detection   – HazardDetectionVector for bumper events
        /key_vel            – TwistStamped from the remapped tele-op node
        /odom               – Odometry for forward-distance tracking

    Publishes to:
        /cmd_vel            – TwistStamped velocity commands

    The control loop runs at 10 Hz.  Sensor callbacks update shared state
    under a lock; the control loop reads that state and selects the
    highest-priority active behaviour.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__('reactive_controller')

        # ---- Publishers --------------------------------------------------
        self._cmd_vel_pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )

        # ---- Subscriptions -----------------------------------------------
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10
        )
        self._hazard_sub = self.create_subscription(
            HazardDetectionVector,
            '/hazard_detection',
            self._hazard_callback,
            10,
        )
        self._key_vel_sub = self.create_subscription(
            TwistStamped, '/key_vel', self._key_vel_callback, 10
        )
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10
        )

        # ---- Shared state (protected by _lock) ---------------------------
        self._lock = threading.Lock()

        # Latest LiDAR scan.
        self._scan: Optional[LaserScan] = None
        # Cached front-sector distances from the latest scan.
        self._front_left: List[float] = []
        self._front_right: List[float] = []
        # Pre-collision flag from front laser proximity.
        self._laser_collision: bool = False

        # True while a bumper collision event is active.
        self._collision_detected: bool = False

        # Most recent keyboard velocity command and the wall-clock time at
        # which it arrived (None if no command has ever been received).
        self._keyboard_cmd: Optional[TwistStamped] = None
        self._last_key_time: Optional[float] = None  # time.monotonic()

        # Odometry: previous position for incremental distance tracking.
        self._last_odom_x: Optional[float] = None
        self._last_odom_y: Optional[float] = None
        # Current robot heading (yaw, radians) from odometry.
        self._current_yaw: float = 0.0
        # Forward distance travelled since the last random turn.
        self._distance_since_turn: float = 0.0

        # Random-turn state.
        self._turning: bool = False
        self._turn_sign: float = 1.0       # +1 (CCW) or -1 (CW)
        self._turn_start_wall: float = 0.0  # wall time at turn start
        self._turn_duration: float = 0.0   # seconds the turn should last

        # Escape fixed-action pattern state.
        self._escaping: bool = False
        self._escape_turn_sign: float = 1.0
        self._escape_cooldown_until: float = 0.0

        # ---- Control-loop timer at 10 Hz ---------------------------------
        self._timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info('ReactiveController initialised – running')

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _scan_callback(self, msg: LaserScan) -> None:
        """Store scan, cache front-sector ranges, and compute laser halt flag."""
        with self._lock:
            self._scan = msg
            left, right = behaviors.get_front_distances(
                list(msg.ranges),
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
            )
            self._front_left = left
            self._front_right = right

            self._laser_collision = any(
                math.isfinite(r)
                and msg.range_min <= r <= msg.range_max
                and r < behaviors.LASER_HALT_DISTANCE_M
                and abs(
                    (msg.angle_min + i * msg.angle_increment + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                < behaviors.LASER_HALT_SECTOR_RAD
                for i, r in enumerate(msg.ranges)
            )

    def _hazard_callback(self, msg: HazardDetectionVector) -> None:
        """Set the collision flag when a BUMP hazard is reported."""
        with self._lock:
            self._collision_detected = any(
                h.type == HazardDetection.BUMP for h in msg.detections
            )

    def _key_vel_callback(self, msg: TwistStamped) -> None:
        """Store the latest keyboard velocity command with a timestamp."""
        with self._lock:
            self._keyboard_cmd = msg
            self._last_key_time = time.monotonic()

    def _odom_callback(self, msg: Odometry) -> None:
        """Accumulate forward distance travelled using odometry.

        Only forward displacement (projected onto the robot's heading) is
        counted, so reverse motion during Escape does not prematurely
        trigger a random turn.
        """
        with self._lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation

            self._current_yaw = behaviors.yaw_from_quaternion(
                q.x, q.y, q.z, q.w
            )

            if self._last_odom_x is not None:
                dx = x - self._last_odom_x
                dy = y - self._last_odom_y
                fwd = behaviors.forward_displacement(
                    dx, dy, self._current_yaw
                )
                # Only accumulate genuine forward movement.
                if fwd > 0.0:
                    self._distance_since_turn += fwd

            self._last_odom_x = x
            self._last_odom_y = y

    # ------------------------------------------------------------------
    # Helper: build a TwistStamped message
    # ------------------------------------------------------------------

    def _make_twist(
        self,
        linear_x: float = 0.0,
        angular_z: float = 0.0,
    ) -> TwistStamped:
        """Return a stamped velocity command with the given components."""
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        return cmd

    # ------------------------------------------------------------------
    # Behaviour methods – return TwistStamped or None
    # ------------------------------------------------------------------
    # Each method must be called while _lock is already held.

    def _halt_behavior(self) -> Optional[TwistStamped]:
        """Priority 1 – stop on bumper collision.

        When a BUMP hazard is active the robot publishes zero velocity.
        This is the highest-priority behaviour; it supersedes everything
        else, including keyboard input.
        """
        if self._collision_detected or self._laser_collision:
            self.get_logger().warn('HALT: collision or laser hazard detected')
            return self._make_twist(0.0, 0.0)
        return None

    def _keyboard_behavior(self) -> Optional[TwistStamped]:
        """Priority 2 – forward a recent non-zero keyboard command.

        The keyboard command expires after KEYBOARD_TIMEOUT_S seconds of
        silence so the robot resumes autonomous behaviour when the operator
        releases the keys.
        """
        if self._keyboard_cmd is None or self._last_key_time is None:
            return None

        elapsed = time.monotonic() - self._last_key_time
        if elapsed > behaviors.KEYBOARD_TIMEOUT_S:
            return None

        twist = self._keyboard_cmd.twist
        if (
            abs(twist.linear.x) > 1e-3
            or abs(twist.linear.y) > 1e-3
            or abs(twist.angular.z) > 1e-3
        ):
            return self._keyboard_cmd

        return None

    def _escape_behavior(self) -> Optional[TwistStamped]:
        """Priority 3 – back away from a symmetric frontal obstacle.

        This is implemented as a fixed-action pattern: once Escape is
        triggered, it continues running until the front obstacle clears,
        rather than dropping out immediately when readings fluctuate.
        """
        if self._scan is None:
            return None

        now = time.monotonic()

        # Keep escaping once triggered until the path is clear.
        if self._escaping:
            if not behaviors.obstacle_in_range(
                self._front_left,
                self._front_right,
            ):
                self._escaping = False
                self._escape_cooldown_until = (
                    now + behaviors.ESCAPE_COOLDOWN_S
                )
                self.get_logger().info('ESCAPE: completed, cooldown active')
                return None

            return self._make_twist(
                behaviors.ESCAPE_BACKUP_SPEED_MPS,
                self._escape_turn_sign * behaviors.TURN_SPEED_RADS,
            )

        # Cooldown prevents immediate escape retrigger oscillation.
        if now < self._escape_cooldown_until:
            return None

        if behaviors.is_symmetric_obstacle(
            self._front_left,
            self._front_right,
        ):
            self._escaping = True
            self._escape_turn_sign = 1.0 if random.random() > 0.5 else -1.0
            self.get_logger().info('ESCAPE: symmetric obstacle in front')
            return self._make_twist(
                behaviors.ESCAPE_BACKUP_SPEED_MPS,
                self._escape_turn_sign * behaviors.TURN_SPEED_RADS,
            )

        return None

    def _avoid_behavior(self) -> Optional[TwistStamped]:
        """Priority 4 – turn away from an asymmetric frontal obstacle.

        When one side of the front sector is clearly closer to an obstacle
        than the other the robot turns in place toward the open side.
        """
        if self._scan is None:
            return None

        # No obstacle at all – this behaviour is inactive.
        if not behaviors.obstacle_in_range(
            self._front_left,
            self._front_right,
        ):
            return None

        # Symmetric obstacles are handled by _escape_behavior (priority 3).
        if behaviors.is_symmetric_obstacle(
            self._front_left,
            self._front_right,
        ):
            return None

        angular_z = behaviors.get_avoid_direction(
            self._front_left,
            self._front_right,
        )
        self.get_logger().info(
            f'AVOID: asymmetric obstacle, angular_z={angular_z:.2f}'
        )
        return self._make_twist(0.0, angular_z)

    def _turn_behavior(self) -> Optional[TwistStamped]:
        """Priority 5 – random re-orientation after 1 ft of forward travel.

        Once the robot has driven TURN_DISTANCE_M metres forward (tracked
        via odometry projection) this behaviour samples a random angle
        from [-15°, +15°] and turns until the required angular displacement
        has elapsed based on turn speed and duration.
        """
        now = time.monotonic()

        if self._turning:
            elapsed = now - self._turn_start_wall
            if elapsed < self._turn_duration:
                # Still turning – keep going.
                return self._make_twist(
                    0.0, self._turn_sign * behaviors.TURN_SPEED_RADS
                )
            # Turn complete.
            self._turning = False

        if self._distance_since_turn >= behaviors.TURN_DISTANCE_M:
            angle = behaviors.sample_random_turn_angle()
            self._turn_sign = 1.0 if angle >= 0 else -1.0
            self._turn_duration = (
                abs(angle) / behaviors.TURN_SPEED_RADS
            )
            self._turn_start_wall = now
            self._distance_since_turn = 0.0
            self._turning = True
            self.get_logger().info(
                f'TURN: random turn {math.degrees(angle):.1f}°'
                f' for {self._turn_duration:.2f}s'
            )
            return self._make_twist(
                0.0, self._turn_sign * behaviors.TURN_SPEED_RADS
            )

        return None

    def _drive_behavior(self) -> TwistStamped:
        """Priority 6 (default) – drive straight forward.

        This behaviour always returns a command; it acts as the base layer
        that keeps the robot moving whenever no higher-priority behaviour
        is active.
        """
        return self._make_twist(behaviors.FORWARD_SPEED_MPS, 0.0)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """10 Hz arbitration loop.

        Evaluates each behaviour in priority order and publishes the
        command from the first behaviour that is active.
        """
        with self._lock:
            behavior_pipeline = (
                self._halt_behavior,
                self._keyboard_behavior,
                self._escape_behavior,
                self._avoid_behavior,
                self._turn_behavior,
                self._drive_behavior,
            )

            for behavior in behavior_pipeline:
                cmd = behavior()
                if cmd is not None:
                    self._cmd_vel_pub.publish(cmd)
                    return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Optional[List[str]] = None) -> None:
    """Initialise rclpy, spin the node, then shut down cleanly."""
    rclpy.init(args=args)
    node = ReactiveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
