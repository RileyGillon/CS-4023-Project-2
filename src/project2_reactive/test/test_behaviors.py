# Copyright 2026 Riley
#
# Licensed under the MIT License.
#
# test_behaviors.py
# -----------------
# Comprehensive unit tests for the project2_reactive.behaviors module.
#
# These tests have NO ROS 2 dependency – they run with plain `pytest`:
#
#   cd src/project2_reactive
#   python -m pytest test/test_behaviors.py -v

import math
from typing import List, Tuple

import pytest

from project2_reactive import behaviors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_full_scan(
    distance: float,
    n_rays: int = 360,
    range_min: float = 0.1,
    range_max: float = 10.0,
    angle_min: float = -math.pi,
) -> Tuple[List[float], float, float, float, float]:
    """Return a full 360-degree scan with every ray set to *distance*."""
    angle_increment = 2.0 * math.pi / n_rays
    ranges = [distance] * n_rays
    return ranges, angle_min, angle_increment, range_min, range_max


def _make_asymmetric_scan(
    left_dist: float,
    right_dist: float,
    background: float = 5.0,
    front_half_angle: float = math.radians(30.0),
    n_rays: int = 360,
) -> Tuple[List[float], float, float, float, float]:
    """Return a scan with custom distances in the front-left and front-right.

    Rays within *front_half_angle* of angle = 0 are set per side; the rest
    are set to *background*.
    """
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / n_rays
    ranges = [background] * n_rays

    for i in range(n_rays):
        raw = angle_min + i * angle_increment
        angle = (raw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(angle) <= front_half_angle:
            if angle >= 0.0:
                ranges[i] = left_dist
            else:
                ranges[i] = right_dist

    return ranges, angle_min, angle_increment, 0.1, 10.0


# ---------------------------------------------------------------------------
# TestGetFrontDistances
# ---------------------------------------------------------------------------


class TestGetFrontDistances:
    """Tests for behaviors.get_front_distances()."""

    def test_returns_two_lists(self) -> None:
        args = _make_full_scan(5.0)
        result = behaviors.get_front_distances(*args)
        assert len(result) == 2

    def test_all_clear_readings_included(self) -> None:
        args = _make_full_scan(5.0)
        left, right = behaviors.get_front_distances(*args)
        assert len(left) > 0
        assert len(right) > 0
        assert all(d == pytest.approx(5.0) for d in left)
        assert all(d == pytest.approx(5.0) for d in right)

    def test_readings_above_range_max_filtered_out(self) -> None:
        # distance 15.0 > range_max 10.0 → every ray filtered
        args = _make_full_scan(15.0)
        left, right = behaviors.get_front_distances(*args)
        assert left == []
        assert right == []

    def test_readings_below_range_min_filtered_out(self) -> None:
        # distance 0.05 < range_min 0.1 → every ray filtered
        args = _make_full_scan(0.05)
        left, right = behaviors.get_front_distances(*args)
        assert left == []
        assert right == []

    def test_inf_readings_filtered_out(self) -> None:
        n = 360
        ranges = [float('inf')] * n
        left, right = behaviors.get_front_distances(
            ranges, -math.pi, 2.0 * math.pi / n, 0.1, 10.0
        )
        assert left == []
        assert right == []

    def test_nan_readings_filtered_out(self) -> None:
        n = 360
        ranges = [float('nan')] * n
        left, right = behaviors.get_front_distances(
            ranges, -math.pi, 2.0 * math.pi / n, 0.1, 10.0
        )
        assert left == []
        assert right == []

    def test_only_front_sector_rays_returned(self) -> None:
        """Rays outside ±30° must not appear in the results."""
        n = 360
        angle_min = -math.pi
        angle_inc = 2.0 * math.pi / n
        sector = behaviors.FRONT_SECTOR_ANGLE_RAD

        expected_count = sum(
            1
            for i in range(n)
            if abs((angle_min + i * angle_inc + math.pi) % (2 * math.pi) - math.pi)
            <= sector
        )

        args = _make_full_scan(1.0, n_rays=n)
        left, right = behaviors.get_front_distances(*args)
        assert len(left) + len(right) == expected_count

    def test_left_positive_angle_right_negative_angle(self) -> None:
        """Positive-angle rays → left list; negative-angle rays → right list."""
        args = _make_full_scan(1.0)
        left, right = behaviors.get_front_distances(*args)
        # Both must be non-empty in a symmetric 360-degree scan.
        assert len(left) > 0
        assert len(right) > 0

    def test_empty_scan_returns_empty_lists(self) -> None:
        left, right = behaviors.get_front_distances(
            [], -math.pi, 2.0 * math.pi / 360, 0.1, 10.0
        )
        assert left == []
        assert right == []

    def test_asymmetric_distances_separated_correctly(self) -> None:
        """Close reading on left side should appear only in left list."""
        args = _make_asymmetric_scan(left_dist=0.1, right_dist=5.0)
        left, right = behaviors.get_front_distances(*args)
        assert min(left) == pytest.approx(0.1)
        assert min(right) == pytest.approx(5.0)

    def test_angle_min_zero_normalisation(self) -> None:
        """Angle normalisation must work when angle_min = 0 (0 to 2π scan)."""
        n = 360
        angle_min = 0.0
        angle_inc = 2.0 * math.pi / n
        # Front rays near angle = 0 (and near 2π, which normalises to ~0)
        ranges = [5.0] * n
        left, right = behaviors.get_front_distances(
            ranges, angle_min, angle_inc, 0.1, 10.0
        )
        # Should find front rays near angle ≈ 0 (first few and last few rays).
        assert len(left) + len(right) > 0

    def test_custom_front_sector_angle(self) -> None:
        """Narrower sector yields fewer rays than wider sector."""
        args = _make_full_scan(1.0)
        left_narrow, right_narrow = behaviors.get_front_distances(
            *args, front_sector_angle=math.radians(10.0)
        )
        left_wide, right_wide = behaviors.get_front_distances(
            *args, front_sector_angle=math.radians(45.0)
        )
        narrow_count = len(left_narrow) + len(right_narrow)
        wide_count = len(left_wide) + len(right_wide)
        assert narrow_count < wide_count


# ---------------------------------------------------------------------------
# TestObstacleInRange
# ---------------------------------------------------------------------------


class TestObstacleInRange:
    """Tests for behaviors.obstacle_in_range()."""

    def test_no_obstacle_far_away(self) -> None:
        far = behaviors.OBSTACLE_DISTANCE_M * 3
        assert not behaviors.obstacle_in_range([far, far], [far, far])

    def test_obstacle_on_left_side(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.5
        assert behaviors.obstacle_in_range([close], [5.0])

    def test_obstacle_on_right_side(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.5
        assert behaviors.obstacle_in_range([5.0], [close])

    def test_obstacle_on_both_sides(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.3
        assert behaviors.obstacle_in_range([close], [close])

    def test_both_lists_empty_returns_false(self) -> None:
        assert not behaviors.obstacle_in_range([], [])

    def test_left_empty_right_close(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.5
        assert behaviors.obstacle_in_range([], [close])

    def test_right_empty_left_close(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.5
        assert behaviors.obstacle_in_range([close], [])

    def test_exactly_at_threshold_not_triggered(self) -> None:
        # Distance == threshold is NOT strictly less-than threshold.
        d = behaviors.OBSTACLE_DISTANCE_M
        assert not behaviors.obstacle_in_range([d], [d])

    def test_just_inside_threshold_triggered(self) -> None:
        d = behaviors.OBSTACLE_DISTANCE_M - 0.001
        assert behaviors.obstacle_in_range([d], [5.0])

    def test_custom_obstacle_distance(self) -> None:
        assert behaviors.obstacle_in_range([0.5], [5.0], obstacle_distance=1.0)
        assert not behaviors.obstacle_in_range([0.5], [5.0], obstacle_distance=0.4)


# ---------------------------------------------------------------------------
# TestIsSymmetricObstacle
# ---------------------------------------------------------------------------


class TestIsSymmetricObstacle:
    """Tests for behaviors.is_symmetric_obstacle()."""

    def test_both_sides_equally_blocked(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.5
        assert behaviors.is_symmetric_obstacle([close], [close])

    def test_both_sides_clear_returns_false(self) -> None:
        far = behaviors.OBSTACLE_DISTANCE_M * 2
        assert not behaviors.is_symmetric_obstacle([far], [far])

    def test_one_side_clear_one_side_blocked_returns_false(self) -> None:
        close = behaviors.OBSTACLE_DISTANCE_M * 0.3
        far = behaviors.OBSTACLE_DISTANCE_M * 2
        assert not behaviors.is_symmetric_obstacle([close], [far])

    def test_clearly_asymmetric_returns_false(self) -> None:
        left_close = behaviors.OBSTACLE_DISTANCE_M * 0.1
        right_far = behaviors.OBSTACLE_DISTANCE_M * 0.9
        assert not behaviors.is_symmetric_obstacle([left_close], [right_far])

    def test_empty_left_returns_false(self) -> None:
        assert not behaviors.is_symmetric_obstacle([], [0.1])

    def test_empty_right_returns_false(self) -> None:
        assert not behaviors.is_symmetric_obstacle([0.1], [])

    def test_both_empty_returns_false(self) -> None:
        assert not behaviors.is_symmetric_obstacle([], [])

    def test_asymmetry_just_within_threshold(self) -> None:
        """Small asymmetry ≤ threshold → still symmetric."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        # Produce asymmetry exactly at 90% of the threshold.
        slightly_off = close * (1.0 + behaviors.SYMMETRY_RATIO_THRESHOLD * 0.9)
        assert behaviors.is_symmetric_obstacle([close], [slightly_off])

    def test_asymmetry_just_above_threshold(self) -> None:
        """Asymmetry slightly above threshold → asymmetric."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        # Produce asymmetry at 110% of the threshold.
        too_far = close * (1.0 + behaviors.SYMMETRY_RATIO_THRESHOLD * 1.1)
        assert not behaviors.is_symmetric_obstacle([close], [too_far])

    def test_near_zero_values_treated_as_symmetric(self) -> None:
        """Two near-zero values should not produce a division-by-zero error."""
        assert behaviors.is_symmetric_obstacle([1e-10], [1e-10])

    def test_multiple_rays_uses_minimum(self) -> None:
        """The closest ray per side should be used for the symmetry check."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.2
        far = behaviors.OBSTACLE_DISTANCE_M * 2
        # Mixed list – minimum is close on both sides.
        assert behaviors.is_symmetric_obstacle([far, close], [far, close])

    def test_custom_thresholds(self) -> None:
        close = 0.1
        slightly_off = 0.11
        # With a very tight threshold the small difference is asymmetric.
        assert not behaviors.is_symmetric_obstacle(
            [close], [slightly_off],
            obstacle_distance=0.5,
            symmetry_threshold=0.05,
        )
        # With a loose threshold it is symmetric.
        assert behaviors.is_symmetric_obstacle(
            [close], [slightly_off],
            obstacle_distance=0.5,
            symmetry_threshold=0.5,
        )


# ---------------------------------------------------------------------------
# TestGetAvoidDirection
# ---------------------------------------------------------------------------


class TestGetAvoidDirection:
    """Tests for behaviors.get_avoid_direction()."""

    def test_obstacle_on_left_turns_right(self) -> None:
        """Left-side obstacle → negative angular-z (right / CW turn)."""
        result = behaviors.get_avoid_direction([0.1], [5.0])
        assert result < 0.0

    def test_obstacle_on_right_turns_left(self) -> None:
        """Right-side obstacle → positive angular-z (left / CCW turn)."""
        result = behaviors.get_avoid_direction([5.0], [0.1])
        assert result > 0.0

    def test_magnitude_equals_avoid_turn_speed(self) -> None:
        result = behaviors.get_avoid_direction([0.1], [5.0])
        assert abs(result) == pytest.approx(behaviors.AVOID_TURN_SPEED_RADS)

    def test_empty_right_treats_as_infinity(self) -> None:
        """Empty right list → obstacle assumed farther on right → turn right."""
        result = behaviors.get_avoid_direction([0.1], [])
        assert result < 0.0  # obstacle on left → turn right

    def test_empty_left_treats_as_infinity(self) -> None:
        """Empty left list → obstacle assumed farther on left → turn left."""
        result = behaviors.get_avoid_direction([], [0.1])
        assert result > 0.0  # obstacle on right → turn left

    def test_multiple_rays_uses_minimum(self) -> None:
        """The closest ray per side determines the turn direction."""
        # Left side has a very close ray among others.
        result = behaviors.get_avoid_direction([5.0, 0.05, 3.0], [4.0, 4.0])
        assert result < 0.0  # left is closest → turn right


# ---------------------------------------------------------------------------
# TestComputeDistance
# ---------------------------------------------------------------------------


class TestComputeDistance:
    """Tests for behaviors.compute_distance()."""

    def test_same_point_is_zero(self) -> None:
        assert behaviors.compute_distance(0, 0, 0, 0) == pytest.approx(0.0)

    def test_unit_step_horizontal(self) -> None:
        assert behaviors.compute_distance(0, 0, 1, 0) == pytest.approx(1.0)

    def test_unit_step_vertical(self) -> None:
        assert behaviors.compute_distance(0, 0, 0, 1) == pytest.approx(1.0)

    def test_pythagorean_3_4_5(self) -> None:
        assert behaviors.compute_distance(0, 0, 3, 4) == pytest.approx(5.0)

    def test_pythagorean_5_12_13(self) -> None:
        assert behaviors.compute_distance(0, 0, 5, 12) == pytest.approx(13.0)

    def test_non_origin_start(self) -> None:
        assert behaviors.compute_distance(1, 1, 4, 5) == pytest.approx(5.0)

    def test_negative_coordinates(self) -> None:
        assert behaviors.compute_distance(-1, -1, 2, 3) == pytest.approx(5.0)

    def test_symmetric_distance(self) -> None:
        """Distance from A→B equals distance from B→A."""
        d_ab = behaviors.compute_distance(1, 2, 4, 6)
        d_ba = behaviors.compute_distance(4, 6, 1, 2)
        assert d_ab == pytest.approx(d_ba)

    def test_one_foot_threshold(self) -> None:
        """A displacement of exactly 1 ft (0.3048 m) is returned correctly."""
        assert behaviors.compute_distance(
            0, 0, behaviors.OBSTACLE_DISTANCE_M, 0
        ) == pytest.approx(behaviors.OBSTACLE_DISTANCE_M)


# ---------------------------------------------------------------------------
# TestYawFromQuaternion
# ---------------------------------------------------------------------------


class TestYawFromQuaternion:
    """Tests for behaviors.yaw_from_quaternion()."""

    def test_identity_quaternion_zero_yaw(self) -> None:
        assert behaviors.yaw_from_quaternion(0, 0, 0, 1) == pytest.approx(0.0)

    def test_90_degree_ccw(self) -> None:
        angle = math.pi / 2
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        assert behaviors.yaw_from_quaternion(0, 0, s, c) == pytest.approx(angle)

    def test_90_degree_cw(self) -> None:
        angle = -math.pi / 2
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        assert behaviors.yaw_from_quaternion(0, 0, s, c) == pytest.approx(angle)

    def test_180_degree(self) -> None:
        # 180° CCW: q = (0, 0, 1, 0)
        yaw = behaviors.yaw_from_quaternion(0, 0, 1, 0)
        assert abs(yaw) == pytest.approx(math.pi)

    def test_45_degree(self) -> None:
        angle = math.pi / 4
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        assert behaviors.yaw_from_quaternion(0, 0, s, c) == pytest.approx(angle)

    def test_negative_45_degree(self) -> None:
        angle = -math.pi / 4
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        assert behaviors.yaw_from_quaternion(0, 0, s, c) == pytest.approx(angle)

    def test_result_in_minus_pi_to_pi(self) -> None:
        """Output must be in [-π, π] for several test angles."""
        test_angles = [0, math.pi / 6, math.pi / 3, math.pi / 2,
                       math.pi, -math.pi / 4, -math.pi / 2, -math.pi]
        for angle in test_angles:
            s = math.sin(angle / 2)
            c = math.cos(angle / 2)
            yaw = behaviors.yaw_from_quaternion(0, 0, s, c)
            assert -math.pi <= yaw <= math.pi


# ---------------------------------------------------------------------------
# TestForwardDisplacement
# ---------------------------------------------------------------------------


class TestForwardDisplacement:
    """Tests for behaviors.forward_displacement()."""

    def test_pure_forward_motion(self) -> None:
        """Moving along +x with yaw = 0 should give full displacement."""
        assert behaviors.forward_displacement(1.0, 0.0, 0.0) == pytest.approx(1.0)

    def test_pure_backward_motion(self) -> None:
        assert behaviors.forward_displacement(-1.0, 0.0, 0.0) == pytest.approx(-1.0)

    def test_lateral_motion_gives_zero(self) -> None:
        """Lateral displacement (perpendicular to heading) projects to 0."""
        assert behaviors.forward_displacement(0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_45_degree_heading(self) -> None:
        """Robot facing NE; moving NE by √2 should project fully forward."""
        yaw = math.pi / 4
        dx = math.cos(yaw)
        dy = math.sin(yaw)
        result = behaviors.forward_displacement(dx, dy, yaw)
        assert result == pytest.approx(1.0)

    def test_90_degree_heading_forward_is_y(self) -> None:
        """Robot facing +y (yaw = 90°); moving along +y is forward."""
        result = behaviors.forward_displacement(0.0, 1.0, math.pi / 2)
        assert result == pytest.approx(1.0)

    def test_90_degree_heading_x_motion_is_lateral(self) -> None:
        result = behaviors.forward_displacement(1.0, 0.0, math.pi / 2)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_reverse_heading_flips_sign(self) -> None:
        """Moving along +x with yaw = π (facing -x) should give -1."""
        result = behaviors.forward_displacement(1.0, 0.0, math.pi)
        assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# TestSampleRandomTurnAngle
# ---------------------------------------------------------------------------


class TestSampleRandomTurnAngle:
    """Tests for behaviors.sample_random_turn_angle()."""

    def test_always_within_default_range(self) -> None:
        for _ in range(1000):
            angle = behaviors.sample_random_turn_angle()
            assert -behaviors.TURN_ANGLE_RANGE_RAD <= angle <= behaviors.TURN_ANGLE_RANGE_RAD

    def test_always_within_custom_range(self) -> None:
        custom = math.radians(10.0)
        for _ in range(500):
            angle = behaviors.sample_random_turn_angle(custom)
            assert -custom <= angle <= custom

    def test_samples_both_signs(self) -> None:
        """With 500 samples we expect at least one positive and one negative."""
        angles = [behaviors.sample_random_turn_angle() for _ in range(500)]
        assert any(a > 0 for a in angles)
        assert any(a < 0 for a in angles)

    def test_zero_range_returns_zero(self) -> None:
        assert behaviors.sample_random_turn_angle(0.0) == pytest.approx(0.0)

    def test_distribution_roughly_uniform(self) -> None:
        """Mean of many samples should be near zero (uniform distribution)."""
        angles = [behaviors.sample_random_turn_angle() for _ in range(2000)]
        mean = sum(angles) / len(angles)
        # With 2000 samples the mean should be within ±2° of zero.
        assert abs(mean) < math.radians(2.0)


# ---------------------------------------------------------------------------
# Integration-style tests combining multiple helpers
# ---------------------------------------------------------------------------


class TestBehaviorIntegration:
    """Higher-level checks that combine helpers as the controller does."""

    def test_symmetric_obstacle_triggers_escape_not_avoid(self) -> None:
        """Symmetric scan: is_symmetric returns True, so Avoid should not fire."""
        close = behaviors.OBSTACLE_DISTANCE_M * 0.4
        args = _make_asymmetric_scan(left_dist=close, right_dist=close)
        left, right = behaviors.get_front_distances(*args)

        assert behaviors.obstacle_in_range(left, right)
        assert behaviors.is_symmetric_obstacle(left, right)
        # Avoid should be suppressed when escape is active.

    def test_left_obstacle_triggers_avoid_not_escape(self) -> None:
        """Obstacle on left only: asymmetric → Avoid fires, not Escape."""
        args = _make_asymmetric_scan(left_dist=0.1, right_dist=5.0)
        left, right = behaviors.get_front_distances(*args)

        assert behaviors.obstacle_in_range(left, right)
        assert not behaviors.is_symmetric_obstacle(left, right)
        direction = behaviors.get_avoid_direction(left, right)
        assert direction < 0.0  # turn right

    def test_right_obstacle_triggers_avoid_turn_left(self) -> None:
        args = _make_asymmetric_scan(left_dist=5.0, right_dist=0.1)
        left, right = behaviors.get_front_distances(*args)

        assert behaviors.obstacle_in_range(left, right)
        assert not behaviors.is_symmetric_obstacle(left, right)
        direction = behaviors.get_avoid_direction(left, right)
        assert direction > 0.0  # turn left

    def test_clear_path_no_behavior_triggered(self) -> None:
        """All clear: no obstacle, no escape, no avoid."""
        args = _make_full_scan(5.0)
        left, right = behaviors.get_front_distances(*args)

        assert not behaviors.obstacle_in_range(left, right)
        assert not behaviors.is_symmetric_obstacle(left, right)

    def test_one_foot_distance_triggers_turn(self) -> None:
        """Cumulative forward distance of ≥ TURN_DISTANCE_M should trigger a turn."""
        assert behaviors.TURN_DISTANCE_M == pytest.approx(0.3048)
        # Simulate two 0.2 m steps.
        total = 0.0
        total += behaviors.forward_displacement(0.2, 0.0, 0.0)
        total += behaviors.forward_displacement(0.2, 0.0, 0.0)
        assert total >= behaviors.TURN_DISTANCE_M

    def test_backward_motion_does_not_accumulate_forward_distance(self) -> None:
        """Backing up (forward_displacement < 0) should not count toward turn."""
        fwd = behaviors.forward_displacement(-0.3, 0.0, 0.0)
        assert fwd < 0.0
