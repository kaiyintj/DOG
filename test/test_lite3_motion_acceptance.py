import math

import pytest

from semantic_mapping.runtime.lite3_motion_acceptance import (
    MotionAcceptanceConfig,
    Pose2D,
    VelocityExtrema,
    evaluate_motion_phase,
    evaluate_velocity_direction,
)


def test_forward_is_evaluated_in_the_starting_body_frame():
    result = evaluate_motion_phase(
        'forward',
        Pose2D(x=1.0, y=2.0, yaw=math.pi / 2.0),
        Pose2D(x=1.0, y=2.5, yaw=math.pi / 2.0),
        MotionAcceptanceConfig(),
    )

    assert result.passed
    assert math.isclose(result.forward_m, 0.5, abs_tol=1e-9)
    assert math.isclose(result.left_m, 0.0, abs_tol=1e-9)
    assert math.isclose(result.yaw_rad, 0.0, abs_tol=1e-9)


@pytest.mark.parametrize(
    ('name', 'end', 'expected_forward', 'expected_left'),
    [
        ('forward', Pose2D(0.5, 0.0, 0.0), 0.5, 0.0),
        ('backward', Pose2D(-0.5, 0.0, 0.0), -0.5, 0.0),
        ('left', Pose2D(0.0, 0.5, 0.0), 0.0, 0.5),
        ('right', Pose2D(0.0, -0.5, 0.0), 0.0, -0.5),
    ],
)
def test_translation_phases_require_the_ros_body_axis_direction(
        name, end, expected_forward, expected_left):
    result = evaluate_motion_phase(
        name,
        Pose2D(0.0, 0.0, 0.0),
        end,
        MotionAcceptanceConfig(),
    )

    assert result.passed
    assert result.forward_m == expected_forward
    assert result.left_m == expected_left


def test_forward_rejects_a_half_metre_move_in_reverse():
    result = evaluate_motion_phase(
        'forward',
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(-0.5, 0.0, 0.0),
        MotionAcceptanceConfig(),
    )

    assert not result.passed


@pytest.mark.parametrize(
    ('name', 'start_yaw', 'end_yaw', 'expected_yaw'),
    [
        ('turn_left', math.radians(170.0), math.radians(-145.0), math.pi / 4),
        ('turn_right', 0.0, -math.pi / 4, -math.pi / 4),
    ],
)
def test_turn_phases_accept_ros_yaw_direction_across_angle_wrap(
        name, start_yaw, end_yaw, expected_yaw):
    result = evaluate_motion_phase(
        name,
        Pose2D(0.0, 0.0, start_yaw),
        Pose2D(0.0, 0.0, end_yaw),
        MotionAcceptanceConfig(),
    )

    assert result.passed
    assert math.isclose(result.yaw_rad, expected_yaw, abs_tol=1e-9)


def test_translation_rejects_a_large_heading_change():
    result = evaluate_motion_phase(
        'forward',
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.5, 0.0, math.pi / 2.0),
        MotionAcceptanceConfig(),
    )

    assert not result.passed


@pytest.mark.parametrize(
    ('name', 'extrema'),
    [
        ('forward', VelocityExtrema(linear_x_max=0.20)),
        ('backward', VelocityExtrema(linear_x_min=-0.20)),
        ('left', VelocityExtrema(linear_y_max=0.20)),
        ('right', VelocityExtrema(linear_y_min=-0.20)),
        ('turn_left', VelocityExtrema(angular_z_max=0.40)),
        ('turn_right', VelocityExtrema(angular_z_min=-0.40)),
    ],
)
def test_velocity_direction_uses_ros_body_twist_signs(name, extrema):
    assert evaluate_velocity_direction(
        name,
        extrema,
        MotionAcceptanceConfig(),
    )
