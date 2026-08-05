import math

import pytest

from semantic_mapping.active_perception_node import scale_velocity_command


def test_uncertainty_scaling_preserves_command_curvature():
    linear_x, linear_y, angular_z = scale_velocity_command(
        0.2,
        0.0,
        0.4,
        0.3,
        linear_deadband=0.01,
        angular_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert linear_x == pytest.approx(0.06)
    assert linear_y == 0.0
    assert angular_z == pytest.approx(0.12)
    assert linear_x / angular_z == pytest.approx(0.5)


def test_deadband_is_applied_before_scaling():
    linear_x, _, _ = scale_velocity_command(
        0.02,
        0.0,
        0.0,
        0.3,
        linear_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert linear_x == pytest.approx(0.006)


def test_pure_rotation_remains_pure_rotation():
    linear_x, linear_y, angular_z = scale_velocity_command(
        0.0,
        0.0,
        0.4,
        0.3,
        linear_deadband=0.01,
        angular_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert math.hypot(linear_x, linear_y) == 0.0
    assert angular_z == pytest.approx(0.12)
