"""Evaluate manually commanded Lite3 motion in its starting body frame."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose2D:
    """Planar robot pose."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class MotionAcceptanceConfig:
    """Bounds for the approximate half-metre manual motion sequence."""

    translation_min_m: float = 0.25
    translation_max_m: float = 0.80
    translation_cross_track_max_m: float = 0.20
    translation_yaw_drift_max_rad: float = math.radians(15.0)
    turn_min_rad: float = math.radians(20.0)
    turn_max_rad: float = math.radians(75.0)
    turn_translation_max_m: float = 0.20
    linear_speed_min_mps: float = 0.03
    angular_speed_min_radps: float = 0.10


@dataclass(frozen=True)
class VelocityExtrema:
    """Minimum and maximum planar body velocities observed in one phase."""

    linear_x_min: float = 0.0
    linear_x_max: float = 0.0
    linear_y_min: float = 0.0
    linear_y_max: float = 0.0
    angular_z_min: float = 0.0
    angular_z_max: float = 0.0


@dataclass(frozen=True)
class MotionPhaseResult:
    """Body-frame displacement and acceptance decision for one phase."""

    name: str
    forward_m: float
    left_m: float
    yaw_rad: float
    passed: bool


def _wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def evaluate_motion_phase(name, start, end, config):
    """Evaluate one manual motion from its starting body frame."""
    dx_world = end.x - start.x
    dy_world = end.y - start.y
    cosine = math.cos(start.yaw)
    sine = math.sin(start.yaw)
    forward_m = cosine * dx_world + sine * dy_world
    left_m = -sine * dx_world + cosine * dy_world
    yaw_rad = _wrapped_angle(end.yaw - start.yaw)
    axes = {
        'forward': (forward_m, left_m),
        'backward': (-forward_m, left_m),
        'left': (left_m, forward_m),
        'right': (-left_m, forward_m),
    }
    if name in axes:
        primary_m, cross_track_m = axes[name]
        passed = (
            config.translation_min_m <= primary_m
            <= config.translation_max_m
            and abs(cross_track_m)
            <= config.translation_cross_track_max_m
            and abs(yaw_rad) <= config.translation_yaw_drift_max_rad
        )
    elif name in ('turn_left', 'turn_right'):
        signed_yaw = yaw_rad if name == 'turn_left' else -yaw_rad
        passed = (
            config.turn_min_rad <= signed_yaw <= config.turn_max_rad
            and math.hypot(forward_m, left_m)
            <= config.turn_translation_max_m
        )
    else:
        raise ValueError('unsupported motion phase: {}'.format(name))
    return MotionPhaseResult(
        name=name,
        forward_m=forward_m,
        left_m=left_m,
        yaw_rad=yaw_rad,
        passed=passed,
    )


def evaluate_velocity_direction(name, extrema, config):
    """Return whether body velocity reached the expected signed direction."""
    signed_extrema = {
        'forward': extrema.linear_x_max,
        'backward': -extrema.linear_x_min,
        'left': extrema.linear_y_max,
        'right': -extrema.linear_y_min,
        'turn_left': extrema.angular_z_max,
        'turn_right': -extrema.angular_z_min,
    }
    if name not in signed_extrema:
        raise ValueError('unsupported motion phase: {}'.format(name))
    minimum = (
        config.angular_speed_min_radps
        if name.startswith('turn_')
        else config.linear_speed_min_mps
    )
    return signed_extrema[name] >= minimum
