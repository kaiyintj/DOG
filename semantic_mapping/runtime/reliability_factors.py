#!/usr/bin/env python3
"""ROS-independent reliability factors shared by mapping and benchmarks."""

import numpy as np
from scipy.spatial import cKDTree


def compute_motion_reliability(
    angular_norms,
    acceleration_norms,
    angular_scale,
    acceleration_scale,
    gravity,
    min_reliability,
):
    """Return motion reliability and the two RMS motion statistics."""
    angular_norms = np.asarray(angular_norms, dtype=np.float64).reshape(-1)
    acceleration_norms = np.asarray(
        acceleration_norms, dtype=np.float64).reshape(-1)
    if angular_norms.size == 0 or acceleration_norms.size == 0:
        raise ValueError('motion samples must not be empty')
    if angular_norms.shape != acceleration_norms.shape:
        raise ValueError(
            'angular and acceleration samples must have equal length')

    angular_rms = float(np.sqrt(np.mean(np.square(angular_norms))))
    acceleration_deviation = float(np.sqrt(np.mean(np.square(
        acceleration_norms - float(gravity),
    ))))
    angular_scale = max(float(angular_scale), 1e-6)
    acceleration_scale = max(float(acceleration_scale), 1e-6)
    exponent = -0.5 * (
        (angular_rms / angular_scale) ** 2
        + (acceleration_deviation / acceleration_scale) ** 2
    )
    reliability = float(np.clip(
        np.exp(exponent), float(min_reliability), 1.0))
    return reliability, angular_rms, acceleration_deviation


def compute_temporal_motion_reliability(
    relative_rotation_rad,
    relative_translation_m,
    rotation_scale_rad,
    translation_scale_m,
    min_reliability,
):
    """
    Return Motion V2 reliability from inter-sensor-time pose change.

    This factor measures the temporal registration risk between the camera
    and LiDAR observation times.  The older IMU-RMS factor remains available
    through :func:`compute_motion_reliability` for diagnostics, but it is not
    part of this calculation.
    """
    relative_rotation_rad = float(relative_rotation_rad)
    relative_translation_m = float(relative_translation_m)
    if not np.isfinite(relative_rotation_rad):
        raise ValueError('relative_rotation_rad must be finite')
    if not np.isfinite(relative_translation_m):
        raise ValueError('relative_translation_m must be finite')

    relative_rotation_rad = max(relative_rotation_rad, 0.0)
    relative_translation_m = max(relative_translation_m, 0.0)
    rotation_scale_rad = max(float(rotation_scale_rad), 1e-6)
    translation_scale_m = max(float(translation_scale_m), 1e-6)
    exponent = -0.5 * (
        (relative_rotation_rad / rotation_scale_rad) ** 2
        + (relative_translation_m / translation_scale_m) ** 2
    )
    return float(np.clip(
        np.exp(exponent),
        float(min_reliability),
        1.0,
    ))


def compute_local_point_density(points, neighborhood_radius=0.3):
    """Count 3D neighbors, including each query point itself."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(neighborhood_radius))
    return np.asarray([len(neighbor) for neighbor in neighbors], dtype=np.int64)


def compute_density_reliability(density, density_scale):
    """Return the current saturating local-density reliability."""
    density = np.asarray(density)
    return 1.0 - np.exp(-density / max(float(density_scale), 1e-6))


def compute_range_reliability(sensor_ranges, range_scale_m):
    """Return the current squared-exponential range reliability."""
    sensor_ranges = np.asarray(sensor_ranges)
    return np.exp(-np.square(
        sensor_ranges / max(float(range_scale_m), 1e-6)))


def compute_normalized_view_radius(
    pixel_u,
    pixel_v,
    image_width,
    image_height,
):
    """Return normalized radial image position used by the runtime node."""
    pixel_u = np.asarray(pixel_u)
    pixel_v = np.asarray(pixel_v)
    if pixel_u.shape != pixel_v.shape:
        raise ValueError('pixel_u and pixel_v must have identical shapes')
    normalized_u = (
        pixel_u - 0.5 * image_width
    ) / max(0.5 * image_width, 1.0)
    normalized_v = (
        pixel_v - 0.5 * image_height
    ) / max(0.5 * image_height, 1.0)
    return np.clip(
        np.sqrt(normalized_u ** 2 + normalized_v ** 2) / np.sqrt(2.0),
        0.0,
        1.0,
    )


def compute_view_reliability(view_radius, view_edge_penalty):
    """Return the current quadratic image-edge reliability."""
    view_radius = np.asarray(view_radius)
    return np.clip(
        1.0 - float(view_edge_penalty) * view_radius ** 2,
        0.05,
        1.0,
    )


def compute_semantic_reliability(logits, num_classes, confidence_floor):
    """Return posterior, entropy and normalized-entropy reliability."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / np.maximum(
        np.sum(exp_logits, axis=-1, keepdims=True), 1e-12)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-10, 1.0)),
        axis=-1,
    )
    certainty = 1.0 - entropy / max(np.log(int(num_classes)), 1e-9)
    reliability = (
        float(confidence_floor)
        + (1.0 - float(confidence_floor)) * certainty
    )
    return probabilities, entropy, reliability


def combine_reliability(motion, density, range_, view, semantic):
    """Multiply the five runtime factors and return float32 weights."""
    return np.asarray(
        motion * density * range_ * view * semantic,
    ).astype(np.float32)
