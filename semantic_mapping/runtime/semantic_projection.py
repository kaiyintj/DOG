#!/usr/bin/env python3
"""ROS-independent pinhole projection helpers for semantic observations."""

import numpy as np


def scale_camera_matrix(
    camera_matrix,
    calibration_width,
    calibration_height,
    image_width,
    image_height,
):
    """Scale one pinhole matrix when calibration and image sizes differ."""
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    dimensions = (
        calibration_width,
        calibration_height,
        image_width,
        image_height,
    )
    if any(int(value) <= 0 for value in dimensions):
        return matrix
    scale_x = float(image_width) / float(calibration_width)
    scale_y = float(image_height) / float(calibration_height)
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    matrix[2, :] = [0.0, 0.0, 1.0]
    return matrix


def project_points_pinhole(
    points_3d,
    lidar_to_camera,
    camera_matrix,
    image_height,
    image_width,
    calibration_width=0,
    calibration_height=0,
    min_camera_depth=0.1,
):
    """Project LiDAR points and return the legacy mask and integer pixels."""
    points_3d = np.asarray(points_3d)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError('points_3d must have shape (N, 3)')
    transform = np.asarray(lidar_to_camera, dtype=np.float64).reshape(4, 4)
    count = len(points_3d)
    homogeneous = np.hstack([points_3d, np.ones((count, 1))])
    camera_points = (transform @ homogeneous.T).T
    valid_depth = camera_points[:, 2] > float(min_camera_depth)
    scaled_matrix = scale_camera_matrix(
        camera_matrix,
        calibration_width,
        calibration_height,
        image_width,
        image_height,
    )
    pixel_u = (
        scaled_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2]
        + scaled_matrix[0, 2]
    ).astype(int)
    pixel_v = (
        scaled_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2]
        + scaled_matrix[1, 2]
    ).astype(int)
    valid_pixel = (
        (pixel_u >= 0)
        & (pixel_u < image_width)
        & (pixel_v >= 0)
        & (pixel_v < image_height)
    )
    return valid_depth & valid_pixel, pixel_u, pixel_v
