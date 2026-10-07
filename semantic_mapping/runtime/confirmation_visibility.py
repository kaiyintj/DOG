"""Candidate frustum and sparse observed-depth checks; unknown depth stays unknown."""

import math

import numpy as np
from scipy.spatial import cKDTree


def inspect_view(positions, geometry, max_range_m=20.0, voxel_size=.1):
    map_from_camera = geometry['source_to_map'] @ np.linalg.inv(geometry['lidar_to_camera'])
    camera_from_map = np.linalg.inv(map_from_camera)
    points = np.asarray(positions)
    camera = points @ camera_from_map[:3, :3].T + camera_from_map[:3, 3]
    depth = camera[:, 2]
    matrix = geometry['camera_matrix']
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = np.column_stack((matrix[0, 0] * camera[:, 0] / depth + matrix[0, 2],
                                  matrix[1, 1] * camera[:, 1] / depth + matrix[1, 2]))
    height, width = geometry['image_shape']
    in_view = ((depth > .1) & (np.linalg.norm(camera, axis=1) <= max_range_m)
               & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    measured_pixels = np.column_stack((geometry['pixel_u'], geometry['pixel_v']))
    tree = cKDTree(measured_pixels) if len(measured_pixels) else None
    hidden = supported = 0
    for index in np.flatnonzero(in_view):
        # A stored voxel center is displaced from its sampled surface point.
        # Match a projected voxel footprint, rather than a fixed two-pixel dot.
        radius = max(2.0, float(matrix[0, 0]) * voxel_size / depth[index])
        neighbors = [] if tree is None else tree.query_ball_point(pixels[index], radius)
        if not neighbors:
            continue
        measured_depth = float(np.min(geometry['camera_depth'][neighbors]))
        hidden += int(depth[index] > measured_depth + .15)
        supported += int(abs(depth[index] - measured_depth) <= .15)
    center = points.mean(axis=0)
    delta = center - map_from_camera[:3, 3]
    forward = map_from_camera[:3, 2]
    camera_heading = math.atan2(forward[1], forward[0])
    bearing = math.atan2(delta[1], delta[0])
    error = math.atan2(math.sin(bearing - camera_heading), math.cos(bearing - camera_heading))
    count = max(len(points), 1)
    return {'in_view_ratio': float(np.mean(in_view)), 'occluded_ratio': hidden / count,
            'surface_supported_ratio': supported / count,
            'unknown_depth_ratio': (int(in_view.sum()) - hidden - supported) / count,
            'camera_heading': camera_heading, 'bearing_error_rad': error,
            'visible_for_attempt': bool(np.mean(in_view) >= .5 and hidden / count < .5),
            'visible_support': bool(supported / count >= .5)}
