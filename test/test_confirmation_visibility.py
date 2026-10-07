"""Missing depth permits an attempt but cannot confirm regional visibility."""

import numpy as np

from semantic_mapping.runtime.confirmation_visibility import inspect_view


def geometry(pixels=(), depth=()):
    pixels = np.asarray(pixels).reshape(-1, 2)
    return {'source_to_map': np.eye(4), 'lidar_to_camera': np.eye(4),
            'camera_matrix': np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]]),
            'image_shape': (100, 100), 'pixel_u': pixels[:, 0], 'pixel_v': pixels[:, 1],
            'camera_depth': np.asarray(depth), 'source_frame': 'lidar'}


def test_unknown_depth_is_not_visible_support():
    view = inspect_view([[0, 0, 2]], geometry())
    assert view['visible_for_attempt']
    assert not view['visible_support']
    assert view['unknown_depth_ratio'] == 1


def test_foreground_occludes_old_candidate_but_matching_surface_supports_it():
    assert not inspect_view([[0, 0, 2]], geometry([[50, 50]], [1]))['visible_support']
    view = inspect_view([[.05, 0, 2]], geometry([[50, 50]], [2]))
    assert view['visible_support']  # Quantized voxel footprint includes the real sample.
    assert view['surface_supported_ratio'] == 1


def test_out_of_frustum_points_do_not_claim_unoccluded_visibility():
    view = inspect_view([[2, 0, 1]], geometry([[50, 50]], [1]))
    assert view['in_view_ratio'] == 0
    assert not view['visible_support']
