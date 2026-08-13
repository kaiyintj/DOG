"""Tests for runtime-equivalent reliability voxel ablations."""

import numpy as np

from semantic_mapping.carla.reliability_voxel_ablation import (
    ABLATION_NAMES,
    VoxelAblation,
    ablation_weights,
    voxel_map_kwargs_from_params,
)


def test_ablation_weights_add_one_factor_at_a_time():
    """The six conditions are the intended progressive products."""
    weights = ablation_weights(
        np.array([0.5, 0.25]),
        np.array([0.8, 0.4]),
        np.array([0.5, 0.5]),
        np.array([0.25, 1.0]),
        np.array([1.0, 0.5]),
    )

    assert tuple(weights) == ABLATION_NAMES
    np.testing.assert_allclose(weights['none'], [1.0, 1.0])
    np.testing.assert_allclose(weights['semantic'], [0.5, 0.25])
    np.testing.assert_allclose(weights['semantic_range'], [0.4, 0.1])
    np.testing.assert_allclose(weights['semantic_range_density'], [0.2, 0.05])
    np.testing.assert_allclose(
        weights['semantic_range_density_view'], [0.05, 0.05])
    np.testing.assert_allclose(weights['full'], [0.05, 0.025])


def test_zero_observation_weight_matches_runtime_none_semantics():
    """A zero YAML cap asks VoxelMap to derive its own finite cap."""
    kwargs = voxel_map_kwargs_from_params({
        'ros__parameters': {
            'num_classes': 3,
            'voxel_size': 0.2,
            'max_observation_weight': 0.0,
        },
    }, class_colors=np.zeros((3, 3), dtype=np.uint8))

    assert kwargs['K'] == 3
    assert kwargs['voxel_size'] == 0.2
    assert kwargs['max_observation_weight'] is None


def test_report_counts_missing_gt_voxels_as_wrong_in_all_accuracy():
    """A low reliability condition can omit GT voxels and must be penalized."""
    ablation = VoxelAblation(
        {
            'num_classes': 2,
            'voxel_size': 1.0,
            'evidence_decay': 1.0,
            'max_frame_evidence': 1.0,
        },
        class_colors=np.zeros((2, 3), dtype=np.uint8),
    )
    points = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32)
    logits = np.array([[8.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    gt_ids = np.array([0, 1], dtype=np.int64)
    weights = {name: np.ones(2, dtype=np.float32) for name in ABLATION_NAMES}
    weights['full'][1] = 0.0

    ablation.update(points, logits, gt_ids, weights, timestamp_sec=1.0)
    report = ablation.report()

    assert report['none']['gt_voxel_count'] == 2
    assert report['none']['covered_voxel_count'] == 2
    assert report['none']['covered_accuracy'] == 1.0
    assert report['none']['all_gt_accuracy'] == 1.0
    assert report['full']['covered_voxel_count'] == 1
    assert report['full']['missing_voxel_count'] == 1
    assert report['full']['coverage'] == 0.5
    assert report['full']['covered_accuracy'] == 1.0
    assert report['full']['all_gt_accuracy'] == 0.5


def test_negative_ground_truth_ids_do_not_create_reference_voxels():
    """Unsupported semantic-LiDAR labels stay outside primary metrics."""
    ablation = VoxelAblation(
        {'num_classes': 2, 'voxel_size': 1.0, 'evidence_decay': 1.0},
        class_colors=np.zeros((2, 3), dtype=np.uint8),
    )
    points = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    logits = np.array([[8.0, 0.0]], dtype=np.float32)
    weights = {name: np.ones(1, dtype=np.float32) for name in ABLATION_NAMES}

    ablation.update(points, logits, np.array([-1]), weights, timestamp_sec=1.0)

    assert ablation.report()['full']['gt_voxel_count'] == 0
    assert ablation.report()['full']['coverage'] is None
