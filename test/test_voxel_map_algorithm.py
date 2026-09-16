"""Regression tests for time-aware Dirichlet voxel fusion."""

import numpy as np

from semantic_mapping.runtime.voxel_map import VoxelMap


def test_repeated_consistent_observations_increase_class_probability():
    voxel_map = VoxelMap(
        voxel_size=0.2,
        K=3,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    points = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
    reliability = np.ones(1, dtype=np.float32)
    logits = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)

    key = voxel_map.get_voxel_indices(points[0])
    initial_probability = voxel_map.get_probabilities(key)[2]
    for _ in range(8):
        voxel_map.update(points, reliability, logits)

    probabilities = voxel_map.get_probabilities(key)
    assert probabilities[2] > initial_probability
    assert probabilities[2] > 0.75
    assert int(np.argmax(probabilities)) == 2


def test_map_revision_advances_after_update_and_prune():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)

    assert voxel_map.revision == 0
    voxel_map.update(point, np.ones(1), logits, timestamp_sec=1.0)
    after_update = voxel_map.revision
    assert after_update == 1

    voxel_map.prune(2.0, stale_ttl_sec=100.0)
    assert voxel_map.revision == after_update + 1


def test_evidence_probabilities_exclude_symmetric_prior():
    voxel_map = VoxelMap(
        voxel_size=0.2,
        K=12,
        evidence_decay=1.0,
        max_frame_evidence=3.0,
    )
    point = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
    observed_probabilities = np.full(12, 0.1 / 11.0, dtype=np.float32)
    observed_probabilities[4] = 0.9
    voxel_map.update(
        point,
        np.asarray([1.0], dtype=np.float32),
        np.log(observed_probabilities)[None, :],
    )

    key = voxel_map.get_voxel_indices(point[0])
    assert voxel_map.get_probabilities(key)[4] < 0.25
    assert voxel_map.get_evidence_probabilities(key)[4] > 0.89


def test_one_dense_frame_has_bounded_evidence():
    dense_map = VoxelMap(K=3, evidence_decay=1.0, max_frame_evidence=1.0)
    sparse_map = VoxelMap(K=3, evidence_decay=1.0, max_frame_evidence=1.0)
    dense_points = np.tile([[0.01, 0.01, 0.01]], (100, 1)).astype(np.float32)
    sparse_points = dense_points[:1]
    logits = np.array([0.0, 4.0, 0.0], dtype=np.float32)

    dense_map.update(dense_points, np.ones(100), logits)
    sparse_map.update(sparse_points, np.ones(1), logits)

    key = dense_map.get_voxel_indices(dense_points[0])
    np.testing.assert_allclose(
        dense_map.voxels[key]['alpha'],
        sparse_map.voxels[key]['alpha'],
        rtol=1e-6,
        atol=1e-6,
    )


def test_uniform_semantics_remain_uncertain_even_with_more_evidence():
    voxel_map = VoxelMap(
        K=4,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
        uncertainty_entropy_weight=0.7,
    )
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    for _ in range(20):
        voxel_map.update(points, np.ones(1), np.zeros((1, 4)))

    key = voxel_map.get_voxel_indices(points[0])
    probabilities = voxel_map.get_probabilities(key)
    _, _, uncertainty = voxel_map.get_uncertainty(key)
    np.testing.assert_allclose(probabilities, np.full(4, 0.25), atol=1e-6)
    assert uncertainty >= 0.7
    assert voxel_map.get_confidence(key) <= 0.3


def test_nonfinite_reliability_is_ignored_without_map_contamination():
    voxel_map = VoxelMap(K=2)
    point = np.asarray([[0.01, 0.01, 0.01]], dtype=np.float32)

    voxel_map.update(
        point,
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([[4.0, 0.0]], dtype=np.float32),
        timestamp_sec=1.0,
    )

    assert voxel_map.voxels == {}


def test_open_vocabulary_features_are_normalized_after_fusion():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)
    feature_a = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    feature_b = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)

    voxel_map.update(points, np.ones(1), logits, feature_a)
    voxel_map.update(points, np.ones(1), logits, feature_b)

    key = voxel_map.get_voxel_indices(points[0])
    feature = voxel_map.voxels[key]['feature_512']
    assert np.isclose(np.linalg.norm(feature), 1.0, atol=1e-6)
    assert feature[0] > 0.0
    assert feature[1] > 0.0


def test_point_colors_are_fused_per_voxel():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)

    voxel_map.update(
        points,
        np.ones(1),
        logits,
        colors=np.array([[0.0, 0.0, 255.0]], dtype=np.float32),
    )
    voxel_map.update(
        points,
        np.ones(1),
        logits,
        colors=np.array([[0.0, 0.0, 200.0]], dtype=np.float32),
    )

    key = voxel_map.get_voxel_indices(points[0])
    color = voxel_map.voxels[key]['color_rgb']
    assert color[2] > 0.85
    assert color[0] < 0.05
    assert color[1] < 0.05


def test_continuous_time_fusion_is_equivalent_across_frame_rates():
    """Equal evidence rates over equal time must yield equal posteriors."""
    common = {
        'K': 3,
        'evidence_decay': 0.9,
        'evidence_decay_reference_sec': 0.1,
        'max_frame_evidence': 1.0,
        'max_total_evidence': 1000.0,
        'max_observation_weight': 1000.0,
    }
    slow_map = VoxelMap(**common)
    fast_map = VoxelMap(**common)
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    reliability = np.ones(1, dtype=np.float32)
    logits = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)
    feature = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
    color = np.array([[20.0, 100.0, 220.0]], dtype=np.float32)

    for timestamp in np.linspace(0.0, 1.0, 11):
        slow_map.update(
            point,
            reliability,
            logits,
            features=feature,
            colors=color,
            timestamp_sec=timestamp,
        )
    for timestamp in np.linspace(0.0, 1.0, 21):
        fast_map.update(
            point,
            reliability,
            logits,
            features=feature,
            colors=color,
            timestamp_sec=timestamp,
        )

    key = slow_map.get_voxel_indices(point[0])
    slow_voxel = slow_map.voxels[key]
    fast_voxel = fast_map.voxels[key]
    np.testing.assert_allclose(
        slow_voxel['alpha'], fast_voxel['alpha'], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        slow_voxel['feature_512'], fast_voxel['feature_512'], atol=1e-6)
    np.testing.assert_allclose(
        slow_voxel['color_rgb'], fast_voxel['color_rgb'], atol=1e-6)
    assert np.isclose(
        slow_voxel['weight_sum'], fast_voxel['weight_sum'], atol=1e-6)
    assert np.isclose(
        slow_voxel['feature_weight_sum'],
        fast_voxel['feature_weight_sum'],
        atol=1e-6,
    )
    assert np.isclose(
        slow_voxel['color_weight_sum'],
        fast_voxel['color_weight_sum'],
        atol=1e-6,
    )


def test_auxiliary_weights_are_bounded_and_keep_adapting():
    """Capped EMA weights must stay finite without freezing old values."""
    voxel_map = VoxelMap(
        K=2,
        evidence_decay=1.0,
        max_observation_weight=2.0,
    )
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    reliability = np.ones(1, dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)
    old_feature = np.array([[1.0, 0.0]], dtype=np.float32)
    new_feature = np.array([[0.0, 1.0]], dtype=np.float32)
    old_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    new_color = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    for _ in range(20):
        voxel_map.update(
            point,
            reliability,
            logits,
            features=old_feature,
            colors=old_color,
        )
    key = voxel_map.get_voxel_indices(point[0])
    voxel = voxel_map.voxels[key]
    assert voxel['weight_sum'] == 2.0
    assert voxel['feature_weight_sum'] == 2.0
    assert voxel['color_weight_sum'] == 2.0

    voxel_map.update(
        point,
        reliability,
        logits,
        features=new_feature,
        colors=new_color,
    )
    voxel = voxel_map.voxels[key]
    assert voxel['weight_sum'] == 2.0
    assert voxel['feature_weight_sum'] == 2.0
    assert voxel['color_weight_sum'] == 2.0
    assert voxel['feature_512'][1] > 0.3
    assert voxel['color_rgb'][2] > 0.3


def test_prune_continuously_decays_without_refreshing_ttl():
    """Periodic aging must reduce evidence but preserve observation age."""
    voxel_map = VoxelMap(
        K=2,
        evidence_decay=0.5,
        evidence_decay_reference_sec=1.0,
        max_observation_weight=10.0,
    )
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[5.0, 0.0]], dtype=np.float32)
    voxel_map.update(
        point,
        np.ones(1),
        logits,
        timestamp_sec=10.0,
    )
    key = voxel_map.get_voxel_indices(point[0])
    initial_alpha = voxel_map.voxels[key]['alpha'].copy()

    first_result = voxel_map.prune(12.0, stale_ttl_sec=5.0)
    voxel = voxel_map.voxels[key]
    expected_evidence = (initial_alpha - 1.0) * 0.25
    np.testing.assert_allclose(
        voxel['alpha'] - 1.0, expected_evidence, atol=1e-6)
    assert first_result['total'] == 0
    assert voxel['last_observed_at_sec'] == 10.0
    assert voxel['last_decay_at_sec'] == 12.0

    second_result = voxel_map.prune(15.1, stale_ttl_sec=5.0)
    assert second_result['stale_ttl'] == 1
    assert key not in voxel_map.voxels


def test_omitted_timestamp_preserves_legacy_per_update_decay():
    """Timestamp-free callers retain one decay quantum per update."""
    voxel_map = VoxelMap(
        K=2,
        evidence_decay=0.5,
        evidence_decay_reference_sec=10.0,
        max_frame_evidence=1.0,
    )
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    reliability = np.ones(1, dtype=np.float32)
    probabilities = np.array([[0.8, 0.2]], dtype=np.float32)
    logits = np.log(probabilities)

    voxel_map.update(point, reliability, logits)
    voxel_map.update(point, reliability, logits)

    key = voxel_map.get_voxel_indices(point[0])
    expected_evidence = 1.5 * probabilities[0]
    np.testing.assert_allclose(
        voxel_map.voxels[key]['alpha'] - 1.0,
        expected_evidence,
        rtol=1e-6,
        atol=1e-6,
    )


def test_voxel_timestamps_are_created_and_refreshed():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)

    voxel_map.update(point, np.ones(1), logits, timestamp_sec=10.0)
    key = voxel_map.get_voxel_indices(point[0])
    assert voxel_map.voxels[key]['created_at_sec'] == 10.0
    assert voxel_map.voxels[key]['last_observed_at_sec'] == 10.0

    voxel_map.update(point, np.ones(1), logits, timestamp_sec=12.5)
    assert voxel_map.voxels[key]['created_at_sec'] == 10.0
    assert voxel_map.voxels[key]['last_observed_at_sec'] == 12.5


def test_dynamic_voxels_use_shorter_ttl_than_static_voxels():
    voxel_map = VoxelMap(voxel_size=1.0, K=3, evidence_decay=0.0)
    points = np.array([[0.1, 0.1, 0.1], [2.1, 0.1, 0.1]], dtype=np.float32)
    logits = np.array([[0.0, 6.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
    voxel_map.update(points, np.ones(2), logits, timestamp_sec=10.0)

    result = voxel_map.prune(
        21.0,
        dynamic_class_ids={1},
        dynamic_ttl_sec=10.0,
        stale_ttl_sec=300.0,
    )

    assert result['dynamic_ttl'] == 1
    assert voxel_map.get_voxel_indices(points[0]) not in voxel_map.voxels
    assert voxel_map.get_voxel_indices(points[1]) in voxel_map.voxels


def test_prune_applies_stale_distance_and_count_limits_deterministically():
    voxel_map = VoxelMap(voxel_size=1.0, K=2, evidence_decay=1.0)
    logits = np.array([[5.0, 0.0]], dtype=np.float32)
    observations = [
        ([0.1, 0.1, 0.1], 1.0),
        ([2.1, 0.1, 0.1], 8.0),
        ([4.1, 0.1, 0.1], 9.0),
        ([20.1, 0.1, 0.1], 9.5),
    ]
    for point, timestamp in observations:
        voxel_map.update(
            np.asarray([point], dtype=np.float32),
            np.ones(1),
            logits,
            timestamp_sec=timestamp,
        )

    result = voxel_map.prune(
        10.0,
        stale_ttl_sec=8.0,
        center=np.zeros(3),
        max_distance_m=10.0,
        max_count=1,
    )

    assert result == {
        'dynamic_ttl': 0,
        'stale_ttl': 1,
        'distance': 1,
        'max_count': 1,
        'total': 3,
        'remaining': 1,
    }
    newest_near_key = voxel_map.get_voxel_indices(observations[2][0])
    assert set(voxel_map.voxels) == {newest_near_key}


def test_ttl_pruning_preserves_legacy_voxel_without_timestamp_metadata():
    voxel_map = VoxelMap(K=2)
    key = (0, 0, 0)
    voxel_map.voxels[key] = {
        'alpha': np.ones(2, dtype=np.float32),
        'pos': np.zeros(3, dtype=np.float32),
    }

    result = voxel_map.prune(100.0, stale_ttl_sec=1.0)

    assert result['total'] == 0
    assert key in voxel_map.voxels
