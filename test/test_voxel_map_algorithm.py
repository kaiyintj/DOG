"""Regression tests for time-aware Dirichlet voxel fusion."""

import numpy as np
import pytest

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


def test_continuous_observations_within_reference_period_are_rate_equivalent():
    """Only continuously observed intervals <= reference share evidence rates."""
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


@pytest.mark.parametrize('evidence_cap', [3.0, 1000.0])
@pytest.mark.parametrize('retention', [0.0, 0.5, 0.995, 1.0])
def test_pruning_before_delayed_observation_preserves_fusion(evidence_cap, retention):
    """A delayed frame must give the same evidence at the same final time."""
    maps = [VoxelMap(K=2, evidence_decay=retention,
                     evidence_decay_reference_sec=1.0,
                     max_total_evidence=evidence_cap,
                     max_observation_weight=evidence_cap - 2.0)
            for _ in range(2)]
    point = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    for index, voxel_map in enumerate(maps):
        voxel_map.update(point, np.ones(1), np.array([[4., 0.]]),
                         features=np.array([[1., 0.]]), colors=np.array([[1., 0., 0.]]),
                         timestamp_sec=10.)
        if index:
            voxel_map.prune(12.)
        voxel_map.update(point, np.ones(1), np.array([[0., 4.]]),
                         features=np.array([[0., 1.]]), colors=np.array([[0., 0., 1.]]),
                         timestamp_sec=11.)
        voxel_map.prune(12.)
    first, second = [m.voxels[(0, 0, 0)] for m in maps]
    for field in ('alpha', 'weight_sum', 'feature_weight_sum', 'color_weight_sum'):
        np.testing.assert_allclose(first[field], second[field], atol=2e-6)
    if first['weight_sum'] > 0:
        for field in ('feature_512', 'color_rgb'):
            np.testing.assert_allclose(first[field], second[field], atol=2e-6)
    assert second['last_decay_at_sec'] == 12.
    assert second['last_observed_at_sec'] == 11.


def test_clock_rewind_remains_distinct_from_late_pruned_frame():
    """A genuine observation-clock restart can establish a new time base."""
    voxel_map = VoxelMap(K=2)
    point = np.array([[0.01, 0.01, 0.01]])
    voxel_map.update(point, np.ones(1), np.array([[4., 0.]]), timestamp_sec=10.)
    voxel_map.prune(12.)
    voxel_map.update(point, np.ones(1), np.array([[4., 0.]]), timestamp_sec=1.)
    assert voxel_map.voxels[(0, 0, 0)]['last_decay_at_sec'] == 1.
    assert voxel_map.voxels[(0, 0, 0)]['last_observed_at_sec'] == 1.


@pytest.mark.parametrize('point_count', [20, 200])
def test_saturated_support_preserves_quality_and_auxiliary_budget(point_count):
    points = np.tile([[0.01, 0.01, 0.01]], (point_count, 1))
    maps = []
    for quality in (0.2, 1.0):
        voxel_map = VoxelMap(K=2, evidence_decay=1.0, max_frame_evidence=3.0)
        voxel_map.update(
            points, np.full(point_count, quality), np.array([6.0, 0.0]),
            features=np.array([1.0, 0.0]), colors=np.tile([0.0, 0.0, 1.0], (point_count, 1)),
            timestamp_sec=10.0, source_id=('camera', 10_000_000_000))
        maps.append(voxel_map)
    low, high = [m.voxels[(0, 0, 0)] for m in maps]
    assert np.sum(low['alpha'] - 1.0) < np.sum(high['alpha'] - 1.0)
    assert low['weight_sum'] == pytest.approx(0.6)
    assert high['weight_sum'] == pytest.approx(3.0)
    for voxel in (low, high):
        assert np.sum(voxel['alpha'] - 1.0) == pytest.approx(voxel['weight_sum'])
        assert voxel['feature_weight_sum'] == pytest.approx(voxel['weight_sum'])
        assert voxel['color_weight_sum'] == pytest.approx(voxel['weight_sum'])


def test_near_zero_quality_returns_still_reduce_mean_quality():
    """Discarding weak returns before the mean must not inflate its budget."""
    points = np.tile([[0.01, 0.01, 0.01]], (20, 1))
    evidence = []
    for weak_quality in (0.001, 0.0009, 0.0):
        voxel_map = VoxelMap(K=2, evidence_decay=1.0, max_frame_evidence=3.0)
        reliability = np.full(20, weak_quality)
        reliability[0] = 1.0
        voxel_map.update(points, reliability, [6.0, 0.0])
        evidence.append(voxel_map.voxels[(0, 0, 0)]['weight_sum'])
    assert evidence[0] > evidence[1] > evidence[2]
    assert evidence[2] < 1.0


def test_duplicate_image_cannot_change_existing_voxel_or_refresh_its_age():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    point = np.array([[0.01, 0.01, 0.01]])
    source = ('camera', 10_000_000_000)
    voxel_map.update(point, [1.0], [6.0, 0.0], features=[1.0, 0.0],
                     colors=[[1.0, 0.0, 0.0]], timestamp_sec=10.0, source_id=source)
    before = {field: value.copy() if isinstance(value, np.ndarray) else value
              for field, value in voxel_map.voxels[(0, 0, 0)].items()}
    revision = voxel_map.revision
    for timestamp in (10.1, 10.2, 10.3):
        assert voxel_map.update(point, [1.0], [0.0, 6.0], features=[0.0, 1.0],
                                colors=[[0.0, 0.0, 1.0]], timestamp_sec=timestamp,
                                source_id=source) == (0, 0)
    voxel = voxel_map.voxels[(0, 0, 0)]
    for field in ('alpha', 'feature_512', 'color_rgb', 'weight_sum',
                  'feature_weight_sum', 'color_weight_sum', 'observation_count',
                  'last_observed_at_sec', 'last_decay_at_sec'):
        np.testing.assert_array_equal(voxel[field], before[field])
    assert voxel_map.revision == revision
    voxel_map.prune(11.0, stale_ttl_sec=0.5)
    assert voxel_map.voxels == {}


def test_same_image_can_contribute_to_new_coverage_and_another_camera():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    source = ('camera', 10_000_000_000)
    first = [[0.01, 0.01, 0.01]]
    both = [[0.01, 0.01, 0.01], [0.21, 0.01, 0.01]]
    voxel_map.update(first, [1.0], [6.0, 0.0], timestamp_sec=10.0, source_id=source)
    assert voxel_map.update(both, [1.0, 1.0], [6.0, 0.0], timestamp_sec=10.1,
                            source_id=source) == (1, 0)
    assert voxel_map.voxels[(0, 0, 0)]['observation_count'] == 1
    assert voxel_map.voxels[(2, 0, 0)]['observation_count'] == 1
    assert voxel_map.update(first, [1.0], [6.0, 0.0], timestamp_sec=10.1,
                            source_id=('other_camera', source[1])) == (0, 1)
    assert voxel_map.voxels[(0, 0, 0)]['weight_sum'] == pytest.approx(2.0)
    assert voxel_map.update(first, [1.0], [6.0, 0.0], timestamp_sec=10.2,
                            source_id=source) == (0, 0)


def test_unseen_out_of_order_image_is_admitted_without_clock_rewind():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    point = [[0.01, 0.01, 0.01]]
    voxel_map.update(point, [1.0], [6.0, 0.0], timestamp_sec=10.0,
                     source_id=('camera', 10_060_000_000))
    before = voxel_map.voxels[(0, 0, 0)]['weight_sum']
    assert voxel_map.update(point, [1.0], [0.0, 6.0], timestamp_sec=10.1,
                            source_id=('camera', 10_000_000_000)) == (0, 1)
    voxel = voxel_map.voxels[(0, 0, 0)]
    assert before < voxel['weight_sum'] <= before + 1.0
    assert voxel['last_observed_at_sec'] == 10.1
    assert voxel['last_decay_at_sec'] == 10.1
    assert voxel_map.update(point, [1.0], [0.0, 6.0], timestamp_sec=10.2,
                            source_id=('camera', 10_000_000_000)) == (0, 0)


def test_source_history_is_bounded_and_retired_sources_cannot_reenter():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0, source_history_size=2)
    point = [[0.01, 0.01, 0.01]]
    for stamp in (10, 12, 11, 13):
        voxel_map.update(point, [1.0], [6.0, 0.0], timestamp_sec=20.0 + stamp,
                         source_id=('camera', stamp * 1_000_000_000))
    voxel = voxel_map.voxels[(0, 0, 0)]
    history = voxel['source_history']['camera']
    assert len(history['stamps']) == 2
    assert voxel['observation_count'] == 4
    for stamp in (9, 10, 11, 12, 13):
        assert voxel_map.update(point, [1.0], [0.0, 6.0], timestamp_sec=40.0,
                                source_id=('camera', stamp * 1_000_000_000)) == (0, 0)
    voxel_map.prune(41.0, stale_ttl_sec=1.0)
    assert voxel_map.voxels == {}
    assert voxel_map.update(point, [1.0], [6.0, 0.0], timestamp_sec=42.0,
                            source_id=('camera', 10_000_000_000)) == (1, 0)


@pytest.mark.parametrize('gap', [0.1, 1.0, 10.0, 100.0])
def test_reappearance_adds_only_one_observation_after_aging_history(gap):
    voxel_map = VoxelMap(K=2, max_frame_evidence=3.0)
    points = np.tile([[0.01, 0.01, 0.01]], (20, 1))
    voxel_map.update(points, np.ones(20), [6.0, 0.0], features=[1.0, 0.0],
                     colors=np.tile([1.0, 0.0, 0.0], (20, 1)), timestamp_sec=10.0,
                     source_id=('camera', 10_000_000_000))
    previous = voxel_map.voxels[(0, 0, 0)]['alpha'].copy() - 1.0
    voxel_map.update(points, np.ones(20), [0.0, 6.0], features=[0.0, 1.0],
                     colors=np.tile([0.0, 0.0, 1.0], (20, 1)), timestamp_sec=10.0 + gap,
                     source_id=('camera', int(round((10.0 + gap) * 1e9))))
    voxel = voxel_map.voxels[(0, 0, 0)]
    aged = previous * (0.995 ** (gap / 0.1))
    injected = voxel['alpha'] - 1.0 - aged
    assert np.sum(injected) == pytest.approx(3.0, abs=2e-6)
    assert voxel['weight_sum'] == pytest.approx(np.sum(aged) + 3.0, abs=2e-6)
    assert voxel['feature_weight_sum'] == pytest.approx(voxel['weight_sum'])
    assert voxel['color_weight_sum'] == pytest.approx(voxel['weight_sum'])


@pytest.mark.parametrize('evidence_cap,new_frames', [(20.0, 10), (200.0, 100)])
def test_capped_source_evidence_resists_noise_but_can_correct_the_class(
    evidence_cap, new_frames,
):
    voxel_map = VoxelMap(K=2, max_frame_evidence=3.0, max_total_evidence=evidence_cap)
    points = np.tile([[0.01, 0.01, 0.01]], (20, 1))
    for frame in range(100):
        voxel_map.update(points, np.ones(20), [6.0, 0.0], timestamp_sec=10.0 + frame * 0.1,
                         source_id=('camera', 10_000_000_000 + frame * 100_000_000))
    # One low-quality disagreement and copies of it cannot flip a stable voxel.
    noisy_source = ('camera', 20_000_000_000)
    for index in range(10):
        voxel_map.update(points, np.full(20, 0.2), [0.0, 6.0], timestamp_sec=20.0 + index * 0.001,
                         source_id=noisy_source)
    assert np.argmax(voxel_map.get_probabilities((0, 0, 0))) == 0
    for frame in range(1, new_frames + 1):
        voxel_map.update(points, np.ones(20), [0.0, 6.0], timestamp_sec=20.0 + frame * 0.1,
                         source_id=('camera', 20_000_000_000 + frame * 100_000_000))
    assert np.argmax(voxel_map.get_probabilities((0, 0, 0))) == 1
    assert voxel_map.get_evidence_probabilities((0, 0, 0))[1] > 0.7


def test_pruning_with_lagging_clock_does_not_decay_history_twice():
    maps = [VoxelMap(K=2, evidence_decay=.5) for _ in range(2)]
    for voxel_map in maps:
        for stamp in (10.0, 10.08):
            voxel_map.update([[.01, .01, .01]], [1.0], [6.0, 0.0],
                             features=[1.0, 0.0], colors=[[1.0, 0.0, 0.0]],
                             timestamp_sec=stamp, source_id=('camera', round(stamp * 1e9)))
    maps[1].prune(10.04)
    assert maps[1].voxels[(0, 0, 0)]['last_decay_at_sec'] == 10.08
    for voxel_map in maps:
        voxel_map.update([[.01, .01, .01]], [1.0], [0.0, 6.0],
                         features=[0.0, 1.0], colors=[[0.0, 0.0, 1.0]],
                         timestamp_sec=10.16, source_id=('camera', 10_160_000_000))
    for field in ('alpha', 'weight_sum', 'feature_weight_sum', 'color_weight_sum',
                  'feature_512', 'color_rgb'):
        np.testing.assert_allclose(maps[0].voxels[(0, 0, 0)][field],
                                   maps[1].voxels[(0, 0, 0)][field])


def test_confirmed_clock_restart_reuses_source_stamps_without_reusing_old_clocks():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    point = [[0.01, 0.01, 0.01]]
    voxel_map.update(point, [1.0], [6.0, 0.0], timestamp_sec=1.0,
                     source_id=('camera', 1_000_000_000))
    voxel_map.update(point, [1.0], [6.0, 0.0], timestamp_sec=10.0,
                     source_id=('camera', 10_000_000_000))
    old_evidence = voxel_map.voxels[(0, 0, 0)]['alpha'].copy()
    voxel_map.reset_clock(1.0)
    np.testing.assert_array_equal(voxel_map.voxels[(0, 0, 0)]['alpha'], old_evidence)
    assert voxel_map.update(point, [1.0], [0.0, 6.0], timestamp_sec=1.0,
                            source_id=('camera', 1_000_000_000)) == (0, 1)
    assert voxel_map.voxels[(0, 0, 0)]['weight_sum'] == pytest.approx(3.0)
    assert voxel_map.prune(1.1, stale_ttl_sec=0.5)['total'] == 0


@pytest.mark.parametrize('retention', [0.0, 0.5])
def test_delayed_source_after_reset_ages_to_pruned_horizon_and_counts_only_contribution(retention):
    voxel_map = VoxelMap(K=2, evidence_decay=retention, evidence_decay_reference_sec=1.0)
    point = [[0.01, 0.01, 0.01]]
    voxel_map.update(point, [1.0], [6.0, 0.0], features=[1.0, 0.0],
                     colors=[[1.0, 0.0, 0.0]], timestamp_sec=10.0,
                     source_id=('camera', 10_000_000_000))
    previous = voxel_map.voxels[(0, 0, 0)]['alpha'].copy() - 1.0
    voxel_map.reset_clock(1.1)
    voxel_map.prune(1.2)

    counts = voxel_map.update(point, [1.0], [0.0, 6.0], features=[0.0, 1.0],
                              colors=[[0.0, 0.0, 1.0]], timestamp_sec=1.0,
                              source_id=('camera', 1_000_000_000))

    voxel = voxel_map.voxels[(0, 0, 0)]
    assert counts == (0, int(retention > 0.0))
    assert voxel['last_decay_at_sec'] == 1.2
    expected = previous * retention**0.1 + retention**0.2 * VoxelMap.softmax([0.0, 6.0])
    np.testing.assert_allclose(voxel['alpha'] - 1.0, expected, atol=2e-6)
    assert voxel['observation_count'] == 1 + int(retention > 0.0)
    for field in ('weight_sum', 'feature_weight_sum', 'color_weight_sum'):
        assert voxel[field] == pytest.approx(retention**0.1 + retention**0.2)
