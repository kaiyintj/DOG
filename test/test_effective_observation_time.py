"""Observation age is separate from decay anchoring and total concentration."""

import numpy as np
import pytest

from semantic_mapping.runtime.voxel_map import VoxelMap


def observed_map(**parameters):
    mapping = VoxelMap(K=2, **parameters)
    mapping.update([[.01, .01, .01]], [1.0], [6, 0], timestamp_sec=10,
                   source_id=('camera', 10_000_000_000))
    return mapping


def test_rejected_and_zero_weight_sources_do_not_refresh_effective_time():
    mapping = observed_map(evidence_decay=1)
    for stamp, quality, source in [(11, 1, 10_000_000_000), (12, 0, 12_000_000_000)]:
        mapping.update([[.01, .01, .01]], [quality], [6, 0], timestamp_sec=stamp,
                       source_id=('camera', source))
        assert mapping.voxels[(0, 0, 0)]['last_effective_observed_at_sec'] == 10
        assert mapping.last_effective_updates == {}


def test_zero_after_delay_aging_does_not_refresh_effective_time():
    mapping = observed_map(evidence_decay=0)
    mapping.reset_clock(1.1)
    mapping.prune(1.2)
    mapping.update([[.01, .01, .01]], [1], [0, 6], timestamp_sec=1,
                   source_id=('camera', 1_000_000_000))
    assert mapping.voxels[(0, 0, 0)]['last_effective_observed_at_sec'] is None
    assert mapping.last_effective_updates == {}


def test_clock_reanchoring_and_pruning_are_not_effective_observations():
    mapping = observed_map(evidence_decay=1)
    mapping.prune(100)
    assert mapping.voxels[(0, 0, 0)]['last_effective_observed_at_sec'] == 10
    mapping.reset_clock(1)
    assert mapping.voxels[(0, 0, 0)]['last_effective_observed_at_sec'] is None
    mapping.update([[.01, .01, .01]], [1], [6, 0], timestamp_sec=1.1,
                   source_id=('camera', 1_100_000_000))
    assert mapping.voxels[(0, 0, 0)]['last_effective_observed_at_sec'] == 1.1


def test_saturated_new_source_is_effective_without_net_concentration_growth():
    mapping = VoxelMap(K=2, evidence_decay=1, max_total_evidence=5)
    points = np.repeat([[.01, .01, .01]], 20, axis=0)
    for stamp in (10, 11):
        mapping.update(points, np.ones(20), [6, 0], timestamp_sec=stamp,
                       source_id=('camera', stamp * 1_000_000_000))
    before = mapping.voxels[(0, 0, 0)]['alpha'].copy()
    mapping.update(points, np.ones(20), [6, 0], timestamp_sec=12,
                   source_id=('camera', 12_000_000_000))
    voxel = mapping.voxels[(0, 0, 0)]
    np.testing.assert_allclose(voxel['alpha'], before)
    assert voxel['last_effective_observed_at_sec'] == 12
    assert mapping.last_effective_updates[(0, 0, 0)][0] == pytest.approx(3)
