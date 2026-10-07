"""Static indoor memory through the real fusion and query boundaries."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.voxel_map import VoxelMap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAIR_POINTS = np.repeat(
    [[2.01, 0.01, 0.41], [2.11, 0.01, 0.41], [2.21, 0.01, 0.41]], 20, axis=0)


def make_indoor_query_node(overrides=None):
    """Use the shipped preset and real ranking, without starting ROS transport."""
    preset = yaml.safe_load(
        (PROJECT_ROOT / 'config/semantic_mapping_sim_indoor.yaml').read_text())
    parameters = {**preset['ga_bsvm_node']['ros__parameters'], **(overrides or {})}
    profile = open_profile(parameters['ontology_profile'])
    node = object.__new__(GABsvmNode)
    for name, value in parameters.items():
        if name.startswith('query_'):
            setattr(node, name, value)
    node.query_class_max_extent_m = profile.query_max_extents
    node.get_robot_position = lambda: np.array([0.0, 0.0, 0.225])
    node.voxel_map = VoxelMap(
        voxel_size=parameters['voxel_size'], K=profile.K, class_colors=profile.colors,
        **{name: parameters[name] for name in (
            'evidence_prior', 'evidence_strength', 'evidence_decay',
            'evidence_decay_reference_sec', 'max_frame_evidence',
            'max_total_evidence', 'source_history_size', 'uncertainty_entropy_weight',
        )},
    )
    return node, parameters, profile


def observe_class(node, profile, class_name, stamp, *, points=CHAIR_POINTS,
                  quality=1.0, source_stamp=None):
    """One new source image projected onto three neighboring surface voxels."""
    logits = np.zeros(profile.K)
    logits[profile.classes.index(class_name)] = 6.0
    node.voxel_map.update(
        points, np.full(len(points), quality), logits, timestamp_sec=stamp,
        source_id=('camera', int(round((stamp if source_stamp is None else source_stamp) * 1e9))))


def prune_indoor(node, parameters, profile, stamp):
    return node.voxel_map.prune(
        stamp, dynamic_class_ids=profile.dynamic_ids,
        dynamic_ttl_sec=parameters['dynamic_voxel_ttl_sec'],
        stale_ttl_sec=parameters['voxel_ttl_sec'], center=node.get_robot_position(),
        max_distance_m=parameters['voxel_prune_radius_m'],
        max_count=parameters['voxel_max_count'])


def select_chair(node, profile):
    return node.select_class_query_target(profile.classes.index('chair'))[0]


@pytest.mark.parametrize('gap', [5.0, 60.0, 600.0])
def test_static_chair_remains_queryable_while_other_surfaces_are_observed(gap):
    node, parameters, profile = make_indoor_query_node()
    observe_class(node, profile, 'chair', 10.0)
    initial = select_chair(node, profile)
    assert initial is not None
    keys = set(node.voxel_map.voxels)
    before = {key: node.voxel_map.voxels[key]['alpha'].copy() for key in keys}

    # Looking elsewhere updates the map, but neither sees nor contradicts the chair.
    observe_class(node, profile, 'wall', 10.0 + gap,
                  points=np.repeat([[4.01, 1.01, 0.41]], 20, axis=0))
    assert prune_indoor(node, parameters, profile, 10.0 + gap)['total'] == 0
    selected = select_chair(node, profile)
    assert selected is not None
    np.testing.assert_allclose(selected['pos'], initial['pos'])
    assert selected['evidence'] == pytest.approx(initial['evidence'])
    for key in keys:
        np.testing.assert_array_equal(node.voxel_map.voxels[key]['alpha'], before[key])
        assert node.voxel_map.voxels[key]['last_observed_at_sec'] == 10.0


def test_disabling_decay_alone_does_not_prevent_static_ttl_forgetting():
    node, parameters, profile = make_indoor_query_node(
        {'evidence_decay': 1.0, 'voxel_ttl_sec': 300.0})
    observe_class(node, profile, 'chair', 10.0)
    assert select_chair(node, profile) is not None
    assert prune_indoor(node, parameters, profile, 311.0)['stale_ttl'] == 3
    assert select_chair(node, profile) is None


@pytest.mark.parametrize('gap', [5.0, 60.0, 600.0])
@pytest.mark.parametrize('quality', [0.2, 1.0])
def test_static_reappearance_adds_only_current_quality_budget(gap, quality):
    node, parameters, profile = make_indoor_query_node()
    observe_class(node, profile, 'chair', 10.0)
    keys = list(node.voxel_map.voxels)
    before = {key: node.voxel_map.voxels[key]['alpha'].copy() for key in keys}
    prune_indoor(node, parameters, profile, 10.0 + gap)
    observe_class(node, profile, 'chair', 10.0 + gap, quality=quality)
    for key in keys:
        voxel = node.voxel_map.voxels[key]
        assert np.sum(voxel['alpha'] - before[key]) == pytest.approx(3 * quality, abs=2e-6)
        assert voxel['weight_sum'] == pytest.approx(3 + 3 * quality)


def test_static_old_source_copies_cannot_refresh_or_increase_evidence():
    node, parameters, profile = make_indoor_query_node()
    observe_class(node, profile, 'chair', 10.0)
    keys = list(node.voxel_map.voxels)
    before = {key: node.voxel_map.voxels[key]['alpha'].copy() for key in keys}
    prune_indoor(node, parameters, profile, 610.0)
    for stamp in (610.0, 611.0, 612.0):
        observe_class(node, profile, 'table', stamp, source_stamp=10.0)
    assert select_chair(node, profile) is not None
    for key in keys:
        np.testing.assert_array_equal(node.voxel_map.voxels[key]['alpha'], before[key])
        assert node.voxel_map.voxels[key]['last_observed_at_sec'] == 10.0


def test_static_saturated_mistake_can_be_corrected_by_new_sources():
    node, parameters, profile = make_indoor_query_node()
    for frame in range(100):
        observe_class(node, profile, 'table', 10.0 + frame * 0.1)
    assert select_chair(node, profile) is None
    prune_indoor(node, parameters, profile, 620.0)
    for frame in range(100):
        observe_class(node, profile, 'chair', 620.0 + frame * 0.1)
    selected = select_chair(node, profile)
    assert selected is not None
    assert selected['class_probability'] > 0.7
    for voxel in node.voxel_map.voxels.values():
        assert np.sum(voxel['alpha']) <= parameters['max_total_evidence'] + 2e-5
