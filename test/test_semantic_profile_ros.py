"""Startup contract and capability transport shared by ROS callers."""

import pytest

from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.semantic_profile_ros import load_semantic_contract


def test_startup_contract_derives_profile_and_rejects_outdoor_indoor_mix():
    contract = load_semantic_contract({'ontology_profile': 'indoor7'})

    assert contract.profile.K == 8
    assert contract.profile.classes[0] == 'floor'
    with pytest.raises(ValueError, match='num_classes'):
        load_semantic_contract({
            'ontology_profile': 'indoor7', 'num_classes': 13,
        })
    with pytest.raises(ValueError, match='vocab'):
        load_semantic_contract({
            'ontology_profile': 'indoor7', 'vocab': ['road', 'car'],
        })
    assert load_semantic_contract({}).profile.id == 'outdoor13'


def test_capability_transports_actual_labels_and_rejects_another_profile():
    from semantic_mapping.runtime.semantic_profile_ros import (
        make_semantic_capability,
        read_semantic_capability,
    )

    labels = {0: 'chair', 1: 'wall'}
    frontend = open_profile('indoor7', labels)
    message = make_semantic_capability(frontend, 'test/partial', labels)
    received = read_semantic_capability(message, open_profile('indoor7'))

    assert received.checkpoint_id == 'test/partial'
    assert received.profile.resolve_query('chair').accepted
    assert received.profile.resolve_query('table').status == 'unsupported'
    with pytest.raises(ValueError, match='profile'):
        read_semantic_capability(message, open_profile('outdoor13'))


@pytest.mark.parametrize('class_id', [1, 2, 3, 4, 5, 6, 7])
def test_cost_override_cannot_make_an_indoor_obstacle_or_unknown_free(class_id):
    costs = [0, 100, 100, 100, 100, 100, 100, -1]
    costs[class_id] = 0
    with pytest.raises(ValueError, match='semantic_costs'):
        load_semantic_contract({
            'ontology_profile': 'indoor7', 'semantic_costs': costs,
        })


def test_mapping_node_starts_with_indoor_defaults(monkeypatch, tmp_path):
    import rclpy
    from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode

    monkeypatch.setenv('ROS_DOMAIN_ID', '227')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    rclpy.init(args=[
        '--ros-args', '-p', 'ontology_profile:=indoor7',
        '-p', 'semantic_backend:=segformer',
    ])
    node = None
    try:
        node = GABsvmNode()
        assert node.get_parameter('num_classes').value == 8
        assert node.get_parameter('vocab').value == [
            'floor', 'wall', 'door', 'chair', 'table', 'shelf', 'bed',
            'unknown background',
        ]
        assert node.get_parameter('semantic_costs').value == [
            0, 100, 100, 100, 100, 100, 100, -1,
        ]
        assert len(node.get_parameter('query_class_max_extent_m').value) == 8
        assert node.get_parameter('projection_calibration_verified').value is False
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_active_perception_uses_indoor_class_count(monkeypatch, tmp_path):
    import math
    import rclpy
    from semantic_mapping.runtime.active_perception_node import ActivePerceptionNode

    monkeypatch.setenv('ROS_DOMAIN_ID', '227')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    rclpy.init(args=[
        '--ros-args', '-p', 'ontology_profile:=indoor7',
        '-p', 'cmd_vel_topic:=/profile_test/unused_velocity',
    ])
    node = None
    try:
        node = ActivePerceptionNode()
        assert node.get_parameter('num_classes').value == 8
        assert node.h_max == pytest.approx(math.log(8))
        assert node.get_parameter('cmd_vel_timeout_sec').value == 0.25
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
