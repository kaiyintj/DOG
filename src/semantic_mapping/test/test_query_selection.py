import numpy as np

from semantic_mapping.ga_bsvm_node import GABsvmNode
from semantic_mapping.voxel_map import VoxelMap


def make_candidate(index, x, y, score, evidence=2.0):
    return {
        'key': (index, 0, 0),
        'pos': np.array([x, y, 0.5], dtype=np.float32),
        'similarity': score,
        'class_probability': score,
        'score': score,
        'evidence': evidence,
    }


def test_person_query_rejects_wall_sized_cluster():
    node = object.__new__(GABsvmNode)
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.25
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0, 100.0, 100.0, 1.5, 5.0, 100.0]
    node.get_robot_position = lambda: None

    person = [
        make_candidate(0, 0.0, 0.0, 0.70),
        make_candidate(1, 0.1, 0.0, 0.72),
        make_candidate(2, 0.2, 0.0, 0.71),
    ]
    wall = [
        make_candidate(10 + index, 2.0 + 0.2 * index, 0.0, 0.80)
        for index in range(12)
    ]

    selected = node.select_query_cluster(person + wall, query_class_idx=3)

    assert selected is not None
    assert selected['voxel_count'] == 3
    assert selected['pos'][0] < 1.0


def test_simulation_fallback_keeps_standoff_distance():
    node = object.__new__(GABsvmNode)
    node.voxel_map = VoxelMap(K=6)
    node.query_min_weight_sum = 2.0
    node.query_approach_min_distance_m = 0.6
    node.query_approach_radius_m = 1.5
    node.query_approach_distance_m = 1.0
    node.query_clearance_radius_m = 0.35
    node.query_require_safe_approach = False
    node.semantic_cost_dict = {0: 0, 1: 100, 2: 100, 3: 100, 4: 100, 5: -1}
    node.get_robot_position = lambda: np.array([0.0, 0.0, 0.0])
    node.get_logger = lambda: type(
        'Logger', (), {'warn': lambda self, message: None})()

    _, goal = node.find_approach_goal(
        np.array([3.0, 0.0, 1.0], dtype=np.float32), query_class_idx=3)

    np.testing.assert_allclose(goal, [2.0, 0.0, 0.0], atol=1e-6)
