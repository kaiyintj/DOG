import numpy as np
from collections import deque

from semantic_mapping.ga_bsvm_node import GABsvmNode
from semantic_mapping.voxel_map import VoxelMap


class FakeApproachMap:
    voxel_size = 0.1

    def __init__(self, class_count=3):
        self.class_count = class_count
        self.voxels = {}
        self._probabilities = {}
        self._confidences = {}

    def add(self, key, position, class_index, confidence=0.8):
        probabilities = np.zeros(self.class_count, dtype=np.float32)
        probabilities[class_index] = 1.0
        self.voxels[key] = {
            'weight_sum': 3.0,
            'pos': np.asarray(position, dtype=np.float32),
        }
        self._probabilities[key] = probabilities
        self._confidences[key] = confidence

    def get_probabilities(self, key):
        return self._probabilities[key]

    def get_confidence(self, key):
        return self._confidences[key]

    def get_voxel_indices(self, position):
        return tuple(np.floor(np.asarray(position) / self.voxel_size).astype(int))


def make_approach_node(robot_position):
    node = object.__new__(GABsvmNode)
    node.voxel_map = FakeApproachMap()
    node.query_min_weight_sum = 2.0
    node.query_approach_min_distance_m = 1.5
    node.query_approach_radius_m = 2.5
    node.query_approach_distance_m = 2.0
    node.query_clearance_radius_m = 0.35
    node.query_path_clearance_radius_m = 0.35
    node.query_robot_distance_weight = 0.0
    node.query_require_safe_approach = True
    node.query_prefer_robot_side = True
    node.query_require_robot_side = True
    node.query_prefer_direct_path = True
    node.query_same_side_min_cosine = 0.0
    node.query_require_robot_pose_for_approach = True
    node.semantic_cost_dict = {0: 0, 1: 100, 2: -1}
    node.get_robot_position = lambda: robot_position
    node.get_logger = lambda: type(
        'Logger',
        (),
        {
            'info': lambda self, message: None,
            'warn': lambda self, message: None,
        },
    )()
    return node


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


def test_approach_goal_prefers_robot_side_over_far_side():
    node = make_approach_node(np.array([4.0, 0.0, 0.0]))
    node.voxel_map.add((20, 0, 0), [2.0, 0.0, 0.0], 0, confidence=0.1)
    node.voxel_map.add((-20, 0, 0), [-2.0, 0.0, 0.0], 0, confidence=1.0)
    node.voxel_map.add((0, 0, 0), [0.0, 0.0, 0.0], 1)

    _, goal = node.find_approach_goal(
        np.array([0.0, 0.0, 0.0]), query_class_idx=1)

    np.testing.assert_allclose(goal, [2.0, 0.0, 0.0], atol=1e-6)
    assert node._last_approach_was_safe is True


def test_school_parking_regression_avoids_goal_behind_yellow_vehicle():
    node = make_approach_node(np.array([5.03, -2.40, 0.0]))
    object_position = np.array([1.86, -4.29, 0.0])
    node.voxel_map.add(
        (-2, -49, 0), [-0.15, -4.85, 0.0], 0, confidence=1.0)
    node.voxel_map.add(
        (30, -26, 0), [3.0, -2.6, 0.0], 0, confidence=0.5)
    node.voxel_map.add((19, -43, 0), object_position, 1)

    _, goal = node.find_approach_goal(object_position, query_class_idx=1)

    np.testing.assert_allclose(goal, [3.0, -2.6, 0.0], atol=1e-6)


def test_approach_goal_rejects_only_far_side_candidate():
    node = make_approach_node(np.array([4.0, 0.0, 0.0]))
    node.voxel_map.add((-20, 0, 0), [-2.0, 0.0, 0.0], 0)
    node.voxel_map.add((0, 0, 0), [0.0, 0.0, 0.0], 1)

    key, goal = node.find_approach_goal(
        np.array([0.0, 0.0, 0.0]), query_class_idx=1)

    assert key is None
    assert goal is None
    assert node._last_approach_was_safe is False


def test_approach_goal_prefers_clear_direct_route_on_same_side():
    node = make_approach_node(np.array([4.0, 0.0, 0.0]))
    diagonal_y = np.sqrt(4.0 - 1.5 ** 2)
    node.voxel_map.add((20, 0, 0), [2.0, 0.0, 0.0], 0, confidence=1.0)
    node.voxel_map.add(
        (15, 13, 0), [1.5, diagonal_y, 0.0], 0, confidence=0.1)
    node.voxel_map.add((30, 0, 0), [3.0, 0.0, 0.0], 1)
    node.voxel_map.add((0, 0, 0), [0.0, 0.0, 0.0], 1)

    _, goal = node.find_approach_goal(
        np.array([0.0, 0.0, 0.0]), query_class_idx=1)

    np.testing.assert_allclose(goal, [1.5, diagonal_y, 0.0], atol=1e-6)


def test_approach_goal_requires_robot_pose_when_configured():
    node = make_approach_node(None)
    node.voxel_map.add((20, 0, 0), [2.0, 0.0, 0.0], 0)

    key, goal = node.find_approach_goal(
        np.array([0.0, 0.0, 0.0]), query_class_idx=1)

    assert key is None
    assert goal is None


def test_pending_tf_frame_is_fused_after_transform_arrives():
    node = object.__new__(GABsvmNode)
    frame_data = {'points': np.array([[1.0, 2.0, 3.0]])}
    node.pending_tf_frames = deque([{
        'queued_at_ns': 1_000_000_000,
        'target_frame': 'odom',
        'source_frame': 'base_link',
        'stamp': object(),
        'frame_data': frame_data,
    }])
    node.tf_retry_max_age_sec = 2.5
    node.tf_queue_drop_count = 0
    node.consecutive_tf_drops = 1
    node.get_clock = lambda: type(
        'Clock',
        (),
        {'now': lambda self: type('Now', (), {'nanoseconds': 2_000_000_000})()},
    )()
    node.get_logger = lambda: type(
        'Logger', (), {'info': lambda self, message: None})()
    node._transform_semantic_points = (
        lambda points, target, source, stamp: points + 1.0)
    fused = []
    node._fuse_projected_semantic_frame = (
        lambda data, points: fused.append((data, points)))

    node.retry_pending_tf_frames()

    assert len(node.pending_tf_frames) == 0
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0][1], [[2.0, 3.0, 4.0]])
    assert node.consecutive_tf_drops == 0


def test_pending_tf_frame_expires_without_latest_tf_fallback():
    node = object.__new__(GABsvmNode)
    node.pending_tf_frames = deque([{
        'queued_at_ns': 1_000_000_000,
        'target_frame': 'odom',
        'source_frame': 'base_link',
        'stamp': object(),
        'frame_data': {'points': np.zeros((1, 3))},
    }])
    node.tf_retry_max_age_sec = 2.5
    node.tf_queue_drop_count = 0
    node.get_clock = lambda: type(
        'Clock',
        (),
        {'now': lambda self: type('Now', (), {'nanoseconds': 4_000_000_000})()},
    )()
    node.get_logger = lambda: type(
        'Logger', (), {'warn': lambda self, message: None})()
    node._transform_semantic_points = lambda *args: (_ for _ in ()).throw(
        AssertionError('expired frames must not be transformed'))

    node.retry_pending_tf_frames()

    assert len(node.pending_tf_frames) == 0
    assert node.tf_queue_drop_count == 1


def test_pending_tf_retry_does_not_block_newer_transform():
    node = object.__new__(GABsvmNode)
    node.pending_tf_frames = deque([
        {
            'queued_at_ns': 1_000_000_000,
            'target_frame': 'odom',
            'source_frame': 'base_link',
            'stamp': 'missing',
            'frame_data': {'points': np.array([[1.0, 0.0, 0.0]])},
        },
        {
            'queued_at_ns': 1_100_000_000,
            'target_frame': 'odom',
            'source_frame': 'base_link',
            'stamp': 'ready',
            'frame_data': {'points': np.array([[2.0, 0.0, 0.0]])},
        },
    ])
    node.tf_retry_max_age_sec = 2.5
    node.tf_queue_drop_count = 0
    node.consecutive_tf_drops = 2
    node.get_clock = lambda: type(
        'Clock',
        (),
        {'now': lambda self: type('Now', (), {'nanoseconds': 2_000_000_000})()},
    )()
    node.get_logger = lambda: type(
        'Logger', (), {'info': lambda self, message: None})()

    def transform(points, target, source, stamp):
        if stamp == 'missing':
            raise LookupError('historical transform is unavailable')
        return points + 1.0

    node._transform_semantic_points = transform
    fused = []
    node._fuse_projected_semantic_frame = (
        lambda data, points: fused.append(points))

    node.retry_pending_tf_frames()
    node.retry_pending_tf_frames()

    assert len(node.pending_tf_frames) == 1
    np.testing.assert_allclose(fused, [[[3.0, 1.0, 1.0]]])
