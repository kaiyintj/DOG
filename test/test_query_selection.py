import numpy as np
from collections import deque
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Imu

from semantic_mapping.runtime.ga_bsvm_node import (
    GABsvmNode,
    resolve_semantic_class_indices,
    scale_camera_matrix,
)
from semantic_mapping.runtime.voxel_map import VoxelMap


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


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


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
    node.road_class_idx = 0
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


def make_costmap_node(robot_position):
    node = object.__new__(GABsvmNode)
    node.voxel_map = VoxelMap(
        voxel_size=1.0,
        K=2,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    node.cost_map_size_m = 10.0
    node.cost_map_origin_x = -5.0
    node.cost_map_origin_y = -5.0
    node.costmap_min_confidence = 0.0
    node.costmap_robot_clearance_m = 0.1
    node.costmap_min_height_m = -0.6
    node.costmap_max_height_m = 1.0
    node.semantic_cost_dict = {0: 0, 1: 100}
    node.current_goal_key = None
    node.odom_frame = 'odom'
    node.costmap_tf_warned = False
    node.semantic_cost_pub = CapturingPublisher()
    node.map_pub = None
    node.get_robot_position = lambda: robot_position
    node.get_clock = lambda: type(
        'Clock',
        (),
        {'now': lambda self: type(
            'Now', (), {'to_msg': lambda self: TimeMsg()})()},
    )()
    warnings = []
    node.get_logger = lambda: type(
        'Logger',
        (),
        {
            'info': lambda self, message: None,
            'warn': lambda self, message: warnings.append(message),
        },
    )()
    return node, warnings


def test_camera_matrix_scales_to_runtime_image_resolution():
    camera_matrix = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 620.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    scaled = scale_camera_matrix(camera_matrix, 640, 480, 424, 240)

    np.testing.assert_allclose(
        scaled,
        [
            [397.5, 0.0, 212.0],
            [0.0, 310.0, 120.0],
            [0.0, 0.0, 1.0],
        ],
    )


def make_motion_node(samples=()):
    node = object.__new__(GABsvmNode)
    node.imu_buffer = deque(samples, maxlen=1000)
    node.imu_acceleration_scale = 1.0
    node.imu_window_sec = 0.15
    node.motion_angular_scale = 2.0
    node.motion_accel_scale = 3.0
    node.motion_min_reliability = 0.2
    node.motion_missing_reliability = 0.2
    node.imu_gravity = 9.81
    node.imu_missing_warned = False
    node.imu_match_count = 0
    node.imu_miss_count = 0
    messages = {'info': [], 'warn': []}
    node.get_logger = lambda: type(
        'Logger',
        (),
        {
            'info': lambda self, message: messages['info'].append(message),
            'warn': lambda self, message: messages['warn'].append(message),
        },
    )()
    return node, messages


def test_imu_callback_converts_livox_g_to_si_acceleration():
    node, _ = make_motion_node()
    node.imu_acceleration_scale = 9.80665
    message = Imu()
    message.header.stamp.sec = 10
    message.linear_acceleration.z = 1.0

    node.imu_callback(message)

    stamp_ns, angular_norm, acceleration_norm = node.imu_buffer[-1]
    assert stamp_ns == 10_000_000_000
    assert angular_norm == 0.0
    assert acceleration_norm == 9.80665


def test_missing_imu_motion_reliability_fails_closed():
    node, messages = make_motion_node()
    stamp = TimeMsg(sec=10, nanosec=0)

    reliability, angular_rms, acceleration_deviation = (
        node.compute_motion_reliability(stamp))

    assert reliability == 0.2
    assert angular_rms == 0.0
    assert acceleration_deviation == 0.0
    assert node.imu_miss_count == 1
    assert len(messages['warn']) == 1


def test_unaligned_imu_samples_are_not_treated_as_fully_reliable():
    node, _ = make_motion_node([(1_000_000_000, 0.0, 9.81)])

    reliability, _, _ = node.compute_motion_reliability(
        TimeMsg(sec=10, nanosec=0))

    assert reliability == 0.2
    assert node.imu_match_count == 0


def test_semantic_class_indices_are_resolved_from_reordered_vocab():
    vocab = ['car', 'unknown background', 'road', 'electric bicycle', 'person']

    assert resolve_semantic_class_indices(vocab, ('road',)) == {2}
    assert resolve_semantic_class_indices(
        vocab,
        ('person', 'car', 'electric_bicycle'),
    ) == {0, 3, 4}


def test_semantic_costmap_filters_height_relative_to_robot_base():
    node, _ = make_costmap_node(np.array([0.0, 0.0, 1.0]))
    points = np.array([
        [2.1, 0.1, 0.1],   # obstacle center z=0.5, relative height=-0.5
        [3.1, 0.1, 2.1],   # obstacle center z=2.5, relative height=1.5
    ], dtype=np.float32)
    obstacle_logits = np.array([[0.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    node.voxel_map.update(points, np.ones(2), obstacle_logits)

    assert node.publish_semantic_costmap() is True
    assert len(node.semantic_cost_pub.messages) == 1
    grid = node.semantic_cost_pub.messages[0]
    low_key = node.voxel_map.get_voxel_indices(points[0])
    high_key = node.voxel_map.get_voxel_indices(points[1])
    low_index = (low_key[1] + 5) * 10 + low_key[0] + 5
    high_index = (high_key[1] + 5) * 10 + high_key[0] + 5
    assert grid.data[low_index] == 100
    assert grid.data[high_index] == -1


def test_semantic_costmap_fails_closed_without_robot_tf():
    node, warnings = make_costmap_node(None)
    node.voxel_map.update(
        np.array([[2.1, 0.1, 0.1]], dtype=np.float32),
        np.ones(1),
        np.array([[0.0, 6.0]], dtype=np.float32),
    )

    assert node.publish_semantic_costmap() is False
    assert node.semantic_cost_pub.messages == []
    assert node.costmap_tf_warned is True
    assert warnings


def test_periodic_voxel_prune_uses_ros_time_and_robot_position():
    calls = []

    class FakePrunableMap:
        voxels = {}

        def prune(self, now_sec, **kwargs):
            calls.append((now_sec, kwargs))
            return {
                'dynamic_ttl': 0,
                'stale_ttl': 0,
                'distance': 0,
                'max_count': 0,
                'total': 0,
                'remaining': 0,
            }

    node = object.__new__(GABsvmNode)
    node.voxel_map = FakePrunableMap()
    node.dynamic_class_ids = {3, 4}
    node.dynamic_voxel_ttl_sec = 10.0
    node.voxel_ttl_sec = 300.0
    node.voxel_prune_radius_m = 30.0
    node.voxel_max_count = 250000
    node.current_goal_key = None
    robot_position = np.array([1.0, 2.0, 0.5])
    node.get_robot_position = lambda: robot_position
    node.get_clock = lambda: type(
        'Clock',
        (),
        {'now': lambda self: type(
            'Now', (), {'nanoseconds': 42_500_000_000})()},
    )()
    node.get_logger = lambda: type(
        'Logger', (), {'info': lambda self, message: None})()

    node.prune_voxel_map()

    assert len(calls) == 1
    assert calls[0][0] == 42.5
    assert calls[0][1]['dynamic_class_ids'] == {3, 4}
    np.testing.assert_allclose(calls[0][1]['center'], robot_position)


def test_unverified_projection_calibration_suppresses_navigation_pose():
    node = object.__new__(GABsvmNode)
    node.projection_calibration_verified = False
    node.last_query_text = 'white truck'
    warnings = []
    node.get_logger = lambda: type(
        'Logger', (), {'warn': lambda self, message: warnings.append(message)})()

    published = node._publish_query_goal({}, 5, 'segformer')

    assert published is False
    assert warnings
    assert '标定' in warnings[0]


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
    node.road_class_idx = 0
    node.get_robot_position = lambda: np.array([0.0, 0.0, 0.0])
    node.get_logger = lambda: type(
        'Logger', (), {'warn': lambda self, message: None})()

    _, goal = node.find_approach_goal(
        np.array([3.0, 0.0, 1.0], dtype=np.float32), query_class_idx=3)

    np.testing.assert_allclose(goal, [2.0, 0.0, 0.0], atol=1e-6)


def test_approach_goal_uses_named_road_class_after_vocab_reordering():
    node = make_approach_node(np.array([4.0, 0.0, 0.0]))
    node.road_class_idx = 1
    node.semantic_cost_dict = {0: 100, 1: 0, 2: -1}
    node.voxel_map.add((20, 0, 0), [2.0, 0.0, 0.0], 1)
    node.voxel_map.add((0, 0, 0), [0.0, 0.0, 0.0], 0)

    _, goal = node.find_approach_goal(
        np.array([0.0, 0.0, 0.0]), query_class_idx=0)

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
