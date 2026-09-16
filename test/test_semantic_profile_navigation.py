"""Observe profile admission at the navigation node's text message seam."""

import numpy as np
import pytest
from std_msgs.msg import String

from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.semantic_profile_ros import (
    SemanticCapability,
    make_semantic_capability,
)
from semantic_mapping.runtime.voxel_map import VoxelMap


class QueryOutputs:
    """Capture ROS output messages and refusal logs without a ROS graph."""

    def __init__(self):
        self.messages = []
        self.warnings = []

    def publish(self, message):
        self.messages.append(message)

    def info(self, message):
        pass

    def warn(self, message, **kwargs):
        self.warnings.append(message)

    def error(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize('labels,query,reason', [
    ({0: 'chair'}, 'table', 'unsupported'),
    ({0: 'floor'}, 'floor', 'nonqueryable'),
    ({0: 'chair'}, 'spaceship', 'unresolved'),
    (None, 'table', 'not_ready'),
])
def test_text_query_refusal_stops_retry_and_emits_no_target_or_goal(
    labels, query, reason,
):
    node = object.__new__(GABsvmNode)
    node.profile = open_profile('indoor7')
    node.semantic_capability = (
        None if labels is None else SemanticCapability(
            'test/checkpoint', open_profile('indoor7', labels))
    )
    node.vocab = list(node.profile.classes)
    node.num_classes = node.profile.K
    node.semantic_backend = 'segformer'
    node.voxel_map = VoxelMap(K=8, class_colors=node.profile.colors)
    node.query_retry_pending = True
    node.last_query_feature = np.ones(2)
    outputs = QueryOutputs()
    node.query_target_pub = outputs
    node.goal_pub = outputs
    node.get_logger = lambda: outputs

    node.text_query_cb(String(data=query))

    assert not node.query_retry_pending
    assert node.last_query_feature is None
    assert outputs.messages == []
    assert any(reason in warning for warning in outputs.warnings)


@pytest.mark.parametrize('candidate_class', range(8))
def test_indoor_approach_requires_floor_role_even_with_low_cost(candidate_class):
    node = object.__new__(GABsvmNode)
    profile = open_profile('indoor7')
    node.traversable_class_ids = profile.traversable_ids
    node.voxel_map = VoxelMap(K=8, class_colors=profile.colors)
    candidate = np.array([1.05, 0.05, 0.05], dtype=np.float32)
    logits = np.full(8, -10.0)
    logits[candidate_class] = 10.0
    for _ in range(30):
        node.voxel_map.update([candidate], [1.0], logits)
    node.semantic_cost_dict = dict(enumerate(profile.semantic_costs))
    node.semantic_cost_dict[candidate_class] = 0
    node.query_min_weight_sum = 2.0
    node.query_approach_min_confidence = 0.15
    node.query_approach_min_distance_m = 0.6
    node.query_approach_radius_m = 1.5
    node.query_approach_distance_m = 1.0
    node.query_require_safe_approach = True
    node.query_require_robot_side = True
    node.query_require_robot_pose_for_approach = True
    node.get_robot_position = lambda: np.array([3.05, 0.05, 0.05])
    node.get_logger = lambda: QueryOutputs()

    _, goal = node.find_approach_goal(
        np.array([0.05, 0.05, 0.05]), query_class_idx=3)

    if candidate_class == 0:
        np.testing.assert_allclose(goal, candidate, atol=1e-6)
    else:
        assert goal is None


def test_capability_messages_allow_supported_query_and_reject_live_model_change():
    from collections import deque

    node = object.__new__(GABsvmNode)
    node.profile = open_profile('indoor7')
    node.semantic_capability = None
    node._accepted_semantic_capability = None
    node.semantic_backend = 'segformer'
    node.voxel_map = VoxelMap(K=8, class_colors=node.profile.colors)
    node.pending_tf_frames = deque()
    outputs = QueryOutputs()
    node.get_logger = lambda: outputs
    node.query_target_pub = outputs
    node.goal_pub = outputs
    node.vocab = list(node.profile.classes)

    node.semantic_capability_callback(make_semantic_capability(
        node.profile, 'test/chair', {0: 'chair'}))
    node.text_query_cb(String(data='chair'))
    assert node.query_retry_pending  # Supported, but no map evidence yet.
    assert outputs.messages == []

    node.pending_tf_frames.append('waiting observation')
    node.semantic_capability_callback(make_semantic_capability(
        node.profile, 'test/changed', {0: 'table'}))
    node.text_query_cb(String(data='table'))
    assert not node.query_retry_pending
    assert not node.pending_tf_frames
    assert outputs.messages == []
    assert any('restart' in warning for warning in outputs.warnings)


@pytest.mark.parametrize('full_posterior', [True, False])
def test_segformer_frame_waits_for_capability_before_projection(full_posterior):
    from cv_bridge import CvBridge
    from sensor_msgs.msg import PointCloud2
    from semantic_mapping.runtime.semantic_posterior import posterior_array_to_image

    node = object.__new__(GABsvmNode)
    node.semantic_capability = None
    node.frame_count = 0
    node.frame_stride = 1
    node.num_classes = 8
    node.bridge = CvBridge()
    node.require_camera_info = True
    node.camera_info_received = False
    node.camera_info_wait_warned = False
    node.camera_info_topic = '/test/camera_info'
    outputs = QueryOutputs()
    node.get_logger = lambda: outputs
    source = node.bridge.cv2_to_imgmsg(
        np.zeros((1, 1, 3), dtype=np.uint8), encoding='rgb8')
    source.header.stamp.sec = 1
    source.header.frame_id = 'camera'
    if full_posterior:
        posterior = posterior_array_to_image(
            np.array([[[1., 0., 0., 0., 0., 0., 0., 0.]]]), source.header)
        node.segformer_posterior_sync_callback(PointCloud2(), posterior, source)
    else:
        mask = node.bridge.cv2_to_imgmsg(np.zeros((1, 1), dtype=np.uint8))
        confidence = node.bridge.cv2_to_imgmsg(np.ones((1, 1), dtype=np.float32))
        node.segformer_sync_callback(PointCloud2(), mask, confidence, source)

    assert any('capability' in warning for warning in outputs.warnings)
    assert not node.camera_info_wait_warned
