"""Opt-in, offline checkpoint/ROS transport integration; never starts motion."""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.skipif(
    not os.environ.get('SEMANTIC_PROFILE_TEST_MODEL'),
    reason='Set SEMANTIC_PROFILE_TEST_MODEL to a local ADE20K snapshot',
)
def test_actual_indoor_checkpoint_publishes_eight_channels_and_latched_capability(
    monkeypatch, tmp_path,
):
    import rclpy
    import torch
    from cv_bridge import CvBridge
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
    from semantic_mapping.runtime.segformer_node import SegformerNode
    from semantic_mapping.runtime.semantic_posterior import posterior_image_to_array
    from semantic_mapping.runtime.semantic_profile import open_profile
    from semantic_mapping.runtime.semantic_profile_ros import (
        read_semantic_capability,
        semantic_capability_qos,
    )

    monkeypatch.setenv('ROS_DOMAIN_ID', '227')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    thread_count = torch.get_num_threads()
    torch.set_num_threads(2)
    rclpy.init(args=[
        '--ros-args', '-p', 'ontology_profile:=indoor7',
        '-p', 'model_id:=' + os.environ['SEMANTIC_PROFILE_TEST_MODEL'],
        '-p', 'device:=cpu', '-p', 'semantic_backend:=segformer',
        '-p', 'image_topic:=/profile_test/rgb',
        '-p', 'image_is_compressed:=false',
        '-p', 'inference_interval_sec:=0.0',
        '-p', 'debug_log_raw_predictions:=false',
    ])
    nodes = []
    executor = SingleThreadedExecutor()

    def wait_for(predicate, seconds=15.0):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert predicate(), 'Timed out waiting for local ROS profile transport'

    try:
        frontend = SegformerNode()
        nodes.append(frontend)
        interpolation_calls = []
        interpolate = frontend.functional.interpolate

        def tracked_interpolate(*args, **kwargs):
            interpolation_calls.append(kwargs.get('size'))
            return interpolate(*args, **kwargs)

        frontend.functional = SimpleNamespace(interpolate=tracked_interpolate)
        # Both receivers start after the frontend's one-time capability publish.
        mapper = GABsvmNode()
        nodes.append(mapper)
        probe = Node('profile_test_probe', use_global_arguments=False)
        nodes.append(probe)
        capabilities, posteriors, sources = [], [], []
        probe.create_subscription(
            String, '/segformer/semantic_capability', capabilities.append,
            semantic_capability_qos())
        probe.create_subscription(
            Image, '/segformer/project_posterior', posteriors.append,
            qos_profile_sensor_data)
        probe.create_subscription(
            Image, '/segformer/source_image', sources.append,
            qos_profile_sensor_data)
        image_pub = probe.create_publisher(
            Image, '/profile_test/rgb', qos_profile_sensor_data)
        for node in nodes:
            executor.add_node(node)
        wait_for(lambda: capabilities and mapper.semantic_capability is not None)
        capability = read_semantic_capability(capabilities[0], open_profile('indoor7'))
        assert capability.profile.supported_ids == frozenset(range(7))
        assert capability.profile.resolve_query('书架').accepted
        wait_for(lambda: image_pub.get_subscription_count() >= 1)
        image = CvBridge().cv2_to_imgmsg(
            np.full((64, 96, 3), 127, dtype=np.uint8), encoding='rgb8')
        image.header.stamp = probe.get_clock().now().to_msg()
        image.header.frame_id = 'profile_test_camera'
        image_pub.publish(image)
        wait_for(lambda: posteriors and sources, seconds=30.0)
        posterior = posterior_image_to_array(posteriors[0], expected_classes=8)
        assert posteriors[0].encoding == '16FC8'
        assert posterior.shape[-1] == 8
        assert np.isfinite(posterior).all()
        np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, atol=1e-3)
        assert posteriors[0].header == sources[0].header == image.header
        assert interpolation_calls == []
        # A late hard-mask consumer must enable compatible outputs on the next frame.
        masks, confidences = [], []
        probe.create_subscription(Image, frontend.class_mask_topic, masks.append,
                                  qos_profile_sensor_data)
        probe.create_subscription(Image, frontend.confidence_topic, confidences.append,
                                  qos_profile_sensor_data)
        wait_for(lambda: frontend.class_mask_pub.get_subscription_count() > 0
                 and frontend.confidence_pub.get_subscription_count() > 0)
        image.header.stamp = probe.get_clock().now().to_msg()
        image_pub.publish(image)
        wait_for(lambda: masks and confidences and len(posteriors) >= 2, seconds=30.)
        assert interpolation_calls == [(64, 96)]
        assert masks[0].header == confidences[0].header == image.header
        assert masks[0].encoding == 'mono8'
        assert confidences[0].encoding == '32FC1'
        np.testing.assert_allclose(
            posterior, posterior_image_to_array(posteriors[-1], expected_classes=8), atol=1e-3)
        assert mapper.get_parameter('projection_calibration_verified').value is False
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()
        torch.set_num_threads(thread_count)
