from types import SimpleNamespace

import pytest
from std_msgs.msg import String

from semantic_mapping.runtime.clip_query import SemanticQuery


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class CapturingTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class SilentLogger:
    def info(self, unused_message):
        pass


def test_query_waits_for_ga_subscription_before_one_shot_publish():
    node = object.__new__(SemanticQuery)
    node.pub = CapturingPublisher()
    node.timer = CapturingTimer()
    node.sent = False
    node.wait_reported = False
    node.get_logger = lambda: SilentLogger()
    ready = []
    node._receivers_ready = lambda: bool(ready)
    message = String(data='chair')

    node._send_once(message)

    assert node.pub.messages == []
    assert not node.sent
    assert not node.timer.cancelled

    ready.append(True)
    node._send_once(message)
    node._send_once(message)

    assert node.pub.messages == [message]
    assert node.sent
    assert node.timer.cancelled


@pytest.mark.parametrize('backend,receivers,expected', [
    ('segformer', {'/text_query': ['rosbag2_recorder']}, False),
    ('clip', {'/text_query': ['clip_node']}, False),
    ('clip', {'/text_query': ['ga_bsvm_node']}, False),
    ('clip', {'/text_query': ['ga_bsvm_node', 'clip_node']}, False),
    ('clip', {'/text_query': ['ga_bsvm_node', 'clip_node'],
              '/query_feature': ['ga_bsvm_node']}, True),
    ('segformer', {'/text_query': ['ga_bsvm_node']}, True),
])
def test_query_requires_actual_backend_receivers(backend, receivers, expected):
    node = object.__new__(SemanticQuery)
    node.backend = backend
    node.backend_future = None
    node.get_subscriptions_info_by_topic = lambda topic: [
        SimpleNamespace(node_name=name, node_namespace='/')
        for name in receivers.get(topic, [])]
    assert node._receivers_ready() is expected


def test_query_reads_backend_from_mapper_before_sending():
    node = object.__new__(SemanticQuery)
    node.backend = None
    node.backend_future = None
    node.get_subscriptions_info_by_topic = lambda topic: [
        SimpleNamespace(node_name='ga_bsvm_node', node_namespace='/')]
    requests = []
    future = SimpleNamespace(
        done=lambda: True,
        result=lambda: SimpleNamespace(values=[SimpleNamespace(string_value='segformer')]))
    node.backend_client = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda request: requests.append(request) or future)
    assert not node._receivers_ready()
    assert requests[0].names == ['semantic_backend']
    assert node._receivers_ready()


def test_query_cli_waits_for_clip_and_delivers_to_both_receivers(monkeypatch, tmp_path):
    """Exercise parameter discovery and one-shot delivery through real local DDS."""
    import subprocess
    import sys
    import time

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    monkeypatch.setenv('ROS_DOMAIN_ID', '226')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    rclpy.init(args=[])
    executor = SingleThreadedExecutor()
    mapper = Node('ga_bsvm_node')
    mapper.declare_parameter('semantic_backend', 'clip')
    mapped, encoded = [], []
    mapper.create_subscription(String, '/text_query', lambda msg: mapped.append(msg.data), 10)
    mapper.create_subscription(Float32MultiArray, '/query_feature', lambda msg: None, 10)
    executor.add_node(mapper)
    process = subprocess.Popen(
        [sys.executable, '-m', 'semantic_mapping.runtime.clip_query', 'chair'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    frontend = None
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.05)
        assert process.poll() is None
        assert mapped == []
        frontend = Node('clip_node')
        frontend.create_subscription(
            String, '/text_query', lambda msg: encoded.append(msg.data), 10)
        executor.add_node(frontend)
        deadline = time.monotonic() + 10.
        while time.monotonic() < deadline and not (mapped and encoded):
            executor.spin_once(timeout_sec=.05)
        output, _ = process.communicate(timeout=5.)
        assert process.returncode == 0, output
        assert mapped == encoded == ['chair']
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5.)
        executor.shutdown()
        if frontend is not None:
            frontend.destroy_node()
        mapper.destroy_node()
        rclpy.shutdown()
