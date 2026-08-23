import pytest

from semantic_mapping.runtime import active_perception_node


class FakeNode:
    def __init__(self):
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self):
        self.nodes = []
        self.shutdown_calls = 0

    def add_node(self, node):
        self.nodes.append(node)

    def spin(self):
        raise KeyboardInterrupt

    def shutdown(self):
        self.shutdown_calls += 1


def test_active_main_does_not_repeat_shutdown_after_sigint(monkeypatch):
    node = FakeNode()
    executor = FakeExecutor()
    monkeypatch.setattr(active_perception_node.rclpy, 'init', lambda: None)
    monkeypatch.setattr(
        active_perception_node, 'ActivePerceptionNode', lambda: node)
    monkeypatch.setattr(
        active_perception_node,
        'MultiThreadedExecutor',
        lambda num_threads: executor,
    )
    monkeypatch.setattr(active_perception_node.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(
        active_perception_node.rclpy,
        'shutdown',
        lambda: pytest.fail('shutdown must not repeat after SIGINT'),
    )

    active_perception_node.main()

    assert executor.nodes == [node]
    assert executor.shutdown_calls == 1
    assert node.destroyed
