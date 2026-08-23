import pytest
from rclpy._rclpy_pybind11 import RCLError

from semantic_mapping.runtime import ga_bsvm_node


class FakeNode:
    def __init__(self):
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


def _raise_rcl_error(unused_node):
    raise RCLError('the given context is not valid')


def test_ga_main_accepts_rcl_error_after_context_shutdown(monkeypatch):
    node = FakeNode()
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'init', lambda: None)
    monkeypatch.setattr(ga_bsvm_node, 'GABsvmNode', lambda: node)
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'spin', _raise_rcl_error)
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(
        ga_bsvm_node.rclpy,
        'shutdown',
        lambda: pytest.fail('shutdown must not repeat after context shutdown'),
    )

    ga_bsvm_node.main()

    assert node.destroyed


def test_ga_main_preserves_rcl_error_while_context_is_valid(monkeypatch):
    node = FakeNode()
    shutdown_calls = []
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'init', lambda: None)
    monkeypatch.setattr(ga_bsvm_node, 'GABsvmNode', lambda: node)
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'spin', _raise_rcl_error)
    monkeypatch.setattr(ga_bsvm_node.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        ga_bsvm_node.rclpy,
        'shutdown',
        lambda: shutdown_calls.append(True),
    )

    with pytest.raises(RCLError, match='context is not valid'):
        ga_bsvm_node.main()

    assert node.destroyed
    assert shutdown_calls == [True]
