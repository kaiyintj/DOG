#!/usr/bin/env python3
"""Publish one query to the selected semantic backend and exit."""
import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import String


class SemanticQuery(Node):
    """Wait for GA-BSVM, publish one text query, and expose completion."""

    def __init__(self, text):
        """Create the one-shot publisher for ``text``."""
        super().__init__('semantic_query')
        self.pub = self.create_publisher(String, '/text_query', 10)
        self.backend_client = self.create_client(
            GetParameters, '/ga_bsvm_node/get_parameters')
        self.backend_future = None
        self.backend = None
        msg = String()
        msg.data = text

        self.timer = self.create_timer(0.1, lambda: self._send_once(msg))
        self.sent = False
        self.wait_reported = False

    def _send_once(self, msg):
        if self.sent:
            return
        if not self._receivers_ready():
            if not self.wait_reported:
                self.get_logger().info('等待 GA-BSVM 与所选查询后端接入...')
                self.wait_reported = True
            return
        self.pub.publish(msg)
        self.get_logger().info(f'文本查询已发布: "{msg.data}"')
        self.sent = True
        self.timer.cancel()

    def _has_receiver(self, topic, name):
        return any(
            info.node_name == name and info.node_namespace == '/'
            for info in self.get_subscriptions_info_by_topic(topic))

    def _receivers_ready(self):
        """Require the mapper and its backend, not merely a topic subscriber."""
        if not self._has_receiver('/text_query', 'ga_bsvm_node'):
            self.backend = None
            self.backend_future = None
            return False
        if self.backend is None:
            if self.backend_future is None:
                if self.backend_client.service_is_ready():
                    request = GetParameters.Request(names=['semantic_backend'])
                    self.backend_future = self.backend_client.call_async(request)
                return False
            if not self.backend_future.done():
                return False
            try:
                self.backend = self.backend_future.result().values[0].string_value
            except (IndexError, AttributeError, RuntimeError):
                self.backend = None
            self.backend_future = None
        if self.backend == 'segformer':
            return True
        if self.backend == 'clip':
            return (
                self._has_receiver('/text_query', 'clip_node')
                and self._has_receiver('/query_feature', 'ga_bsvm_node'))
        self.backend = None
        return False


# Keep imports of the old name working while ``clip_query`` remains a CLI alias.
ClipQuery = SemanticQuery


def main():
    """Publish the command-line text through the canonical query topic."""
    arguments = remove_ros_args(args=sys.argv)
    if len(arguments) < 2:
        print('用法: ros2 run semantic_mapping semantic_query "<text>"')
        sys.exit(1)
    text = ' '.join(arguments[1:])
    rclpy.init()
    node = SemanticQuery(text)
    try:
        while rclpy.ok() and not node.sent:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
