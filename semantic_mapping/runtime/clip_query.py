#!/usr/bin/env python3
"""Publish one query to the selected semantic backend and exit."""
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SemanticQuery(Node):
    """Wait for GA-BSVM, publish one text query, and expose completion."""

    def __init__(self, text):
        """Create the one-shot publisher for ``text``."""
        super().__init__('semantic_query')
        self.pub = self.create_publisher(String, '/text_query', 10)
        msg = String()
        msg.data = text

        self.timer = self.create_timer(0.1, lambda: self._send_once(msg))
        self.sent = False
        self.wait_reported = False

    def _send_once(self, msg):
        if self.sent:
            return
        if self.pub.get_subscription_count() == 0:
            if not self.wait_reported:
                self.get_logger().info('等待 GA-BSVM 接收文本查询...')
                self.wait_reported = True
            return
        self.pub.publish(msg)
        self.get_logger().info(f'文本查询已发布: "{msg.data}"')
        self.sent = True
        self.timer.cancel()


# Keep imports of the old name working while ``clip_query`` remains a CLI alias.
ClipQuery = SemanticQuery


def main():
    """Publish the command-line text through the canonical query topic."""
    if len(sys.argv) < 2:
        print('用法: ros2 run semantic_mapping semantic_query "<text>"')
        sys.exit(1)
    text = ' '.join(sys.argv[1:])
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
