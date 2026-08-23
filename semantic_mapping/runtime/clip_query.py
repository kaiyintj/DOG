#!/usr/bin/env python3
"""
Publish one text query for the running semantic frontend to encode.

clip_node 使用与图像特征相同的模型、QuickGELU 变体和提示模板编码文本；
ga_bsvm_node 收到对应特征后在 voxel_map 中搜索目标。

用法:
    ros2 run semantic_mapping clip_query "brown rock"
    ros2 run semantic_mapping clip_query "person"
"""
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ClipQuery(Node):
    """Publish one text query and expose completion to the main loop."""

    def __init__(self, text):
        """Create the one-shot publisher for ``text``."""
        super().__init__('clip_query')
        self.pub = self.create_publisher(String, '/text_query', 10)
        msg = String()
        msg.data = text

        self.timer = self.create_timer(0.5, lambda: self._send_once(msg))
        self.sent = False

    def _send_once(self, msg):
        if self.sent:
            return
        self.pub.publish(msg)
        self.get_logger().info(f'文本查询已发布: "{msg.data}"')
        self.sent = True
        self.timer.cancel()


def main():
    """Publish the command-line text through the canonical query topic."""
    if len(sys.argv) < 2:
        print('用法: ros2 run semantic_mapping clip_query "<text>"')
        sys.exit(1)
    text = ' '.join(sys.argv[1:])
    rclpy.init()
    node = ClipQuery(text)
    try:
        while rclpy.ok() and not node.sent:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
