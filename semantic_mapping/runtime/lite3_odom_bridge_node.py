#!/usr/bin/env python3
"""Repair Lite3 vendor odometry for isolated TF validation."""

import copy
from dataclasses import dataclass

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


@dataclass(frozen=True)
class OdometryRepairConfig:
    """Names used by the isolated Lite3 odometry validation chain."""

    odom_frame_id: str = 'lite3_test_odom'
    base_frame_id: str = 'lite3_test_base_link'
    pose_covariance_diagonal: tuple = (
        0.05, 0.05, 0.10, 0.05, 0.05, 0.10)
    twist_covariance_diagonal: tuple = (
        0.05, 0.05, 0.10, 0.10, 0.10, 0.10)


def repair_lite3_odometry(raw_odom, receipt_time, config):
    """Return corrected odometry and its equivalent isolated transform."""
    fixed = copy.deepcopy(raw_odom)
    fixed.header.stamp = copy.deepcopy(receipt_time)
    fixed.header.frame_id = config.odom_frame_id
    fixed.child_frame_id = config.base_frame_id
    diagonal_indices = (0, 7, 14, 21, 28, 35)
    if not any(fixed.pose.covariance):
        for index, value in zip(
                diagonal_indices, config.pose_covariance_diagonal):
            fixed.pose.covariance[index] = value
    if not any(fixed.twist.covariance):
        for index, value in zip(
                diagonal_indices, config.twist_covariance_diagonal):
            fixed.twist.covariance[index] = value

    transform = TransformStamped()
    transform.header.stamp = copy.deepcopy(receipt_time)
    transform.header.frame_id = config.odom_frame_id
    transform.child_frame_id = config.base_frame_id
    transform.transform.translation.x = fixed.pose.pose.position.x
    transform.transform.translation.y = fixed.pose.pose.position.y
    transform.transform.translation.z = fixed.pose.pose.position.z
    transform.transform.rotation = copy.deepcopy(
        fixed.pose.pose.orientation)
    return fixed, transform


class Lite3OdomBridgeNode(Node):
    """Publish a repaired odometry stream under isolated test names."""

    SOURCE_TOPIC = '/leg_odom2'
    FIXED_TOPIC = '/diagnostics/lite3/leg_odom_fixed'

    def __init__(self):
        """Create the fixed diagnostic odometry and isolated TF outputs."""
        super().__init__('lite3_odom_bridge_node')
        self.config = OdometryRepairConfig()
        self.fixed_odom_publisher = self.create_publisher(
            Odometry, self.FIXED_TOPIC, 10)
        self.test_tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.SOURCE_TOPIC,
            self.odom_cb,
            10,
        )
        self.get_logger().info(
            'Lite3 isolated odometry bridge started: '
            f'{self.SOURCE_TOPIC} -> {self.FIXED_TOPIC}, '
            f'TF={self.config.odom_frame_id} -> '
            f'{self.config.base_frame_id}; no command topics are used')

    def odom_cb(self, raw_odom):
        """Repair one vendor odometry sample at its ROS receipt time."""
        fixed, transform = repair_lite3_odometry(
            raw_odom,
            self.get_clock().now().to_msg(),
            self.config,
        )
        self.fixed_odom_publisher.publish(fixed)
        self.test_tf_broadcaster.sendTransform(transform)


def main(args=None):
    """Run the isolated Lite3 odometry bridge."""
    rclpy.init(args=args)
    node = Lite3OdomBridgeNode()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
