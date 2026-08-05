#!/usr/bin/env python3
"""Forward semantic PoseStamped goals to the Nav2 NavigateToPose action."""

import copy
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class NavGoalBridgeNode(Node):
    """Make semantic navigation independent of an RViz navigation panel."""

    def __init__(self):
        super().__init__('nav_goal_bridge_node')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('navigate_to_pose_action', '/navigate_to_pose')
        self.declare_parameter('retry_period_sec', 0.5)

        self.goal_pose_topic = str(
            self.get_parameter('goal_pose_topic').value)
        self.navigate_to_pose_action = str(
            self.get_parameter('navigate_to_pose_action').value)
        retry_period_sec = max(
            0.1, float(self.get_parameter('retry_period_sec').value))

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_to_pose_action,
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            self.goal_pose_topic,
            self.goal_pose_cb,
            10,
        )
        self.retry_timer = self.create_timer(
            retry_period_sec, self.try_dispatch_pending_goal)

        self.pending_goal = None
        self.send_in_progress = False
        self.goal_sequence = 0
        self.latest_dispatched_sequence = 0
        self.last_server_warning_ns = None

        self.get_logger().info(
            f'Nav2 目标桥接已启动: {self.goal_pose_topic} -> '
            f'{self.navigate_to_pose_action}')

    @staticmethod
    def _is_valid_goal(msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        values = (
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        quaternion_norm = math.sqrt(
            orientation.x ** 2
            + orientation.y ** 2
            + orientation.z ** 2
            + orientation.w ** 2
        )
        return (
            bool(msg.header.frame_id)
            and all(math.isfinite(value) for value in values)
            and quaternion_norm > 1e-6
        )

    def goal_pose_cb(self, msg):
        if not self._is_valid_goal(msg):
            self.get_logger().error(
                '拒绝无效 /goal_pose：frame_id、坐标或四元数不合法。')
            return

        self.goal_sequence += 1
        self.pending_goal = (self.goal_sequence, copy.deepcopy(msg))
        self.get_logger().info(
            f'收到语义导航目标 #{self.goal_sequence}: '
            f'frame={msg.header.frame_id}, '
            f'x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}')
        self.try_dispatch_pending_goal()

    def _warn_server_unavailable(self):
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_server_warning_ns is None
            or now_ns - self.last_server_warning_ns >= 5_000_000_000
        ):
            self.get_logger().warn(
                f'Nav2 action {self.navigate_to_pose_action} 尚不可用；'
                '目标已缓存。请启动 nav_with_remap.launch.py。')
            self.last_server_warning_ns = now_ns

    def try_dispatch_pending_goal(self):
        if self.pending_goal is None or self.send_in_progress:
            return
        if not self.action_client.server_is_ready():
            self._warn_server_unavailable()
            return

        sequence, pose = self.pending_goal
        self.pending_goal = None
        self.send_in_progress = True
        goal = NavigateToPose.Goal()
        goal.pose = pose
        response_future = self.action_client.send_goal_async(goal)
        response_future.add_done_callback(
            lambda future, sequence=sequence: self.goal_response_cb(
                future, sequence))

    def goal_response_cb(self, future, sequence):
        self.send_in_progress = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'发送 Nav2 目标 #{sequence} 失败: {exc}')
            return

        if not goal_handle.accepted:
            self.get_logger().error(f'Nav2 拒绝目标 #{sequence}。')
            self.try_dispatch_pending_goal()
            return

        self.latest_dispatched_sequence = max(
            self.latest_dispatched_sequence, sequence)
        self.get_logger().info(f'Nav2 已接受目标 #{sequence}。')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, sequence=sequence: self.goal_result_cb(
                result, sequence))
        self.try_dispatch_pending_goal()

    def goal_result_cb(self, future, sequence):
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().error(
                f'读取 Nav2 目标 #{sequence} 结果失败: {exc}')
            return

        status_names = {
            GoalStatus.STATUS_SUCCEEDED: '成功',
            GoalStatus.STATUS_CANCELED: '取消',
            GoalStatus.STATUS_ABORTED: '失败',
        }
        status_name = status_names.get(status, f'状态码 {status}')
        log = (
            self.get_logger().info
            if status == GoalStatus.STATUS_SUCCEEDED
            else self.get_logger().warn
        )
        log(f'Nav2 目标 #{sequence} 结束: {status_name}。')


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
