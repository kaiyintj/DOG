"""Synthetic ROS adapter for runner tests, never a Gazebo/model acceptance result."""

import json
import os
from pathlib import Path

from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TransformStamped, Twist
from lifecycle_msgs.srv import GetState
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.semantic_profile_ros import (
    make_semantic_capability, semantic_capability_qos,
)


def main():
    """Supply real ROS transport with synthetic static inputs and a refusal reply."""
    rclpy.init()
    node = rclpy.create_node('active_perception_node')
    mapper = rclpy.create_node('ga_bsvm_node')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(mapper)
    streams = {
        '/clock': Clock, '/imu/data': Imu, '/livox/lidar': CustomMsg,
        '/d435i/image_raw': Image, '/cloud_registered': PointCloud2,
        '/semantic_cloud': PointCloud2, '/semantic_cost_map': OccupancyGrid,
        '/segformer/project_posterior': Image, '/odom/ground_truth': Odometry,
        '/Odometry': Odometry, '/tf': TFMessage, '/cmd_vel_champ': Twist,
    }
    publishers = {topic: node.create_publisher(kind, topic, qos_profile_sensor_data)
                  for topic, kind in streams.items()}
    capability = node.create_publisher(
        String, '/segformer/semantic_capability', semantic_capability_qos())
    checkpoint = os.environ['BENCHMARK_TEST_CHECKPOINT']
    labels = json.loads((Path(checkpoint) / 'config.json').read_text())['id2label']
    capability.publish(make_semantic_capability(open_profile('indoor7'), checkpoint, labels))

    def controllers(_request, response):
        response.controller = [ControllerState(name=name, state='active') for name in (
            'joint_states_controller', 'joint_group_effort_controller')]
        return response

    def active(_request, response):
        response.current_state.id = 3
        response.current_state.label = 'active'
        return response

    node.create_service(ListControllers, '/controller_manager/list_controllers', controllers)
    for name in ('controller_server', 'planner_server', 'bt_navigator',
                 'behavior_server', 'velocity_smoother'):
        node.create_service(GetState, f'/{name}/get_state', active)
    ActionServer(node, NavigateToPose, '/navigate_to_pose', lambda _goal: NavigateToPose.Result())
    mapper.create_subscription(
        String, '/text_query', lambda msg: mapper.get_logger().warning(
            f'SegFormer query rejected (nonqueryable): "{msg.data}"'), 10)
    timestamp = 1.0
    try:
        while rclpy.ok():
            timestamp += .01
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = divmod(round(timestamp * 1e9), 10**9)
            publishers['/clock'].publish(clock)
            for topic, kind in streams.items():
                if kind in (Clock, Twist):
                    continue
                message = kind()
                if hasattr(message, 'header'):
                    message.header.stamp = clock.clock
                    message.header.frame_id = 'odom'
                if kind is Imu:
                    message.linear_acceleration.z = 9.81
                elif kind is CustomMsg:
                    message.points = [CustomPoint(x=1.0)]
                    message.point_num = 1
                elif kind is PointCloud2:
                    message.width = message.height = 1
                    message.point_step = message.row_step = 12
                    message.data = bytes(12)
                elif kind is Image:
                    message.width = message.height = 2
                    if 'posterior' in topic:
                        posterior = np.zeros((2, 2, 8), dtype=np.float16)
                        posterior[..., 7] = 1.0
                        message.encoding = '16FC8'
                        message.step = 32
                        message.data = posterior.tobytes()
                    else:
                        message.encoding = 'rgb8'
                        message.step = 6
                        message.data = bytes(12)
                elif kind is Odometry:
                    message.child_frame_id = 'base_link'
                    message.pose.pose.orientation.w = 1.0
                    if topic.endswith('ground_truth'):
                        message.header.frame_id = 'world'
                elif kind is OccupancyGrid:
                    message.info.width = message.info.height = 1
                    message.info.resolution = .1
                    message.data = [0]
                elif kind is TFMessage:
                    transform = TransformStamped()
                    transform.header.stamp = clock.clock
                    transform.header.frame_id = 'odom'
                    transform.child_frame_id = 'base_link'
                    transform.transform.rotation.w = 1.0
                    message.transforms = [transform]
                publishers[topic].publish(message)
            executor.spin_once(timeout_sec=.01)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        mapper.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
