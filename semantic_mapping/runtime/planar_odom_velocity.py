"""
Derive body-frame planar velocity from stamped localization poses.

Used by Gazebo navigation when localization publishes pose-only odometry.
No TF is published and no simulator ground truth enters the controller.
"""
import copy
import math
from collections import deque


def yaw(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


class PlanarVelocityEstimator:
    def __init__(self, window=0.2):
        self.window = window
        self.history = deque()

    def update(self, message):
        stamp = message.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        frame = (message.header.frame_id, message.child_frame_id)
        if self.history:
            last_t, last_frame, _ = self.history[-1]
            if t <= last_t or t - last_t > 1.0 or frame != last_frame:
                self.history.clear()
        self.history.append((t, frame, copy.deepcopy(message.pose.pose)))
        while len(self.history) > 2 and t - self.history[1][0] >= self.window:
            self.history.popleft()
        first_t, _, first = self.history[0]
        dt = t - first_t
        if dt < 0.05:
            return None
        pose = message.pose.pose
        heading = yaw(pose.orientation)
        vx = (pose.position.x - first.position.x) / dt
        vy = (pose.position.y - first.position.y) / dt
        delta = heading - yaw(first.orientation)
        fixed = copy.deepcopy(message)
        fixed.twist.twist.linear.x = math.cos(heading)*vx + math.sin(heading)*vy
        fixed.twist.twist.linear.y = -math.sin(heading)*vx + math.cos(heading)*vy
        fixed.twist.twist.angular.z = math.atan2(math.sin(delta), math.cos(delta)) / dt
        return fixed


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry

    rclpy.init(args=args)
    node = Node('planar_odom_velocity')
    node.declare_parameter('input_topic', '/Odometry')
    node.declare_parameter('output_topic', '/navigation/odometry')
    estimator = PlanarVelocityEstimator()
    publisher = node.create_publisher(Odometry, node.get_parameter('output_topic').value, 10)

    def receive(message):
        fixed = estimator.update(message)
        if fixed is not None:
            publisher.publish(fixed)

    node.create_subscription(Odometry, node.get_parameter('input_topic').value,
                             receive, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
