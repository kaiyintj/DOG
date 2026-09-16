"""Wait for the simulated Go2 to settle before FAST-LIO gravity initialization."""

from collections import deque
import json
import math
import time

from controller_manager_msgs.srv import ListControllers
from livox_ros_driver2.msg import CustomMsg
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def _seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class _SensorGate(Node):
    def __init__(self):
        super().__init__(
            'sim_sensor_gate',
            parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.declare_parameter('startup_timeout_sec', 45.0)
        self.controllers_ready = False
        self.imu_window = deque()
        self.imu_stamp = None
        self.lidar_stamp = None
        self.controller_future = None
        self.controller_client = self.create_client(
            ListControllers, '/controller_manager/list_controllers')
        self.create_timer(
            .2, self._controllers, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_subscription(Imu, '/imu/data', self._imu, qos_profile_sensor_data)
        self.create_subscription(CustomMsg, '/livox/lidar', self._lidar, qos_profile_sensor_data)

    def _controllers(self):
        if self.controller_future is not None:
            if not self.controller_future.done():
                return
            response = self.controller_future.result()
            active = {entry.name for entry in response.controller if entry.state == 'active'}
            self.controllers_ready = {
                'joint_states_controller', 'joint_group_effort_controller',
            }.issubset(active)
            self.controller_future = None
        if self.controller_client.service_is_ready():
            self.controller_future = self.controller_client.call_async(ListControllers.Request())

    def _fresh(self, stamp):
        if stamp is None or stamp <= 0:
            return False
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        return -.05 <= age <= .5

    def _imu(self, message):
        stamp = _seconds(message.header.stamp)
        acceleration = message.linear_acceleration
        angular_velocity = message.angular_velocity
        acc_norm = math.hypot(acceleration.x, acceleration.y, acceleration.z)
        gyro_norm = math.hypot(angular_velocity.x, angular_velocity.y, angular_velocity.z)
        usable = (
            self.controllers_ready and self._fresh(stamp)
            and math.isfinite(acc_norm) and .5 * 9.81 <= acc_norm <= 1.5 * 9.81
            and math.isfinite(gyro_norm) and gyro_norm <= 1.0)
        continuous = self.imu_stamp is None or 0 < stamp - self.imu_stamp <= .1
        if not usable or not continuous:
            self.imu_window.clear()
        if usable:
            self.imu_window.append((
                stamp, (acceleration.x, acceleration.y, acceleration.z), acc_norm, gyro_norm))
            while len(self.imu_window) > 1 and stamp - self.imu_window[1][0] >= 1.0:
                self.imu_window.popleft()
        self.imu_stamp = stamp

    def _lidar(self, message):
        self.lidar_stamp = (
            _seconds(message.header.stamp) if message.point_num > 0 and message.points else None)

    def result(self):
        # Standing Go2 vibration is not quiet at every sample. The one-second
        # window must have a gravity-consistent mean vector and bounded noise.
        # Gross freefall/impact and discontinuities already discard the window.
        stable_since = self.imu_window[0][0] if self.imu_window else None
        statistics = {}
        stable = False
        if (len(self.imu_window) >= 20 and self.imu_stamp - stable_since >= 1.0
                and self._fresh(self.imu_stamp)):
            count = len(self.imu_window)
            mean_acceleration = [
                sum(row[1][axis] for row in self.imu_window) / count for axis in range(3)]
            mean_norm = sum(row[2] for row in self.imu_window) / count
            statistics = {
                'mean_acceleration_norm': math.hypot(*mean_acceleration),
                'acceleration_norm_std': math.sqrt(
                    sum((row[2] - mean_norm) ** 2 for row in self.imu_window) / count),
                'gyro_rms': math.sqrt(sum(row[3] ** 2 for row in self.imu_window) / count),
            }
            stable = (
                abs(statistics['mean_acceleration_norm'] - 9.81) <= .5
                and statistics['acceleration_norm_std'] <= 1.5
                and statistics['gyro_rms'] <= .2)
        if not self.controllers_ready:
            reason = 'controllers_not_active'
        elif not stable:
            reason = 'imu_not_stable'
        elif not self._fresh(self.lidar_stamp):
            reason = 'lidar_not_ready'
        else:
            reason = 'ready'
        return {
            'ready': reason == 'ready', 'reason': reason,
            'stable_since': stable_since, 'imu_stamp': self.imu_stamp,
            'lidar_stamp': self.lidar_stamp, **statistics,
        }


def main(args=None):
    """Exit zero only after active controllers and fresh stationary sensor evidence."""
    rclpy.init(args=args)
    node = _SensorGate()
    try:
        timeout = float(node.get_parameter('startup_timeout_sec').value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('startup_timeout_sec must be finite and positive')
        deadline = time.monotonic() + timeout
        report = node.result()
        while rclpy.ok() and time.monotonic() < deadline and not report['ready']:
            rclpy.spin_once(node, timeout_sec=.05)
            report = node.result()
        print(json.dumps(report), flush=True)
        return 0 if report['ready'] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
