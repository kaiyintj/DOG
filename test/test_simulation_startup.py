"""Exercise the startup command through ROS sensor and controller interfaces."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from livox_ros_driver2.msg import CustomMsg, CustomPoint
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


@pytest.fixture
def simulated_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv('ROS_DOMAIN_ID', '229')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    context = Context()
    rclpy.init(context=context)
    node = Node('startup_input_fixture', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    clock_pub = node.create_publisher(Clock, '/clock', 10)
    imu_pub = node.create_publisher(Imu, '/imu/data', qos_profile_sensor_data)
    lidar_pub = node.create_publisher(CustomMsg, '/livox/lidar', qos_profile_sensor_data)
    state = {'active': True, 'stamp': 1.0, 'lidar_enabled': True}

    def controllers(_request, response):
        response.controller = [
            ControllerState(name=name, state='active' if state['active'] else 'inactive')
            for name in ('joint_states_controller', 'joint_group_effort_controller')]
        return response

    node.create_service(ListControllers, '/controller_manager/list_controllers', controllers)

    def publish(acceleration=.8, gyro=.01, dt=.02):
        state['stamp'] += dt
        sec, ns = divmod(round(state['stamp'] * 1e9), 1_000_000_000)
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = sec, ns
        clock_pub.publish(clock)
        imu = Imu()
        imu.header.stamp = clock.clock
        imu.header.frame_id = 'imu_link'
        imu.linear_acceleration.z = acceleration
        imu.angular_velocity.z = gyro
        imu_pub.publish(imu)
        lidar = CustomMsg()
        lidar.header.stamp = clock.clock
        lidar.header.frame_id = 'mid360_link'
        lidar.points = [CustomPoint(x=1.0)]
        lidar.point_num = 1
        if state['lidar_enabled']:
            lidar_pub.publish(lidar)
        executor.spin_once(timeout_sec=.01)

    try:
        yield state, publish, imu_pub
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def _start_gate(timeout=5.0):
    return subprocess.Popen([
        sys.executable, '-m', 'semantic_mapping.runtime.simulation_startup',
        '--ros-args', '-p', f'startup_timeout_sec:={timeout}',
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())


def test_startup_waits_through_freefall_then_accepts_stationary_sensors(simulated_inputs):
    state, publish, imu_pub = simulated_inputs
    process = _start_gate()
    try:
        deadline = time.monotonic() + 3.0
        while imu_pub.get_subscription_count() == 0 and process.poll() is None:
            assert time.monotonic() < deadline, 'startup command did not subscribe to IMU'
            publish(.8)
        for _ in range(70):
            publish(.8)  # Recorded startup transient: too small to initialize gravity.
        assert process.poll() is None, process.communicate()[0]
        transient_end = state['stamp']
        deadline = time.monotonic() + 3.0
        while process.poll() is None and time.monotonic() < deadline:
            publish(9.81)
        stdout, _ = process.communicate(timeout=1)
        assert process.returncode == 0, stdout
        report = json.loads(stdout.splitlines()[-1])
        assert report['ready'] is True
        assert report['stable_since'] > transient_end
        assert report['imu_stamp'] - report['stable_since'] >= 1.0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


def test_startup_accepts_recorded_standing_vibration(simulated_inputs):
    _, publish, _ = simulated_inputs
    trace_path = Path(__file__).parent / 'fixtures/go2_standing_imu_norms.json'
    trace = json.loads(trace_path.read_text())
    process = _start_gate(timeout=3.0)
    try:
        deadline = time.monotonic() + 5.0
        while process.poll() is None and time.monotonic() < deadline:
            for _, acceleration, gyro in trace['samples']:
                publish(acceleration, gyro, dt=.01)
                if process.poll() is not None:
                    break
        stdout, _ = process.communicate(timeout=1)
        assert process.returncode == 0, stdout
        report = json.loads(stdout.splitlines()[-1])
        assert report['ready'] is True
        assert report['imu_stamp'] - report['stable_since'] >= 1.0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


@pytest.mark.parametrize('changes, accelerations, gyro, reason', [
    ({'active': False}, [9.81], .01, 'controllers_not_active'),
    ({}, [.8], .01, 'imu_not_stable'),
    ({'lidar_enabled': False}, [9.81], .01, 'lidar_not_ready'),
    ({}, [8.0], .01, 'imu_not_stable'),
    ({}, [6.0, 13.62], .01, 'imu_not_stable'),
    ({}, [9.81], .3, 'imu_not_stable'),
])
def test_startup_times_out_without_required_readiness(
    simulated_inputs, changes, accelerations, gyro, reason,
):
    state, publish, _ = simulated_inputs
    state.update(changes)
    process = _start_gate(timeout=1.5)
    try:
        deadline = time.monotonic() + 4.0
        index = 0
        while process.poll() is None and time.monotonic() < deadline:
            publish(accelerations[index % len(accelerations)], gyro)
            index += 1
        stdout, _ = process.communicate(timeout=1)
        assert process.returncode == 1, stdout
        report = json.loads(stdout.splitlines()[-1])
        assert report['ready'] is False
        assert report['reason'] == reason
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
