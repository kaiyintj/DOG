import math
import threading
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Twist
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from semantic_mapping.runtime.active_perception_node import ActivePerceptionNode
from semantic_mapping.runtime.active_perception_node import scale_velocity_command
from semantic_mapping.runtime.active_perception_node import aggregate_uncertainty
from semantic_mapping.runtime.active_perception_node import command_watchdog_expired
from semantic_mapping.runtime.active_perception_node import is_valid_frame_id
from semantic_mapping.runtime.active_perception_node import is_sample_fresh
from semantic_mapping.runtime.active_perception_node import make_imu_sensor_qos
from semantic_mapping.runtime.active_perception_node import rate_limit_scale
from semantic_mapping.runtime.active_perception_node import resolve_h_max
from semantic_mapping.runtime.active_perception_node import sample_age_seconds
from semantic_mapping.runtime.active_perception_node import stale_required_inputs
from semantic_mapping.runtime.active_perception_node import stamp_to_seconds
from semantic_mapping.runtime.active_perception_node import transform_xyz_points


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def make_stale_control_harness(steady_times=(100.0, 100.0)):
    times = iter(steady_times)
    publishers = {
        'cmd_pub': RecordingPublisher(),
        'entropy_pub': RecordingPublisher(),
        'speed_scale_pub': RecordingPublisher(),
        'mode_pub': RecordingPublisher(),
    }
    harness = SimpleNamespace(
        _steady_now_sec=lambda: next(times),
        _ros_now_sec=lambda: 10.0,
        _warn_throttled=lambda *args, **kwargs: None,
        command_lock=threading.Lock(),
        state_lock=threading.RLock(),
        command_sequence=0,
        last_cmd_receipt_steady_sec=None,
        watchdog_stop_published=False,
        path_points=None,
        cloud_tree=None,
        cloud_uncertainties=None,
        path_stamp_sec=None,
        cloud_stamp_sec=None,
        imu_stamp_sec=None,
        plan_max_age_sec=2.0,
        entropy_max_age_sec=2.5,
        imu_max_age_sec=0.5,
        freshness_future_tolerance_sec=0.1,
        require_fresh_imu=True,
        h_max=math.log(13),
        min_speed_ratio=0.3,
        unknown_speed_ratio=0.65,
        filtered_uncertainty=0.0,
        current_speed_scale=1.0,
        last_scale_update_sec=9.9,
        uncertainty_ema_gain=0.2,
        uncertainty_percentile=90.0,
        uncertainty_percentile_weight=0.35,
        lookahead_dist=2.0,
        search_radius=0.5,
        max_scale_rate_per_sec=1.6,
        linear_deadband=0.01,
        angular_deadband=0.01,
        preserve_command_curvature=True,
        scale_angular_velocity=True,
        angular_min_speed_ratio=0.3,
        cmd_vel_timeout_sec=0.25,
        **publishers,
    )
    return harness, publishers


def test_uncertainty_scaling_preserves_command_curvature():
    linear_x, linear_y, angular_z = scale_velocity_command(
        0.2,
        0.0,
        0.4,
        0.3,
        linear_deadband=0.01,
        angular_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert linear_x == pytest.approx(0.06)
    assert linear_y == 0.0
    assert angular_z == pytest.approx(0.12)
    assert linear_x / angular_z == pytest.approx(0.5)


def test_deadband_is_applied_before_scaling():
    linear_x, _, _ = scale_velocity_command(
        0.02,
        0.0,
        0.0,
        0.3,
        linear_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert linear_x == pytest.approx(0.006)


def test_pure_rotation_remains_pure_rotation():
    linear_x, linear_y, angular_z = scale_velocity_command(
        0.0,
        0.0,
        0.4,
        0.3,
        linear_deadband=0.01,
        angular_deadband=0.01,
        preserve_command_curvature=True,
    )

    assert math.hypot(linear_x, linear_y) == 0.0
    assert angular_z == pytest.approx(0.12)


def test_h_max_is_derived_from_class_count_unless_explicit():
    assert resolve_h_max(0.0, 13) == pytest.approx(math.log(13))
    assert resolve_h_max(-1.0, 13) == pytest.approx(math.log(13))
    assert resolve_h_max(2.0, 13) == pytest.approx(2.0)


def test_automatic_h_max_rejects_invalid_class_count():
    with pytest.raises(ValueError):
        resolve_h_max(0.0, 1)


def test_frame_validation_fails_closed_for_invalid_tf_names():
    assert is_valid_frame_id('odom')
    assert is_valid_frame_id('robot/odom')
    assert not is_valid_frame_id('')
    assert not is_valid_frame_id('/odom')
    assert not is_valid_frame_id('odom frame')


def test_point_transform_applies_rotation_and_translation():
    half_sqrt_two = math.sqrt(0.5)
    transformed = transform_xyz_points(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [10.0, 20.0, 1.0],
        [0.0, 0.0, half_sqrt_two, half_sqrt_two],
    )

    assert transformed.reshape(-1).tolist() == pytest.approx([
        10.0, 21.0, 1.0,
        8.0, 20.0, 1.0,
    ])


def test_point_transform_rejects_invalid_quaternion():
    with pytest.raises(ValueError):
        transform_xyz_points([[1.0, 0.0, 0.0]], [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0])


def test_uncertainty_aggregation_includes_upper_tail():
    values = [0.0, 0.0, 0.0, 10.0]
    assert aggregate_uncertainty(
        values, percentile=100.0, percentile_weight=0.0) == pytest.approx(2.5)
    assert aggregate_uncertainty(
        values, percentile=100.0, percentile_weight=0.5) == pytest.approx(6.25)
    assert aggregate_uncertainty(
        values, percentile=100.0, percentile_weight=1.0) == pytest.approx(10.0)


def test_scale_rate_limit_depends_on_elapsed_time():
    slow_update, _ = rate_limit_scale(0.5, 1.0, 1.1, 1.0, 0.2)
    long_update, _ = rate_limit_scale(0.5, 1.0, 2.0, 1.0, 0.2)

    assert slow_update == pytest.approx(0.52)
    assert long_update == pytest.approx(0.7)


def test_scale_rate_limit_handles_first_sample_and_clock_reset():
    first_scale, first_stamp = rate_limit_scale(
        None, 0.7, 10.0, None, 0.2)
    reset_scale, reset_stamp = rate_limit_scale(
        first_scale, 1.0, 5.0, first_stamp, 0.2)

    assert first_scale == pytest.approx(0.7)
    assert first_stamp == pytest.approx(10.0)
    assert reset_scale == pytest.approx(0.7)
    assert reset_stamp == pytest.approx(5.0)


def test_ros_stamp_conversion_rejects_absent_or_invalid_values():
    assert stamp_to_seconds(SimpleNamespace(sec=12, nanosec=250_000_000)) \
        == pytest.approx(12.25)
    assert stamp_to_seconds(SimpleNamespace(sec=0, nanosec=0)) is None
    assert stamp_to_seconds(SimpleNamespace(sec=-1, nanosec=0)) is None
    assert stamp_to_seconds(
        SimpleNamespace(sec=1, nanosec=1_000_000_000)) is None


def test_sample_freshness_is_bounded_and_fails_closed_on_clock_reset():
    assert is_sample_fresh(10.0, 9.5, 1.0)
    assert is_sample_fresh(10.0, 10.05, 1.0, future_tolerance_sec=0.1)
    assert not is_sample_fresh(10.0, 8.9, 1.0)
    assert not is_sample_fresh(10.0, 10.2, 1.0, future_tolerance_sec=0.1)
    assert not is_sample_fresh(10.0, None, 1.0)
    assert sample_age_seconds(10.0, None) == math.inf


def test_required_input_gate_names_each_stale_source():
    assert stale_required_inputs(
        now_sec=10.0,
        path_stamp_sec=9.9,
        cloud_stamp_sec=7.0,
        imu_stamp_sec=None,
        path_max_age_sec=2.0,
        cloud_max_age_sec=2.5,
        imu_max_age_sec=0.5,
    ) == ('entropy_cloud', 'imu')

    assert stale_required_inputs(
        now_sec=10.0,
        path_stamp_sec=9.9,
        cloud_stamp_sec=9.9,
        imu_stamp_sec=None,
        path_max_age_sec=2.0,
        cloud_max_age_sec=2.5,
        imu_max_age_sec=0.5,
        require_imu=False,
    ) == ()


def test_imu_profile_matches_sensor_data_best_effort_qos():
    qos = make_imu_sensor_qos()

    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE
    assert qos.depth >= 1


def test_stale_inputs_immediately_force_minimum_speed():
    harness, publishers = make_stale_control_harness()
    command = Twist()
    command.linear.x = 1.0
    command.angular.z = 0.5

    ActivePerceptionNode.cmd_vel_cb(harness, command)

    assert len(publishers['cmd_pub'].messages) == 1
    output = publishers['cmd_pub'].messages[0]
    assert output.linear.x == pytest.approx(0.3)
    assert output.angular.z == pytest.approx(0.15)
    assert harness.filtered_uncertainty == pytest.approx(harness.h_max)
    assert harness.current_speed_scale == pytest.approx(0.3)
    assert publishers['speed_scale_pub'].messages[-1].data \
        == pytest.approx(0.3)
    assert publishers['mode_pub'].messages[-1].data.startswith(
        'STALE[path,entropy_cloud,imu]')


def test_callback_drops_command_that_expires_during_processing():
    harness, publishers = make_stale_control_harness(
        steady_times=(100.0, 100.3))
    command = Twist()
    command.linear.x = 1.0

    ActivePerceptionNode.cmd_vel_cb(harness, command)

    assert publishers['cmd_pub'].messages == []


def test_steady_clock_watchdog_stops_once_on_timeout():
    assert not command_watchdog_expired(10.24, 10.0, 0.25)
    assert command_watchdog_expired(10.25, 10.0, 0.25)
    assert command_watchdog_expired(9.0, 10.0, 0.25)
    assert command_watchdog_expired(10.0, None, 0.25)

    harness, publishers = make_stale_control_harness(
        steady_times=(100.3, 100.4))
    harness.last_cmd_receipt_steady_sec = 100.0
    ActivePerceptionNode.command_watchdog_cb(harness)
    ActivePerceptionNode.command_watchdog_cb(harness)

    assert len(publishers['cmd_pub'].messages) == 1
    stop = publishers['cmd_pub'].messages[0]
    assert stop.linear.x == 0.0
    assert stop.angular.z == 0.0
    assert publishers['speed_scale_pub'].messages[-1].data == 0.0
    assert publishers['mode_pub'].messages[-1].data \
        == 'STOPPED (cmd_vel timeout)'
