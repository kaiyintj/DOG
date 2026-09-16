"""Tests for the controlled-motion sensor capture validator."""

import importlib.util
import sqlite3
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'verify_lite3_controlled_motion_capture.py'
)
SPEC = importlib.util.spec_from_file_location(
    'verify_lite3_controlled_motion_capture', SCRIPT_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _timestamps(start_s, duration_s, rate_hz):
    count = int(duration_s * rate_hz) + 1
    return [
        int((start_s + index / rate_hz) * 1e9)
        for index in range(count)
    ]


def _write_bag(bag_directory, topics):
    bag_directory.mkdir(parents=True)
    database_path = bag_directory / '{}_0.db3'.format(
        bag_directory.name)
    with sqlite3.connect(database_path) as connection:
        connection.executescript("""
            CREATE TABLE topics(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL,
                offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages(
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
        """)
        message_id = 1
        for topic_id, (name, message_type, timestamps) in enumerate(
                topics, start=1):
            connection.execute(
                'INSERT INTO topics VALUES (?, ?, ?, ?, ?)',
                (topic_id, name, message_type, 'cdr', ''),
            )
            rows = []
            for timestamp in timestamps:
                rows.append((message_id, topic_id, timestamp, b'data'))
                message_id += 1
            connection.executemany(
                'INSERT INTO messages VALUES (?, ?, ?, ?)', rows)


def _write_valid_capture(root, command_count=0, marker_count=16):
    _write_bag(root / 'imu', [
        (
            '/timefix/imu',
            'sensor_msgs/msg/Imu',
            _timestamps(1000.0, 70.0, 200.0),
        ),
    ])
    _write_bag(root / 'lidar', [
        (
            '/timefix/lidar',
            'livox_ros_driver2/msg/CustomMsg',
            _timestamps(1000.1, 70.0, 10.0),
        ),
        (
            '/tf_static',
            'tf2_msgs/msg/TFMessage',
            [int(1000.1e9)],
        ),
    ])
    _write_bag(root / 'camera', [
        (
            '/camera/color/image_raw',
            'sensor_msgs/msg/Image',
            _timestamps(1000.2, 70.0, 15.0),
        ),
        (
            '/camera/color/camera_info',
            'sensor_msgs/msg/CameraInfo',
            _timestamps(1000.2, 70.0, 15.0),
        ),
    ])
    state_stream = _timestamps(1000.15, 70.0, 200.0)
    markers = [
        int((1001.0 + index * 4.0) * 1e9)
        for index in range(marker_count)
    ]
    commands = [
        int((1002.0 + index) * 1e9)
        for index in range(command_count)
    ]
    _write_bag(root / 'state', [
        ('/leg_odom2', 'nav_msgs/msg/Odometry', state_stream),
        (
            '/diagnostics/lite3/leg_odom_fixed',
            'nav_msgs/msg/Odometry',
            state_stream,
        ),
        (
            '/diagnostics/lite3/motion_test_marker',
            'std_msgs/msg/String',
            markers,
        ),
        ('/imu/data', 'sensor_msgs/msg/Imu', state_stream),
        (
            '/joint_states',
            'sensor_msgs/msg/JointState',
            state_stream,
        ),
        ('/handle_state', 'geometry_msgs/msg/Twist', state_stream),
        ('/tf', 'tf2_msgs/msg/TFMessage', state_stream),
        ('/cmd_vel', 'geometry_msgs/msg/Twist', commands),
        (
            '/cmd_vel_corrected',
            'geometry_msgs/msg/Twist',
            commands,
        ),
    ])


def test_complete_capture_passes_with_zero_motion_commands(tmp_path):
    """A complete split capture is accepted without command messages."""
    _write_valid_capture(tmp_path)

    result = VALIDATOR.evaluate_controlled_capture(tmp_path)

    assert result['overall']
    assert result['sensor_capture']['overall']
    assert result['common_overlap_s'] >= 60.0
    assert result['fixed_to_raw_ratio'] == 1.0
    assert result['marker_count'] == 16
    assert result['command_message_count'] == 0


def test_capture_fails_when_a_command_message_was_recorded(tmp_path):
    """Any recorded command message fails the safety evidence."""
    _write_valid_capture(tmp_path, command_count=1)

    result = VALIDATOR.evaluate_controlled_capture(tmp_path)

    assert not result['overall']
    assert result['command_message_count'] == 2
    assert any(
        check['name'] == 'zero command messages'
        and not check['passed']
        for check in result['checks']
    )


def test_capture_fails_when_motion_markers_are_incomplete(tmp_path):
    """A capture without all phase boundaries is incomplete."""
    _write_valid_capture(tmp_path, marker_count=14)

    result = VALIDATOR.evaluate_controlled_capture(tmp_path)

    assert not result['overall']
    assert result['marker_count'] == 14
    assert any(
        check['name'] == 'complete motion markers'
        and not check['passed']
        for check in result['checks']
    )
