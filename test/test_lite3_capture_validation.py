"""Tests for the standalone Lite3 rosbag validator."""

import importlib.util
import sqlite3
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'verify_lite3_capture.py'
)
CAPTURE_SCRIPT_PATH = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'record_lite3_sensors.sh'
)
SPEC = importlib.util.spec_from_file_location(
    'verify_lite3_capture', SCRIPT_PATH)
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
                rows.append(
                    (message_id, topic_id, timestamp, b'data'))
                message_id += 1
            connection.executemany(
                'INSERT INTO messages VALUES (?, ?, ?, ?)',
                rows,
            )


def _write_valid_capture(
        root,
        image_rate_hz=9.5,
        camera_info_rate_hz=9.5,
        camera_duration_s=62.0):
    _write_bag(root / 'imu', [
        (
            '/timefix/imu',
            'sensor_msgs/msg/Imu',
            _timestamps(1000.0, 62.0, 200.0),
        ),
    ])
    _write_bag(root / 'lidar', [
        (
            '/timefix/lidar',
            'livox_ros_driver2/msg/CustomMsg',
            _timestamps(1000.1, 62.0, 10.0),
        ),
        (
            '/tf_static',
            'tf2_msgs/msg/TFMessage',
            [int(1000.1 * 1e9)],
        ),
    ])
    _write_bag(root / 'camera', [
        (
            '/camera/color/image_raw',
            'sensor_msgs/msg/Image',
            _timestamps(
                1000.2, camera_duration_s, image_rate_hz),
        ),
        (
            '/camera/color/camera_info',
            'sensor_msgs/msg/CameraInfo',
            _timestamps(
                1000.2, camera_duration_s, camera_info_rate_hz),
        ),
    ])


def test_valid_capture_passes(tmp_path):
    _write_valid_capture(tmp_path)
    result = VALIDATOR.evaluate_capture(
        tmp_path,
        recorder_statuses={
            'imu': 124,
            'lidar': 124,
            'camera': 124,
        },
    )

    assert result['overall']
    assert result['common_overlap_s'] >= 60.0
    assert result['camera_count_difference_ratio'] == 0.0
    assert abs(
        result['topics']['/timefix/imu']['rate_hz'] - 200.0
    ) < 1e-9

    completed = subprocess.run(
        [
            str(CAPTURE_SCRIPT_PATH),
            '--validate-only',
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert 'OVERALL=PASS' in completed.stdout


def test_camera_count_mismatch_fails(tmp_path):
    _write_valid_capture(
        tmp_path,
        image_rate_hz=9.0,
        camera_info_rate_hz=15.0,
    )
    result = VALIDATOR.evaluate_capture(tmp_path)

    assert not result['overall']
    assert result['camera_count_difference_ratio'] > 0.02
    assert any(
        check['name'] == 'camera image/info count difference'
        and not check['passed']
        for check in result['checks']
    )


def test_short_common_overlap_fails(tmp_path):
    _write_valid_capture(tmp_path, camera_duration_s=40.0)
    result = VALIDATOR.evaluate_capture(tmp_path)

    assert not result['overall']
    assert result['common_overlap_s'] < 60.0


def test_long_camera_outage_fails_despite_acceptable_average(tmp_path):
    _write_valid_capture(
        tmp_path,
        image_rate_hz=15.0,
        camera_info_rate_hz=15.0,
    )
    database_path = next((tmp_path / 'camera').glob('*.db3'))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE timestamp > ?
              AND timestamp < ?
        """, (int(1020.0e9), int(1040.0e9)))

    result = VALIDATOR.evaluate_capture(tmp_path)

    image = result['topics']['/camera/color/image_raw']
    assert image['rate_hz'] >= 8.0
    assert image['max_gap_s'] > 1.0
    assert not result['overall']
    assert any(
        check['name'] == (
            '/camera/color/image_raw maximum bag receipt gap')
        and not check['passed']
        for check in result['checks']
    )


def test_subsecond_camera_receipt_gap_warns_but_passes(tmp_path):
    _write_valid_capture(
        tmp_path,
        image_rate_hz=15.0,
        camera_info_rate_hz=15.0,
    )
    database_path = next((tmp_path / 'camera').glob('*.db3'))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE timestamp > ?
              AND timestamp < ?
        """, (int(1020.0e9), int(1020.6e9)))

    result = VALIDATOR.evaluate_capture(tmp_path)

    image = result['topics']['/camera/color/image_raw']
    assert 0.5 < image['max_gap_s'] <= 1.0
    assert result['overall']
    assert any(
        '/camera/color/image_raw bag receipt gap'
        in warning
        for warning in result['warnings']
    )


def test_half_second_imu_receipt_gap_warns_but_passes(tmp_path):
    _write_valid_capture(tmp_path)
    database_path = next((tmp_path / 'imu').glob('*.db3'))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE timestamp > ?
              AND timestamp < ?
        """, (int(1020.0e9), int(1020.52e9)))

    result = VALIDATOR.evaluate_capture(tmp_path)

    imu = result['topics']['/timefix/imu']
    assert 0.5 < imu['max_gap_s'] < 1.0
    assert result['overall']
    assert any(
        '/timefix/imu bag receipt gap'
        in warning
        for warning in result['warnings']
    )


def test_one_second_imu_receipt_gap_warns_but_passes(tmp_path):
    _write_valid_capture(tmp_path)
    database_path = next((tmp_path / 'imu').glob('*.db3'))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE timestamp > ?
              AND timestamp < ?
        """, (int(1020.0e9), int(1021.0e9)))

    result = VALIDATOR.evaluate_capture(tmp_path)

    imu = result['topics']['/timefix/imu']
    assert imu['max_gap_s'] == 1.0
    assert result['overall']
    assert any(
        '/timefix/imu bag receipt gap'
        in warning
        for warning in result['warnings']
    )


def test_imu_receipt_gap_over_one_second_fails(tmp_path):
    _write_valid_capture(tmp_path)
    database_path = next((tmp_path / 'imu').glob('*.db3'))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE timestamp > ?
              AND timestamp < ?
        """, (int(1020.0e9), int(1021.01e9)))

    result = VALIDATOR.evaluate_capture(tmp_path)

    imu = result['topics']['/timefix/imu']
    assert imu['max_gap_s'] > 1.0
    assert not result['overall']
    assert any(
        check['name'] == (
            '/timefix/imu maximum bag receipt gap')
        and not check['passed']
        for check in result['checks']
    )


def test_missing_bag_reports_validation_error(tmp_path):
    _write_valid_capture(tmp_path)
    for database_path in (tmp_path / 'camera').glob('*.db3'):
        database_path.unlink()

    result = VALIDATOR.evaluate_capture(tmp_path)

    assert not result['overall']
    assert result['validation_errors']


def test_capture_script_hardens_camera_recording():
    source = CAPTURE_SCRIPT_PATH.read_text(encoding='utf-8')

    assert '/camera/color/image_raw:' not in source
    assert '/camera/color/camera_info:' not in source
    assert source.count('--qos-profile-overrides-path "$QOS_FILE"') == 1
    camera_recorder = source.split('-o "$RUN_DIR/camera"', maxsplit=1)[0]
    camera_recorder = camera_recorder.rsplit(
        'setsid timeout', maxsplit=1)[-1]
    assert '--max-cache-size 0' in camera_recorder
    assert '--qos-profile-overrides-path' not in camera_recorder
    assert 'A non-zero cache left' in source
    assert 'reliable depth 100 accumulated' in source
