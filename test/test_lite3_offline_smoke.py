"""Tests for the Lite3 offline structural-smoke validator."""

import importlib.util
import math
import sqlite3
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'verify_lite3_offline_smoke.py'
)
SPEC = importlib.util.spec_from_file_location(
    'verify_lite3_offline_smoke', SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_ROOT / 'scripts' / 'run_lite3_offline_smoke.sh')
FAST_LIO_CONFIG_PATH = (
    PROJECT_ROOT / 'config' / 'fast_lio_lite3_offline.yaml')


def _write_bag(path, topics):
    topics = list(topics)
    path.mkdir(parents=True)
    database = path / '{}_0.db3'.format(path.name)
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE topics(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL
            );
            CREATE TABLE messages(
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
        """)
        pending_messages = []
        for topic_id, (name, message_type, timestamps) in enumerate(
                topics, start=1):
            connection.execute(
                'INSERT INTO topics(id, name, type) VALUES (?, ?, ?)',
                (topic_id, name, message_type),
            )
            for timestamp in timestamps:
                pending_messages.append((int(timestamp), topic_id))
        for message_id, (timestamp, topic_id) in enumerate(
                sorted(pending_messages), start=1):
            connection.execute(
                'INSERT INTO messages(id, topic_id, timestamp, data) '
                'VALUES (?, ?, ?, ?)',
                (message_id, topic_id, timestamp, b'x'),
            )

    metadata = {
        'rosbag2_bagfile_information': {
            'version': 5,
            'storage_identifier': 'sqlite3',
            'message_count': len(pending_messages),
            'topics_with_message_count': [
                {
                    'topic_metadata': {
                        'name': name,
                        'type': message_type,
                        'serialization_format': 'cdr',
                        'offered_qos_profiles': '',
                    },
                    'message_count': len(timestamps),
                }
                for name, message_type, timestamps in topics
            ],
            'relative_file_paths': [database.name],
        },
    }
    (path / 'metadata.yaml').write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding='utf-8',
    )


def _valid_split_and_merged(tmp_path, include_tf=False):
    start = 1_000_000_000
    end = start + 61_000_000_000
    timestamp_pair = [start, end]
    input_run = tmp_path / 'input'

    topic_rows = {}
    for name, message_type in MODULE.INPUT_TOPICS.items():
        topic_rows[name] = (name, message_type, timestamp_pair)

    _write_bag(
        input_run / 'imu',
        [topic_rows['/timefix/imu']],
    )
    _write_bag(
        input_run / 'lidar',
        [topic_rows['/timefix/lidar']],
    )
    _write_bag(
        input_run / 'camera',
        [
            topic_rows['/camera/color/image_raw'],
            topic_rows['/camera/color/camera_info'],
        ],
    )

    merged_topics = list(topic_rows.values())
    if include_tf:
        merged_topics.append(
            ('/tf', 'tf2_msgs/msg/TFMessage', timestamp_pair))
    merged = tmp_path / 'merged'
    _write_bag(merged, merged_topics)
    return input_run, merged


def _odom_sample(stamp_ns, x=0.0, yaw_quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        'stamp_ns': stamp_ns,
        'position': (x, 0.0, 0.0),
        'orientation': yaw_quaternion,
        'frame_id': 'odom',
        'child_frame_id': 'base_link',
    }


def test_prepared_validation_accepts_exact_four_topic_merge(tmp_path):
    input_run, merged = _valid_split_and_merged(tmp_path)

    result = MODULE.validate_prepared(input_run, merged)

    assert result['passed']
    assert math.isclose(result['common_receipt_overlap_s'], 61.0)
    assert all(check['passed'] for check in result['checks'])


def test_prepared_validation_rejects_replayed_tf(tmp_path):
    input_run, merged = _valid_split_and_merged(
        tmp_path, include_tf=True)

    result = MODULE.validate_prepared(input_run, merged)

    assert not result['passed']
    failed_names = {
        check['name']
        for check in result['checks']
        if not check['passed']
    }
    assert 'merged contains only four sensor topics' in failed_names
    assert 'source tf excluded from replay' in failed_names


def test_prepared_validation_rejects_non_monotonic_merge_order(tmp_path):
    input_run, merged = _valid_split_and_merged(tmp_path)
    database = next(merged.glob('*.db3'))
    with sqlite3.connect(database) as connection:
        connection.execute('UPDATE messages SET id = -1 WHERE id = 4')
        connection.execute('UPDATE messages SET id = 4 WHERE id = 5')
        connection.execute('UPDATE messages SET id = 5 WHERE id = -1')

    result = MODULE.validate_prepared(input_run, merged)

    assert not result['passed']
    failed_names = {
        check['name']
        for check in result['checks']
        if not check['passed']
    }
    assert 'merged receipt order' in failed_names


def test_corrupt_sqlite_is_rejected(tmp_path):
    bag = tmp_path / 'corrupt'
    bag.mkdir()
    (bag / 'corrupt_0.db3').write_bytes(b'not a sqlite database')
    metadata = {
        'rosbag2_bagfile_information': {
            'version': 5,
            'storage_identifier': 'sqlite3',
            'message_count': 0,
            'topics_with_message_count': [],
            'relative_file_paths': ['corrupt_0.db3'],
        },
    }
    (bag / 'metadata.yaml').write_text(
        yaml.safe_dump(metadata),
        encoding='utf-8',
    )

    with pytest.raises(MODULE.ValidationError, match='cannot read'):
        MODULE.read_bag_summary(bag)


def test_static_odometry_metrics_pass_at_ten_hz():
    samples = [
        _odom_sample(index * 100_000_000, x=index * 0.00001)
        for index in range(701)
    ]

    result = MODULE.evaluate_odometry(samples)

    assert result['post_warmup_count'] == 601
    assert math.isclose(result['duration_s'], 60.0)
    assert abs(result['rate_hz'] - 10.0) < 1e-12
    assert result['non_monotonic'] == 0
    assert result['non_finite'] == 0
    assert result['invalid_frame'] == 0
    assert result['final_translation_m'] < 0.01
    assert result['maximum_step_translation_m'] < 0.001


def test_odometry_metrics_expose_time_and_drift_failures():
    samples = [
        _odom_sample(0, x=0.0),
        _odom_sample(10_000_000_000, x=0.0),
        _odom_sample(11_000_000_000, x=0.4),
        _odom_sample(10_500_000_000, x=0.8),
    ]

    result = MODULE.evaluate_odometry(samples)

    assert result['non_monotonic'] == 1
    assert result['final_translation_m'] == 0.8
    assert result['maximum_step_translation_m'] == 0.4


def test_odometry_checks_timestamp_order_before_warmup():
    samples = [
        _odom_sample(0),
        _odom_sample(5_000_000_000),
        _odom_sample(4_000_000_000),
        _odom_sample(10_000_000_000),
        _odom_sample(11_000_000_000),
    ]

    result = MODULE.evaluate_odometry(samples)

    assert result['non_monotonic'] == 1


def _xyz_cloud(data):
    fields = [
        SimpleNamespace(name='x', offset=0, datatype=7, count=1),
        SimpleNamespace(name='y', offset=4, datatype=7, count=1),
        SimpleNamespace(name='z', offset=8, datatype=7, count=1),
    ]
    return SimpleNamespace(
        width=1,
        height=1,
        point_step=12,
        row_step=12,
        data=data,
        fields=fields,
        is_bigendian=False,
    )


def test_pointcloud_xyz_validation_rejects_nan_and_truncation():
    valid = MODULE._pointcloud_xyz_validity(
        _xyz_cloud(struct.pack('<fff', 1.0, 2.0, 3.0)))
    nan_cloud = MODULE._pointcloud_xyz_validity(
        _xyz_cloud(struct.pack('<fff', math.nan, 2.0, 3.0)))
    truncated = MODULE._pointcloud_xyz_validity(
        _xyz_cloud(struct.pack('<ff', 1.0, 2.0)))

    assert valid['layout_valid']
    assert valid['xyz_schema_valid']
    assert valid['non_finite_xyz_points'] == 0
    assert nan_cloud['non_finite_xyz_points'] == 1
    assert not truncated['layout_valid']


def test_quaternion_angle_uses_shortest_sign_equivalence():
    identity = (0.0, 0.0, 0.0, 1.0)
    negative_identity = (0.0, 0.0, 0.0, -1.0)

    assert MODULE.quaternion_angle_degrees(
        identity, negative_identity) == 0.0


def test_runtime_validator_accepts_13_class_lite3_logits():
    state = {'arrays': {}}
    message = SimpleNamespace(data=[0.0] * (3 * 4 * 13))

    MODULE._inspect_runtime_message(state, '/clip_logits', message)

    assert state['arrays']['/clip_logits']['count'] == 1
    assert state['arrays']['/clip_logits']['invalid_length_count'] == 0


def test_report_never_calls_structural_pass_motion_ready():
    result = {
        'overall': MODULE.PASS_STATUS,
        'passed': True,
        'warnings': ['calibration is unverified'],
    }

    report = MODULE.render_report(result)

    assert 'ALGORITHM_STATIC_PASS_NON_GEOMETRIC' in report
    assert 'MOTION_READY' not in report


def test_offline_runner_is_valid_bash_and_avoids_broad_process_kills():
    subprocess.run(
        ['bash', '-n', str(RUNNER_PATH)],
        check=True,
    )
    source = RUNNER_PATH.read_text(encoding='utf-8')

    assert 'ROS_LOCALHOST_ONLY=1' in source
    assert 'active_perception_node' in source
    assert 'pkill' not in source
    assert 'killall' not in source
    assert 'kill -INT -- "-$pid"' in source
    assert '$WORK_ROOT/.transfers' in source
    assert 'CLIP_OFFLINE_PREFLIGHT=PASS' in source
    assert 'query_after_nonempty_cloud.py' in source
    assert "int(message.width) * int(message.height) > 0" in source
    assert "ga_parameters['pointcloud_frame'] = 'base_link'" in source
    assert 'get_publishers_info_by_topic' in source
    assert 'git-status.txt' in source


def test_fast_lio_offline_config_freezes_unverified_extrinsics():
    configuration = yaml.safe_load(
        FAST_LIO_CONFIG_PATH.read_text(encoding='utf-8'))
    parameters = configuration['/**']['ros__parameters']

    assert parameters['common']['lid_topic'] == '/timefix/lidar'
    assert parameters['common']['imu_topic'] == '/timefix/imu'
    assert parameters['common']['time_sync_en'] is False
    assert parameters['common']['time_offset_lidar_to_imu'] == 0.0
    assert parameters['preprocess']['lidar_type'] == 1
    assert parameters['preprocess']['scan_line'] == 4
    assert parameters['preprocess']['scan_rate'] == 10
    assert parameters['mapping']['extrinsic_est_en'] is False
    assert parameters['mapping']['extrinsic_R'] == [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    assert parameters['publish']['map_en'] is False
    assert parameters['publish']['scan_publish_en'] is True
    assert parameters['publish']['dense_publish_en'] is False
    assert parameters['publish']['scan_bodyframe_pub_en'] is False
    assert parameters['pcd_save']['pcd_save_en'] is False
