#!/usr/bin/env python3
"""Validate prepared and completed Lite3 offline structural smoke tests."""

import argparse
from collections import Counter
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml


INPUT_TOPICS = {
    '/timefix/imu': 'sensor_msgs/msg/Imu',
    '/timefix/lidar': 'livox_ros_driver2/msg/CustomMsg',
    '/camera/color/image_raw': 'sensor_msgs/msg/Image',
    '/camera/color/camera_info': 'sensor_msgs/msg/CameraInfo',
}

OUTPUT_TOPICS = {
    '/clock': 'rosgraph_msgs/msg/Clock',
    '/Odometry': 'nav_msgs/msg/Odometry',
    '/path': 'nav_msgs/msg/Path',
    '/tf': 'tf2_msgs/msg/TFMessage',
    '/cloud_registered': 'sensor_msgs/msg/PointCloud2',
    '/clip_logits': 'std_msgs/msg/Float32MultiArray',
    '/clip_features': 'std_msgs/msg/Float32MultiArray',
    '/query_feature': 'std_msgs/msg/Float32MultiArray',
    '/semantic_cloud': 'sensor_msgs/msg/PointCloud2',
    '/uncertainty_cloud': 'sensor_msgs/msg/PointCloud2',
    '/voxel_entropy_data': 'sensor_msgs/msg/PointCloud2',
    '/semantic_cost_map': 'nav_msgs/msg/OccupancyGrid',
    '/text_query': 'std_msgs/msg/String',
    '/query_target_pose': 'geometry_msgs/msg/PoseStamped',
    '/goal_pose': 'geometry_msgs/msg/PoseStamped',
}

SOURCE_BAG_FOR_TOPIC = {
    '/timefix/imu': 'imu',
    '/timefix/lidar': 'lidar',
    '/camera/color/image_raw': 'camera',
    '/camera/color/camera_info': 'camera',
}

PASS_STATUS = 'ALGORITHM_STATIC_PASS_NON_GEOMETRIC'
PREPARED_STATUS = 'PREPARED'
FAIL_STATUS = 'FAIL'


class ValidationError(RuntimeError):
    """Report an unreadable bag or unavailable ROS message dependency."""


def empty_topic_summary():
    """Return a mutable empty topic summary."""
    return {
        'count': 0,
        'types': set(),
        'first_ns': None,
        'last_ns': None,
    }


def database_paths(bag_directory):
    """Return metadata-ordered SQLite databases in one rosbag directory."""
    path = Path(bag_directory).expanduser().resolve()
    metadata_path = path / 'metadata.yaml'
    try:
        metadata_document = yaml.safe_load(
            metadata_path.read_text(encoding='utf-8'))
        metadata = metadata_document['rosbag2_bagfile_information']
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValidationError(
            'cannot read rosbag metadata {}: {}'.format(
                metadata_path, error)) from error

    if metadata.get('storage_identifier') != 'sqlite3':
        raise ValidationError(
            '{} storage_identifier is not sqlite3'.format(metadata_path))
    relative_paths = metadata.get('relative_file_paths')
    if not isinstance(relative_paths, list) or not relative_paths:
        raise ValidationError(
            '{} has no relative_file_paths'.format(metadata_path))

    databases = []
    for relative_path in relative_paths:
        candidate = Path(str(relative_path))
        if (
            candidate.is_absolute()
            or candidate.name != str(candidate)
            or candidate.suffix != '.db3'
        ):
            raise ValidationError(
                '{} contains unsafe database path {!r}'.format(
                    metadata_path, relative_path))
        database = path / candidate
        if not database.is_file():
            raise ValidationError(
                'metadata database does not exist: {}'.format(database))
        databases.append(database)

    discovered = set(path.glob('*.db3'))
    if discovered != set(databases):
        raise ValidationError(
            'metadata/database file set differs in {}'.format(path))
    return path, databases, metadata


def read_bag_summary(bag_directory):
    """Read SQLite integrity, types, counts and receipt endpoints."""
    path, databases, metadata = database_paths(bag_directory)
    topics = {}
    receipt_count = 0
    receipt_non_monotonic = 0
    previous_receipt_ns = None

    for database in databases:
        try:
            uri = 'file:{}?mode=ro'.format(database)
            with sqlite3.connect(uri, uri=True) as connection:
                quick_check = connection.execute(
                    'PRAGMA quick_check').fetchall()
                if quick_check != [('ok',)]:
                    raise ValidationError(
                        '{} failed PRAGMA quick_check: {}'.format(
                            database, quick_check))

                for topic_id, name, message_type in connection.execute(
                        'SELECT id, name, type FROM topics'):
                    item = topics.setdefault(name, empty_topic_summary())
                    item['types'].add(message_type)
                    count, first_ns, last_ns = connection.execute(
                        'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) '
                        'FROM messages WHERE topic_id = ?',
                        (topic_id,),
                    ).fetchone()
                    item['count'] += int(count)
                    if first_ns is not None:
                        item['first_ns'] = (
                            int(first_ns)
                            if item['first_ns'] is None
                            else min(item['first_ns'], int(first_ns))
                        )
                        item['last_ns'] = (
                            int(last_ns)
                            if item['last_ns'] is None
                            else max(item['last_ns'], int(last_ns))
                        )
                for (receipt_ns,) in connection.execute(
                        'SELECT timestamp FROM messages ORDER BY id'):
                    receipt_ns = int(receipt_ns)
                    receipt_count += 1
                    if (
                        previous_receipt_ns is not None
                        and receipt_ns < previous_receipt_ns
                    ):
                        receipt_non_monotonic += 1
                    previous_receipt_ns = receipt_ns
        except sqlite3.DatabaseError as error:
            raise ValidationError(
                'cannot read {}: {}'.format(database, error)) from error

    serializable = {}
    for name, item in topics.items():
        converted = dict(item)
        converted['types'] = sorted(item['types'])
        serializable[name] = converted

    metadata_topics = {}
    try:
        for entry in metadata.get('topics_with_message_count', []):
            topic_metadata = entry['topic_metadata']
            metadata_topics[topic_metadata['name']] = {
                'type': topic_metadata['type'],
                'count': int(entry['message_count']),
            }
        metadata_count = int(metadata['message_count'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(
            'invalid topic/count metadata in {}: {}'.format(
                path / 'metadata.yaml', error)) from error

    sqlite_topic_metadata = {
        name: {
            'type': (
                item['types'][0]
                if len(item['types']) == 1
                else '|'.join(item['types'])
            ),
            'count': item['count'],
        }
        for name, item in serializable.items()
    }
    if metadata_count != receipt_count:
        raise ValidationError(
            'metadata message_count={} but SQLite count={} in {}'.format(
                metadata_count, receipt_count, path))
    if metadata_topics != sqlite_topic_metadata:
        raise ValidationError(
            'metadata topic type/count differs from SQLite in {}'.format(
                path))
    return {
        'path': str(path),
        'database_count': len(databases),
        'topics': serializable,
        'receipt_count': receipt_count,
        'receipt_non_monotonic': receipt_non_monotonic,
    }


def add_check(result, name, passed, detail):
    """Append one machine-readable check."""
    result['checks'].append({
        'name': name,
        'passed': bool(passed),
        'detail': str(detail),
    })


def topic_duration_s(item):
    """Return endpoint duration for a topic summary."""
    if (
        item is None
        or item.get('first_ns') is None
        or item.get('last_ns') is None
        or item['last_ns'] <= item['first_ns']
    ):
        return None
    return (item['last_ns'] - item['first_ns']) * 1e-9


def validate_prepared(input_run, merged_bag):
    """Check split source bags and a single four-topic merged bag."""
    input_run = Path(input_run).expanduser().resolve()
    result = {
        'input_run': str(input_run),
        'merged_bag': str(Path(merged_bag).expanduser().resolve()),
        'checks': [],
        'warnings': [],
        'source_topics': {},
        'merged_topics': {},
        'common_receipt_overlap_s': None,
    }

    source_summaries = {}
    for bag_name in ('imu', 'lidar', 'camera'):
        source_summaries[bag_name] = read_bag_summary(
            input_run / bag_name)

    merged_summary = read_bag_summary(merged_bag)

    first_times = []
    last_times = []
    for topic, expected_type in INPUT_TOPICS.items():
        source_item = source_summaries[
            SOURCE_BAG_FOR_TOPIC[topic]]['topics'].get(topic)
        merged_item = merged_summary['topics'].get(topic)
        result['source_topics'][topic] = source_item
        result['merged_topics'][topic] = merged_item

        add_check(
            result,
            '{} source type'.format(topic),
            source_item is not None
            and source_item['types'] == [expected_type],
            'expected={}, actual={}'.format(
                expected_type,
                None if source_item is None else source_item['types'],
            ),
        )
        add_check(
            result,
            '{} merged type'.format(topic),
            merged_item is not None
            and merged_item['types'] == [expected_type],
            'expected={}, actual={}'.format(
                expected_type,
                None if merged_item is None else merged_item['types'],
            ),
        )

        source_count = 0 if source_item is None else source_item['count']
        merged_count = 0 if merged_item is None else merged_item['count']
        add_check(
            result,
            '{} count conservation'.format(topic),
            source_count > 0 and source_count == merged_count,
            'source={}, merged={}'.format(source_count, merged_count),
        )

        endpoints_equal = (
            source_item is not None
            and merged_item is not None
            and source_item['first_ns'] == merged_item['first_ns']
            and source_item['last_ns'] == merged_item['last_ns']
        )
        add_check(
            result,
            '{} receipt endpoints'.format(topic),
            endpoints_equal,
            'source={}..{}, merged={}..{}'.format(
                None if source_item is None else source_item['first_ns'],
                None if source_item is None else source_item['last_ns'],
                None if merged_item is None else merged_item['first_ns'],
                None if merged_item is None else merged_item['last_ns'],
            ),
        )

        if source_item is not None and source_item['count']:
            first_times.append(source_item['first_ns'])
            last_times.append(source_item['last_ns'])

    merged_topic_names = set(merged_summary['topics'])
    add_check(
        result,
        'merged contains only four sensor topics',
        merged_topic_names == set(INPUT_TOPICS),
        'actual={}'.format(sorted(merged_topic_names)),
    )
    add_check(
        result,
        'source tf excluded from replay',
        '/tf' not in merged_topic_names
        and '/tf_static' not in merged_topic_names,
        'merged topics={}'.format(sorted(merged_topic_names)),
    )
    add_check(
        result,
        'merged receipt order',
        merged_summary['receipt_non_monotonic'] == 0,
        'non_monotonic={}'.format(
            merged_summary['receipt_non_monotonic']),
    )

    if len(first_times) == len(INPUT_TOPICS):
        overlap_ns = min(last_times) - max(first_times)
        result['common_receipt_overlap_s'] = max(0.0, overlap_ns * 1e-9)
    overlap_s = result['common_receipt_overlap_s']
    add_check(
        result,
        'four-topic receipt overlap',
        overlap_s is not None and overlap_s >= 60.0,
        'expected>=60.000s, actual={}'.format(
            'missing'
            if overlap_s is None
            else '{:.3f}s'.format(overlap_s)),
    )

    result['passed'] = all(
        check['passed'] for check in result['checks'])
    return result


def quaternion_norm(quaternion):
    """Return Euclidean norm of an xyzw quaternion."""
    return math.sqrt(sum(float(value) ** 2 for value in quaternion))


def quaternion_angle_degrees(first, second):
    """Return shortest angular distance between two xyzw quaternions."""
    first_norm = quaternion_norm(first)
    second_norm = quaternion_norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return math.inf
    dot = sum(
        float(left) * float(right)
        for left, right in zip(first, second)
    ) / (first_norm * second_norm)
    dot = min(1.0, max(-1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def distance(first, second):
    """Return Euclidean distance between two xyz positions."""
    return math.sqrt(sum(
        (float(left) - float(right)) ** 2
        for left, right in zip(first, second)
    ))


def evaluate_odometry(samples, warmup_s=10.0):
    """Compute static-drift metrics from timestamped odometry samples."""
    result = {
        'count': len(samples),
        'post_warmup_count': 0,
        'duration_s': None,
        'rate_hz': None,
        'non_monotonic': 0,
        'non_finite': 0,
        'invalid_frame': 0,
        'final_translation_m': None,
        'maximum_radius_m': None,
        'final_angle_deg': None,
        'maximum_angle_deg': None,
        'maximum_step_translation_m': None,
        'maximum_step_angle_deg': None,
    }
    if not samples:
        return result

    first_stamp = samples[0]['stamp_ns']
    selected = [
        sample for sample in samples
        if sample['stamp_ns'] >= first_stamp + int(warmup_s * 1e9)
    ]
    result['post_warmup_count'] = len(selected)
    if not selected:
        return result

    previous_stamp = None
    for sample in samples:
        values = (
            list(sample['position'])
            + list(sample['orientation'])
        )
        if not all(math.isfinite(float(value)) for value in values):
            result['non_finite'] += 1
        if (
            sample.get('frame_id') != 'odom'
            or sample.get('child_frame_id') != 'base_link'
        ):
            result['invalid_frame'] += 1
        if previous_stamp is not None and sample['stamp_ns'] <= previous_stamp:
            result['non_monotonic'] += 1
        previous_stamp = sample['stamp_ns']

    if len(selected) >= 2:
        duration_s = (
            selected[-1]['stamp_ns'] - selected[0]['stamp_ns']
        ) * 1e-9
        result['duration_s'] = duration_s
        if duration_s > 0.0:
            result['rate_hz'] = (len(selected) - 1) / duration_s

    origin_position = selected[0]['position']
    origin_orientation = selected[0]['orientation']
    result['final_translation_m'] = distance(
        origin_position, selected[-1]['position'])
    result['maximum_radius_m'] = max(
        distance(origin_position, sample['position'])
        for sample in selected
    )
    result['final_angle_deg'] = quaternion_angle_degrees(
        origin_orientation, selected[-1]['orientation'])
    result['maximum_angle_deg'] = max(
        quaternion_angle_degrees(
            origin_orientation, sample['orientation'])
        for sample in selected
    )

    if len(selected) >= 2:
        result['maximum_step_translation_m'] = max(
            distance(previous['position'], current['position'])
            for previous, current in zip(selected[:-1], selected[1:])
        )
        result['maximum_step_angle_deg'] = max(
            quaternion_angle_degrees(
                previous['orientation'], current['orientation'])
            for previous, current in zip(selected[:-1], selected[1:])
        )
    return result


def _stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


def _finite(values):
    return all(math.isfinite(float(value)) for value in values)


def inspect_runtime_bag(output_bag):
    """Deserialize the runtime output topics and collect validation metrics."""
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except (ImportError, ModuleNotFoundError) as error:
        raise ValidationError(
            'ROS 2 Python message support is unavailable; source '
            '/opt/ros/humble and the workspace overlay: {}'.format(error)
        ) from error

    path, databases, unused_metadata = database_paths(output_bag)
    message_classes = {}
    state = {
        'path': str(path),
        'clock': {
            'count': 0,
            'first_ns': None,
            'last_ns': None,
            'non_monotonic': 0,
            'duplicate_count': 0,
            'zero_count': 0,
        },
        'path_messages': {
            'count': 0,
            'valid_count': 0,
            'pose_count': 0,
        },
        'odom_samples': [],
        'tf': {
            'message_count': 0,
            'odom_base_count': 0,
            'base_parents': set(),
            'invalid_transform_count': 0,
        },
        'clouds': {},
        'arrays': {},
        'costmaps': {
            'count': 0,
            'valid_count': 0,
        },
        'message_counts': {},
    }

    for database in databases:
        uri = 'file:{}?mode=ro'.format(database)
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                connection.execute('PRAGMA query_only = ON')
                topic_rows = connection.execute(
                    'SELECT id, name, type FROM topics').fetchall()
                topics_by_id = {
                    int(topic_id): (name, message_type)
                    for topic_id, name, message_type in topic_rows
                }
                rows = connection.execute(
                    'SELECT topic_id, timestamp, data '
                    'FROM messages ORDER BY id')
                for topic_id, unused_receipt_ns, serialized in rows:
                    topic_name, message_type = topics_by_id[int(topic_id)]
                    if topic_name not in OUTPUT_TOPICS:
                        continue
                    state['message_counts'][topic_name] = (
                        state['message_counts'].get(topic_name, 0) + 1)

                    expected_type = OUTPUT_TOPICS[topic_name]
                    if message_type != expected_type:
                        raise ValidationError(
                            '{} has type {}, expected {}'.format(
                                topic_name, message_type, expected_type))
                    if message_type not in message_classes:
                        try:
                            message_classes[message_type] = get_message(
                                message_type)
                        except Exception as error:
                            raise ValidationError(
                                'cannot load {}: {}'.format(
                                    message_type, error)) from error
                    try:
                        message = deserialize_message(
                            bytes(serialized),
                            message_classes[message_type],
                        )
                    except Exception as error:
                        raise ValidationError(
                            'cannot deserialize {} in {}: {}'.format(
                                topic_name, database, error)) from error
                    _inspect_runtime_message(state, topic_name, message)
        except sqlite3.DatabaseError as error:
            raise ValidationError(
                'cannot inspect {}: {}'.format(database, error)) from error

    state['tf']['base_parents'] = sorted(state['tf']['base_parents'])
    for item in state['clouds'].values():
        item['frames'] = sorted(item['frames'])
    return state


def _pointcloud_xyz_validity(message):
    """Return layout and finite xyz diagnostics for one PointCloud2."""
    width = int(message.width)
    height = int(message.height)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    expected_data_size = row_step * height
    result = {
        'layout_valid': (
            width >= 0
            and height >= 0
            and point_step > 0
            and row_step >= point_step * width
            and len(message.data) == expected_data_size
        ),
        'xyz_schema_valid': True,
        'non_finite_xyz_points': 0,
    }
    if not result['layout_valid']:
        return result

    fields = {field.name: field for field in message.fields}
    names = []
    formats = []
    offsets = []
    endian = '>' if message.is_bigendian else '<'
    datatype_formats = {
        7: 'f4',
        8: 'f8',
    }
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if (
            field is None
            or int(field.count) != 1
            or int(field.datatype) not in datatype_formats
        ):
            result['xyz_schema_valid'] = False
            return result
        field_format = datatype_formats[int(field.datatype)]
        field_dtype = np.dtype(endian + field_format)
        if (
            int(field.offset) < 0
            or int(field.offset) + field_dtype.itemsize > point_step
        ):
            result['xyz_schema_valid'] = False
            return result
        names.append(name)
        formats.append(field_dtype)
        offsets.append(int(field.offset))

    point_count = width * height
    if point_count == 0:
        return result
    try:
        dtype = np.dtype({
            'names': names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': point_step,
        })
        points = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=message.data,
            strides=(row_step, point_step),
        )
        finite = (
            np.isfinite(points['x'])
            & np.isfinite(points['y'])
            & np.isfinite(points['z'])
        )
        result['non_finite_xyz_points'] = int(
            point_count - np.count_nonzero(finite))
    except (TypeError, ValueError, BufferError):
        result['layout_valid'] = False
    return result


def _inspect_runtime_message(state, topic_name, message):
    if topic_name == '/clock':
        stamp_ns = (
            int(message.clock.sec) * 1000000000
            + int(message.clock.nanosec)
        )
        clock = state['clock']
        if clock['last_ns'] is not None:
            if stamp_ns < clock['last_ns']:
                clock['non_monotonic'] += 1
            elif stamp_ns == clock['last_ns']:
                clock['duplicate_count'] += 1
        if stamp_ns == 0:
            clock['zero_count'] += 1
        if clock['first_ns'] is None:
            clock['first_ns'] = stamp_ns
        clock['last_ns'] = stamp_ns
        clock['count'] += 1
        return

    if topic_name == '/path':
        item = state['path_messages']
        item['count'] += 1
        item['pose_count'] += len(message.poses)
        if (
            message.header.frame_id == 'odom'
            and len(message.poses) > 0
            and all(pose.header.frame_id == 'odom' for pose in message.poses)
        ):
            item['valid_count'] += 1
        return

    if topic_name == '/Odometry':
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        state['odom_samples'].append({
            'stamp_ns': _stamp_ns(message),
            'position': (position.x, position.y, position.z),
            'orientation': (
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
            'frame_id': message.header.frame_id,
            'child_frame_id': message.child_frame_id,
        })
        return

    if topic_name == '/tf':
        state['tf']['message_count'] += 1
        for transform in message.transforms:
            parent = transform.header.frame_id.strip('/')
            child = transform.child_frame_id.strip('/')
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            values = (
                translation.x,
                translation.y,
                translation.z,
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )
            if not _finite(values) or quaternion_norm(values[3:]) <= 1e-12:
                state['tf']['invalid_transform_count'] += 1
            if child == 'base_link':
                state['tf']['base_parents'].add(parent)
                if parent == 'odom':
                    state['tf']['odom_base_count'] += 1
        return

    if topic_name in {
        '/cloud_registered',
        '/semantic_cloud',
        '/uncertainty_cloud',
        '/voxel_entropy_data',
    }:
        item = state['clouds'].setdefault(topic_name, {
            'count': 0,
            'nonempty_count': 0,
            'frames': set(),
            'invalid_layout_count': 0,
            'invalid_xyz_schema_count': 0,
            'non_finite_xyz_messages': 0,
            'non_finite_xyz_points': 0,
            'zero_stamp_count': 0,
            'non_monotonic_stamps': 0,
            'stamps_ns': [],
        })
        item['count'] += 1
        item['frames'].add(message.header.frame_id)
        stamp_ns = _stamp_ns(message)
        if stamp_ns == 0:
            item['zero_stamp_count'] += 1
        if item['stamps_ns'] and stamp_ns <= item['stamps_ns'][-1]:
            item['non_monotonic_stamps'] += 1
        item['stamps_ns'].append(stamp_ns)
        point_count = int(message.width) * int(message.height)
        if point_count > 0 and len(message.data) > 0:
            item['nonempty_count'] += 1
        validity = _pointcloud_xyz_validity(message)
        if not validity['layout_valid']:
            item['invalid_layout_count'] += 1
        if not validity['xyz_schema_valid']:
            item['invalid_xyz_schema_count'] += 1
        if validity['non_finite_xyz_points']:
            item['non_finite_xyz_messages'] += 1
            item['non_finite_xyz_points'] += validity[
                'non_finite_xyz_points']
        return

    if topic_name in {
        '/clip_logits',
        '/clip_features',
        '/query_feature',
    }:
        expected_length = {
            '/clip_logits': 156,
            '/clip_features': 6144,
            '/query_feature': 512,
        }[topic_name]
        item = state['arrays'].setdefault(topic_name, {
            'count': 0,
            'invalid_length_count': 0,
            'non_finite_count': 0,
            'invalid_norm_count': 0,
        })
        item['count'] += 1
        values = list(message.data)
        if len(values) != expected_length:
            item['invalid_length_count'] += 1
        if not _finite(values):
            item['non_finite_count'] += 1
        if (
            topic_name == '/clip_features'
            and len(values) == expected_length
            and _finite(values)
        ):
            for index in range(0, len(values), 512):
                norm = math.sqrt(sum(
                    float(value) ** 2
                    for value in values[index:index + 512]
                ))
                if not 0.95 <= norm <= 1.05:
                    item['invalid_norm_count'] += 1
        elif (
            topic_name == '/query_feature'
            and len(values) == expected_length
            and _finite(values)
        ):
            norm = math.sqrt(sum(float(value) ** 2 for value in values))
            if not 0.95 <= norm <= 1.05:
                item['invalid_norm_count'] += 1
        return

    if topic_name == '/semantic_cost_map':
        state['costmaps']['count'] += 1
        expected_size = (
            int(message.info.width) * int(message.info.height))
        if (
            message.header.frame_id == 'odom'
            and expected_size > 0
            and len(message.data) == expected_size
            and float(message.info.resolution) > 0.0
        ):
            state['costmaps']['valid_count'] += 1


def _metric_check(result, name, value, minimum=None, maximum=None):
    passed = value is not None
    if passed and minimum is not None:
        passed = value >= minimum
    if passed and maximum is not None:
        passed = value <= maximum
    expected = []
    if minimum is not None:
        expected.append('>={}'.format(minimum))
    if maximum is not None:
        expected.append('<={}'.format(maximum))
    add_check(
        result,
        name,
        passed,
        'expected {}, actual={}'.format(
            ' and '.join(expected), value),
    )


def _get_cloud(runtime, topic):
    return runtime['clouds'].get(topic, {
        'count': 0,
        'nonempty_count': 0,
        'frames': set(),
        'invalid_layout_count': 0,
        'invalid_xyz_schema_count': 0,
        'non_finite_xyz_messages': 0,
        'non_finite_xyz_points': 0,
        'zero_stamp_count': 0,
        'non_monotonic_stamps': 0,
        'stamps_ns': [],
    })


def _get_array(runtime, topic):
    return runtime['arrays'].get(topic, {
        'count': 0,
        'invalid_length_count': 0,
        'non_finite_count': 0,
        'invalid_norm_count': 0,
    })


def _cloud_detail(item):
    """Return cloud diagnostics without embedding every timestamp."""
    detail = {
        key: value
        for key, value in item.items()
        if key != 'stamps_ns'
    }
    detail['stamp_count'] = len(item.get('stamps_ns', []))
    return detail


def _exact_stamp_pair_count(first_stamps, second_stamps):
    """Return multiset intersection size for two timestamp sequences."""
    first_counts = Counter(first_stamps)
    second_counts = Counter(second_stamps)
    return sum(
        min(count, second_counts.get(stamp, 0))
        for stamp, count in first_counts.items()
    )


def validate_runtime(output_bag, logs_directory, runtime_graph):
    """Validate FAST-LIO, CLIP and GA-BSVM structural outputs."""
    runtime = inspect_runtime_bag(output_bag)
    result = {
        'output_bag': str(Path(output_bag).expanduser().resolve()),
        'checks': [],
        'warnings': [],
        'runtime': runtime,
    }
    odometry = evaluate_odometry(runtime['odom_samples'])
    result['odometry'] = odometry

    clock = runtime['clock']
    clock_duration_s = None
    if (
        clock['first_ns'] is not None
        and clock['last_ns'] is not None
        and clock['last_ns'] > clock['first_ns']
    ):
        clock_duration_s = (
            clock['last_ns'] - clock['first_ns']) * 1e-9
    add_check(
        result,
        'recorded clock',
        clock['count'] > 0
        and clock_duration_s is not None
        and clock_duration_s >= 60.0
        and clock['non_monotonic'] == 0
        and clock['zero_count'] == 0,
        'count={}, duration={}, regressions={}, duplicates={}, zero={}'.format(
            clock['count'],
            clock_duration_s,
            clock['non_monotonic'],
            clock['duplicate_count'],
            clock['zero_count'],
        ),
    )

    _metric_check(result, 'odometry post-warmup duration',
                  odometry['duration_s'], minimum=50.0)
    _metric_check(result, 'odometry post-warmup rate',
                  odometry['rate_hz'], minimum=8.0, maximum=11.0)
    add_check(result, 'odometry timestamps monotonic',
              odometry['non_monotonic'] == 0,
              'non_monotonic={}'.format(odometry['non_monotonic']))
    add_check(result, 'odometry finite',
              odometry['non_finite'] == 0,
              'non_finite={}'.format(odometry['non_finite']))
    add_check(result, 'odometry frames',
              odometry['invalid_frame'] == 0,
              'expected odom->base_link, invalid={}'.format(
                  odometry['invalid_frame']))
    _metric_check(result, 'odometry final translation',
                  odometry['final_translation_m'], maximum=0.10)
    _metric_check(result, 'odometry maximum radius',
                  odometry['maximum_radius_m'], maximum=0.15)
    _metric_check(result, 'odometry final angle',
                  odometry['final_angle_deg'], maximum=1.0)
    _metric_check(result, 'odometry maximum angle',
                  odometry['maximum_angle_deg'], maximum=2.0)
    _metric_check(result, 'odometry maximum translation step',
                  odometry['maximum_step_translation_m'], maximum=0.10)
    _metric_check(result, 'odometry maximum angle step',
                  odometry['maximum_step_angle_deg'], maximum=5.0)

    tf_state = runtime['tf']
    add_check(
        result,
        'dynamic odom to base_link TF',
        tf_state['odom_base_count'] > 0
        and tf_state['base_parents'] == ['odom']
        and tf_state['invalid_transform_count'] == 0,
        'count={}, base_parents={}, invalid={}'.format(
            tf_state['odom_base_count'],
            tf_state['base_parents'],
            tf_state['invalid_transform_count'],
        ),
    )

    graph_path = Path(runtime_graph).expanduser().resolve()
    try:
        graph = json.loads(graph_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as error:
        raise ValidationError(
            'cannot read runtime graph {}: {}'.format(
                graph_path, error)) from error
    result['runtime_graph'] = graph
    for key, expected in (
        ('clock_publishers', 1),
        ('odometry_publishers', 1),
        ('tf_publishers', 1),
        ('cmd_vel_publishers', 0),
    ):
        add_check(
            result,
            '{} live graph'.format(key),
            graph.get(key) == expected,
            'expected={}, actual={}'.format(expected, graph.get(key)),
        )
    add_check(
        result,
        'publisher authority monitored during playback',
        int(graph.get('authority_sample_count', 0)) >= 1
        and int(graph.get('authority_violations', -1)) == 0,
        'samples={}, violations={}'.format(
            graph.get('authority_sample_count'),
            graph.get('authority_violations'),
        ),
    )

    registered = _get_cloud(runtime, '/cloud_registered')
    add_check(
        result,
        'registered cloud output',
        registered['nonempty_count'] > 0
        and set(registered['frames']) == {'odom'}
        and registered['invalid_layout_count'] == 0
        and registered['invalid_xyz_schema_count'] == 0
        and registered['non_finite_xyz_points'] == 0
        and registered['zero_stamp_count'] == 0
        and registered['non_monotonic_stamps'] == 0,
        str(_cloud_detail(registered)),
    )
    path_messages = runtime['path_messages']
    add_check(
        result,
        'FAST-LIO path output',
        path_messages['valid_count'] > 0
        and path_messages['valid_count'] == path_messages['count'],
        str(path_messages),
    )

    logits = _get_array(runtime, '/clip_logits')
    features = _get_array(runtime, '/clip_features')
    add_check(
        result,
        'CLIP logits',
        logits['count'] >= 10
        and logits['invalid_length_count'] == 0
        and logits['non_finite_count'] == 0,
        str(logits),
    )
    add_check(
        result,
        'CLIP features',
        features['count'] >= 10
        and features['invalid_length_count'] == 0
        and features['non_finite_count'] == 0
        and features['invalid_norm_count'] == 0,
        str(features),
    )
    add_check(
        result,
        'CLIP output pairing',
        abs(logits['count'] - features['count']) <= 1,
        'logits={}, features={}'.format(
            logits['count'], features['count']),
    )

    query = _get_array(runtime, '/query_feature')
    add_check(
        result,
        'query feature',
        query['count'] >= 1
        and query['invalid_length_count'] == 0
        and query['non_finite_count'] == 0
        and query['invalid_norm_count'] == 0,
        str(query),
    )

    semantic = _get_cloud(runtime, '/semantic_cloud')
    uncertainty = _get_cloud(runtime, '/uncertainty_cloud')
    entropy = _get_cloud(runtime, '/voxel_entropy_data')
    for topic, item in (
        ('/semantic_cloud', semantic),
        ('/uncertainty_cloud', uncertainty),
        ('/voxel_entropy_data', entropy),
    ):
        add_check(
            result,
            '{} nonempty output'.format(topic),
            item['nonempty_count'] >= 1
            and set(item['frames']) == {'odom'}
            and item['invalid_layout_count'] == 0
            and item['invalid_xyz_schema_count'] == 0
            and item['non_finite_xyz_points'] == 0
            and item['zero_stamp_count'] == 0
            and item['non_monotonic_stamps'] == 0,
            str(_cloud_detail(item)),
        )
    paired_clouds = _exact_stamp_pair_count(
        semantic['stamps_ns'],
        uncertainty['stamps_ns'],
    )
    paired_denominator = max(
        semantic['count'],
        uncertainty['count'],
        1,
    )
    paired_coverage = paired_clouds / paired_denominator
    add_check(
        result,
        'semantic and uncertainty pairing',
        abs(semantic['count'] - uncertainty['count']) <= 1
        and paired_coverage >= 0.99,
        'semantic={}, uncertainty={}, exact_stamp_pairs={}, '
        'coverage={:.2%}'.format(
            semantic['count'],
            uncertainty['count'],
            paired_clouds,
            paired_coverage,
        ),
    )
    add_check(
        result,
        'semantic costmap',
        runtime['costmaps']['valid_count'] >= 1,
        str(runtime['costmaps']),
    )
    add_check(
        result,
        'recorded text query',
        runtime['message_counts'].get('/text_query', 0) >= 1,
        'count={}'.format(
            runtime['message_counts'].get('/text_query', 0)),
    )

    logs_path = Path(logs_directory).expanduser().resolve()
    if not logs_path.is_dir():
        raise ValidationError(
            'logs directory does not exist: {}'.format(logs_path))
    log_text = ''
    for log_path in sorted(logs_path.glob('*.log')):
        try:
            log_text += '\n' + log_path.read_text(
                encoding='utf-8', errors='replace')
        except OSError as error:
            raise ValidationError(
                'cannot read {}: {}'.format(log_path, error)) from error

    fatal_patterns = (
        r'Traceback \(most recent call last\)',
        r'\bFATAL\b',
        r'cannot deserialize',
        r'IMU and LiDAR not Synced',
        r'lidar loop back',
    )
    fatal_hits = [
        pattern for pattern in fatal_patterns
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]
    add_check(
        result,
        'runtime logs have no fatal patterns',
        not fatal_hits,
        'hits={}'.format(fatal_hits),
    )
    add_check(
        result,
        'FAST-LIO initialized IMU',
        'IMU Initial Done' in log_text,
        'expected log marker "IMU Initial Done"',
    )
    projected = [
        int(match.group(1))
        for match in re.finditer(
            r'投影并融合\s+(\d+)\s*/\s*(\d+)',
            log_text,
        )
    ]
    add_check(
        result,
        'GA-BSVM projected points',
        any(count > 0 for count in projected),
        'projected_counts={}'.format(projected[-10:]),
    )
    no_points_count = len(re.findall(
        r'No Effective Points', log_text, flags=re.IGNORECASE))
    if no_points_count:
        result['warnings'].append(
            'FAST-LIO logged No Effective Points {} time(s)'.format(
                no_points_count))

    goal_count = runtime['message_counts'].get('/goal_pose', 0)
    target_count = runtime['message_counts'].get(
        '/query_target_pose', 0)
    if goal_count == 0:
        result['warnings'].append(
            '/goal_pose has no messages; this is scene-dependent')
    if target_count == 0:
        result['warnings'].append(
            '/query_target_pose has no messages; this is scene-dependent')

    result['passed'] = all(
        check['passed'] for check in result['checks'])
    return result


def render_report(result):
    """Render concise human-readable checks and warnings."""
    lines = []
    prepared = result.get('prepared')
    if prepared is not None:
        lines.append('=== Prepared input and merge ===')
        for check in prepared['checks']:
            lines.append('{} {}: {}'.format(
                'PASS' if check['passed'] else 'FAIL',
                check['name'],
                check['detail'],
            ))
        if prepared['common_receipt_overlap_s'] is not None:
            lines.append('common_receipt_overlap={:.3f}s'.format(
                prepared['common_receipt_overlap_s']))

    runtime = result.get('runtime')
    if runtime is not None:
        lines.append('=== Runtime output ===')
        for check in runtime['checks']:
            lines.append('{} {}: {}'.format(
                'PASS' if check['passed'] else 'FAIL',
                check['name'],
                check['detail'],
            ))
        odometry = runtime.get('odometry', {})
        lines.append(
            'odometry: duration={} rate={} final_translation={} '
            'max_radius={} final_angle={} max_angle={}'.format(
                odometry.get('duration_s'),
                odometry.get('rate_hz'),
                odometry.get('final_translation_m'),
                odometry.get('maximum_radius_m'),
                odometry.get('final_angle_deg'),
                odometry.get('maximum_angle_deg'),
            )
        )

    for warning in result.get('warnings', []):
        lines.append('WARN: {}'.format(warning))
    lines.append('OVERALL={}'.format(result['overall']))
    return '\n'.join(lines) + '\n'


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=(
            'Validate a Lite3 split input, merged replay bag, and optional '
            'offline FAST-LIO/CLIP/GA-BSVM result bag.'
        )
    )
    parser.add_argument('--input-run', required=True, type=Path)
    parser.add_argument('--merged-bag', required=True, type=Path)
    parser.add_argument('--output-bag', type=Path)
    parser.add_argument('--logs-dir', type=Path)
    parser.add_argument('--runtime-graph', type=Path)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--text-out', type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    """Run command-line validation."""
    arguments = parse_arguments(argv)
    result = {
        'overall': FAIL_STATUS,
        'passed': False,
        'warnings': [],
    }
    try:
        prepared = validate_prepared(
            arguments.input_run,
            arguments.merged_bag,
        )
        result['prepared'] = prepared
        result['warnings'].extend(prepared['warnings'])

        runtime_requested = arguments.output_bag is not None
        if runtime_requested:
            if arguments.logs_dir is None or arguments.runtime_graph is None:
                raise ValidationError(
                    '--logs-dir and --runtime-graph are required with '
                    '--output-bag')
            runtime = validate_runtime(
                arguments.output_bag,
                arguments.logs_dir,
                arguments.runtime_graph,
            )
            result['runtime'] = runtime
            result['warnings'].extend(runtime['warnings'])
            result['passed'] = prepared['passed'] and runtime['passed']
            if result['passed']:
                result['overall'] = PASS_STATUS
        else:
            result['passed'] = prepared['passed']
            if result['passed']:
                result['overall'] = PREPARED_STATUS
    except ValidationError as error:
        result['error'] = str(error)
        result['warnings'].append(str(error))

    report = render_report(result)
    sys.stdout.write(report)

    if arguments.text_out is not None:
        arguments.text_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.text_out.write_text(report, encoding='utf-8')
    if arguments.json_out is not None:
        arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding='utf-8',
        )
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
