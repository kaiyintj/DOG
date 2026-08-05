#!/usr/bin/env python3
"""Validate a split Lite3 ROS 2 bag capture using only Python stdlib."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


TOPIC_SPECS = {
    '/timefix/imu': {
        'bag': 'imu',
        'type': 'sensor_msgs/msg/Imu',
        'min_rate_hz': 190.0,
        'max_rate_hz': 210.0,
        'warning_gap_s': 0.1,
        'failure_gap_s': 1.0,
    },
    '/timefix/lidar': {
        'bag': 'lidar',
        'type': 'livox_ros_driver2/msg/CustomMsg',
        'min_rate_hz': 9.0,
        'max_rate_hz': 11.0,
        'warning_gap_s': 0.5,
        'failure_gap_s': 1.0,
    },
    '/camera/color/image_raw': {
        'bag': 'camera',
        'type': 'sensor_msgs/msg/Image',
        'min_rate_hz': 8.0,
        'max_rate_hz': 16.0,
        'warning_gap_s': 0.5,
        'failure_gap_s': 1.0,
    },
    '/camera/color/camera_info': {
        'bag': 'camera',
        'type': 'sensor_msgs/msg/CameraInfo',
        'min_rate_hz': 8.0,
        'max_rate_hz': 16.0,
        'warning_gap_s': 0.5,
        'failure_gap_s': 1.0,
    },
}

PRIMARY_TOPICS = (
    '/timefix/imu',
    '/timefix/lidar',
    '/camera/color/image_raw',
    '/camera/color/camera_info',
)


class BagReadError(RuntimeError):
    """Report a malformed, unreadable, or corrupt rosbag database."""


def _empty_topic_stats():
    return {
        'count': 0,
        'first_ns': None,
        'last_ns': None,
        'types': set(),
        'timestamps_ns': [],
    }


def read_bag_directory(bag_directory):
    """Aggregate topic statistics across every db3 file in one bag."""
    bag_directory = Path(bag_directory)
    database_paths = sorted(bag_directory.glob('*.db3'))
    if not database_paths:
        raise BagReadError(
            'no db3 files found in {}'.format(bag_directory))

    topics = {}
    for database_path in database_paths:
        uri = 'file:{}?mode=ro'.format(database_path)
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                quick_check = connection.execute(
                    'PRAGMA quick_check').fetchall()
                if quick_check != [('ok',)]:
                    raise BagReadError(
                        '{} failed PRAGMA quick_check: {}'.format(
                            database_path, quick_check))

                topic_rows = connection.execute("""
                    SELECT id, name, type
                    FROM topics
                """).fetchall()

                for topic_id, name, message_type in topic_rows:
                    item = topics.setdefault(name, _empty_topic_stats())
                    item['types'].add(message_type)

                    timestamp_rows = connection.execute("""
                        SELECT timestamp
                        FROM messages
                        WHERE topic_id = ?
                        ORDER BY timestamp
                    """, (topic_id,)).fetchall()
                    item['timestamps_ns'].extend(
                        row[0] for row in timestamp_rows)
        except sqlite3.DatabaseError as error:
            raise BagReadError(
                'cannot read {}: {}'.format(database_path, error)) from error

    for item in topics.values():
        timestamps = sorted(item.pop('timestamps_ns'))
        item['count'] = len(timestamps)
        if timestamps:
            item['first_ns'] = timestamps[0]
            item['last_ns'] = timestamps[-1]
        item['max_gap_s'] = _maximum_gap_seconds(timestamps)
        item['types'] = sorted(item['types'])

    return {
        'database_count': len(database_paths),
        'topics': topics,
    }


def _maximum_gap_seconds(timestamps_ns):
    if len(timestamps_ns) < 2:
        return None
    return max(
        current - previous
        for previous, current in zip(
            timestamps_ns[:-1], timestamps_ns[1:])
    ) * 1e-9


def _topic_rate_hz(item):
    if (
        item['count'] < 2
        or item['first_ns'] is None
        or item['last_ns'] <= item['first_ns']
    ):
        return None
    duration_s = (item['last_ns'] - item['first_ns']) * 1e-9
    return (item['count'] - 1) / duration_s


def _serializable_topic(item):
    result = dict(item)
    if item['first_ns'] is not None and item['last_ns'] is not None:
        result['duration_s'] = (
            item['last_ns'] - item['first_ns']) * 1e-9
    else:
        result['duration_s'] = None
    result['rate_hz'] = _topic_rate_hz(item)
    return result


def evaluate_capture(
        run_directory,
        recorder_statuses=None,
        minimum_overlap_s=60.0,
        maximum_camera_count_difference=0.02):
    """Evaluate required topics, rates, overlap, and recorder statuses."""
    run_directory = Path(run_directory).expanduser().resolve()
    bags = {}
    errors = []

    for bag_name in sorted({
            specification['bag']
            for specification in TOPIC_SPECS.values()}):
        try:
            bags[bag_name] = read_bag_directory(
                run_directory / bag_name)
        except BagReadError as error:
            errors.append(str(error))

    result = {
        'run_directory': str(run_directory),
        'overall': False,
        'validation_errors': errors,
        'checks': [],
        'topics': {},
        'common_overlap_s': None,
        'camera_count_difference_ratio': None,
        'warnings': [],
    }

    if errors:
        return result

    for topic_name, specification in TOPIC_SPECS.items():
        topic_item = bags[specification['bag']]['topics'].get(
            topic_name, _empty_topic_stats())
        if 'max_gap_s' not in topic_item:
            topic_item['max_gap_s'] = None
        if isinstance(topic_item['types'], set):
            topic_item['types'] = sorted(topic_item['types'])

        serializable = _serializable_topic(topic_item)
        result['topics'][topic_name] = serializable

        expected_type = specification['type']
        type_passed = serializable['types'] == [expected_type]
        result['checks'].append({
            'name': '{} type'.format(topic_name),
            'passed': type_passed,
            'detail': 'expected={}, actual={}'.format(
                expected_type, serializable['types']),
        })

        rate_hz = serializable['rate_hz']
        rate_passed = (
            rate_hz is not None
            and specification['min_rate_hz'] <= rate_hz
            <= specification['max_rate_hz']
        )
        result['checks'].append({
            'name': '{} rate'.format(topic_name),
            'passed': rate_passed,
            'detail': 'expected={:.3f}..{:.3f}Hz, actual={}'.format(
                specification['min_rate_hz'],
                specification['max_rate_hz'],
                (
                    '{:.3f}Hz'.format(rate_hz)
                    if rate_hz is not None
                    else 'missing'
                ),
            ),
        })

        max_gap_s = serializable['max_gap_s']
        gap_passed = (
            max_gap_s is not None
            and max_gap_s <= specification['failure_gap_s']
        )
        result['checks'].append({
            'name': '{} maximum bag receipt gap'.format(topic_name),
            'passed': gap_passed,
            'detail': 'expected<={:.3f}s, actual={}'.format(
                specification['failure_gap_s'],
                (
                    '{:.3f}s'.format(max_gap_s)
                    if max_gap_s is not None
                    else 'missing'
                ),
            ),
        })
        if (
            max_gap_s is not None
            and specification['warning_gap_s'] < max_gap_s
            <= specification['failure_gap_s']
        ):
            result['warnings'].append(
                '{} bag receipt gap {:.3f}s exceeds the normal {:.3f}s; '
                'audit header.stamp before motion tests'.format(
                    topic_name,
                    max_gap_s,
                    specification['warning_gap_s'],
                ))

    image_count = result['topics'][
        '/camera/color/image_raw']['count']
    camera_info_count = result['topics'][
        '/camera/color/camera_info']['count']
    maximum_camera_count = max(image_count, camera_info_count)
    if maximum_camera_count:
        difference_ratio = (
            abs(image_count - camera_info_count)
            / maximum_camera_count
        )
    else:
        difference_ratio = 1.0
    result['camera_count_difference_ratio'] = difference_ratio
    result['checks'].append({
        'name': 'camera image/info count difference',
        'passed': difference_ratio <= maximum_camera_count_difference,
        'detail': 'expected<= {:.2%}, actual={:.2%}'.format(
            maximum_camera_count_difference, difference_ratio),
    })

    primary_items = [
        result['topics'][topic_name]
        for topic_name in PRIMARY_TOPICS
    ]
    if all(
            item['first_ns'] is not None and item['last_ns'] is not None
            for item in primary_items):
        common_start_ns = max(
            item['first_ns'] for item in primary_items)
        common_end_ns = min(
            item['last_ns'] for item in primary_items)
        overlap_s = max(
            0.0, (common_end_ns - common_start_ns) * 1e-9)
    else:
        overlap_s = 0.0
    result['common_overlap_s'] = overlap_s
    result['checks'].append({
        'name': 'four-topic common overlap',
        'passed': overlap_s >= minimum_overlap_s,
        'detail': 'expected>={:.3f}s, actual={:.3f}s'.format(
            minimum_overlap_s, overlap_s),
    })

    if recorder_statuses is not None:
        for recorder_name in ('imu', 'lidar', 'camera'):
            status = recorder_statuses.get(recorder_name)
            result['checks'].append({
                'name': '{} recorder status'.format(recorder_name),
                'passed': status == 124,
                'detail': 'expected=124, actual={}'.format(status),
            })

    lidar_topics = bags['lidar']['topics']
    tf_static_count = lidar_topics.get(
        '/tf_static', {'count': 0})['count']
    tf_count = lidar_topics.get('/tf', {'count': 0})['count']
    if tf_static_count < 1:
        result['warnings'].append(
            '/tf_static has no messages; verify the sensor TF separately')
    if tf_count < 1:
        result['warnings'].append(
            '/tf has no messages; this does not fail sensor capture')

    result['overall'] = all(
        check['passed'] for check in result['checks'])
    return result


def format_report(result):
    """Render a concise human-readable validation report."""
    lines = []
    for topic_name in PRIMARY_TOPICS:
        topic = result['topics'].get(topic_name)
        if topic is None:
            lines.append('FAIL {}: missing'.format(topic_name))
            continue
        rate = topic['rate_hz']
        lines.append(
            '{}: count={} duration={} rate={} max_gap={}'.format(
                topic_name,
                topic['count'],
                (
                    '{:.3f}s'.format(topic['duration_s'])
                    if topic['duration_s'] is not None
                    else 'n/a'
                ),
                (
                    '{:.3f}Hz'.format(rate)
                    if rate is not None
                    else 'n/a'
                ),
                (
                    '{:.3f}s'.format(topic['max_gap_s'])
                    if topic['max_gap_s'] is not None
                    else 'n/a'
                ),
            ))

    if result['camera_count_difference_ratio'] is not None:
        lines.append(
            'camera_count_difference={:.2%}'.format(
                result['camera_count_difference_ratio']))
    if result['common_overlap_s'] is not None:
        lines.append(
            'common_overlap={:.3f}s'.format(
                result['common_overlap_s']))

    for check in result['checks']:
        lines.append('{} {}: {}'.format(
            'PASS' if check['passed'] else 'FAIL',
            check['name'],
            check['detail'],
        ))
    for warning in result['warnings']:
        lines.append('WARN: {}'.format(warning))
    for error in result['validation_errors']:
        lines.append('ERROR: {}'.format(error))

    lines.append(
        'OVERALL={}'.format('PASS' if result['overall'] else 'FAIL'))
    return '\n'.join(lines)


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description='Validate split Lite3 ROS 2 bag directories.')
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--imu-status', type=int)
    parser.add_argument('--lidar-status', type=int)
    parser.add_argument('--camera-status', type=int)
    parser.add_argument('--minimum-overlap', type=float, default=60.0)
    parser.add_argument(
        '--maximum-camera-count-difference',
        type=float,
        default=0.02,
    )
    parser.add_argument('--json-out', type=Path)
    return parser.parse_args()


def main():
    """Run the command-line validator and return a process status."""
    args = _parse_arguments()
    status_values = (
        args.imu_status,
        args.lidar_status,
        args.camera_status,
    )
    if any(value is not None for value in status_values):
        if not all(value is not None for value in status_values):
            print(
                'ERROR: provide all three recorder statuses or none',
                file=sys.stderr,
            )
            return 2
        recorder_statuses = {
            'imu': args.imu_status,
            'lidar': args.lidar_status,
            'camera': args.camera_status,
        }
    else:
        recorder_statuses = None

    result = evaluate_capture(
        args.run_directory,
        recorder_statuses=recorder_statuses,
        minimum_overlap_s=args.minimum_overlap,
        maximum_camera_count_difference=(
            args.maximum_camera_count_difference),
    )
    print(format_report(result))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    if result['validation_errors']:
        return 2
    return 0 if result['overall'] else 1


if __name__ == '__main__':
    sys.exit(main())
