#!/usr/bin/env python3
"""Validate a split Lite3 controlled-motion sensor capture."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


BASE_VALIDATOR_PATH = (
    Path(__file__).resolve().parent / 'verify_lite3_capture.py'
)
BASE_SPEC = importlib.util.spec_from_file_location(
    'verify_lite3_capture', BASE_VALIDATOR_PATH)
BASE_VALIDATOR = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE_VALIDATOR)


STATE_TOPIC_TYPES = {
    '/leg_odom2': 'nav_msgs/msg/Odometry',
    '/diagnostics/lite3/leg_odom_fixed': 'nav_msgs/msg/Odometry',
    '/diagnostics/lite3/motion_test_marker': 'std_msgs/msg/String',
    '/imu/data': 'sensor_msgs/msg/Imu',
    '/joint_states': 'sensor_msgs/msg/JointState',
    '/handle_state': 'geometry_msgs/msg/Twist',
    '/tf': 'tf2_msgs/msg/TFMessage',
}

STREAMING_TOPICS = (
    '/timefix/imu',
    '/timefix/lidar',
    '/camera/color/image_raw',
    '/camera/color/camera_info',
    '/leg_odom2',
    '/diagnostics/lite3/leg_odom_fixed',
    '/imu/data',
    '/joint_states',
    '/handle_state',
)

COMMAND_TOPICS = ('/cmd_vel', '/cmd_vel_corrected')


def _check(name, passed, detail):
    return {
        'name': name,
        'passed': bool(passed),
        'detail': detail,
    }


def _serializable_state_topics(state_bag):
    topics = {}
    for name, item in state_bag['topics'].items():
        topics[name] = BASE_VALIDATOR._serializable_topic(item)
    return topics


def _common_overlap_seconds(sensor_topics, state_topics):
    items = []
    for topic_name in STREAMING_TOPICS:
        source = (
            sensor_topics
            if topic_name in sensor_topics
            else state_topics
        )
        item = source.get(topic_name)
        if item is None or item['first_ns'] is None or item['last_ns'] is None:
            return 0.0
        items.append(item)
    common_start_ns = max(item['first_ns'] for item in items)
    common_end_ns = min(item['last_ns'] for item in items)
    return max(0.0, (common_end_ns - common_start_ns) * 1e-9)


def evaluate_controlled_capture(
        run_directory,
        minimum_overlap_s=60.0,
        minimum_markers=16,
        minimum_fixed_raw_ratio=0.95,
        maximum_fixed_raw_ratio=1.05):
    """Evaluate sensor, state, marker, and command-safety evidence."""
    run_directory = Path(run_directory).expanduser().resolve()
    sensor_result = BASE_VALIDATOR.evaluate_capture(
        run_directory,
        minimum_overlap_s=minimum_overlap_s,
    )
    result = {
        'run_directory': str(run_directory),
        'overall': False,
        'sensor_capture': sensor_result,
        'state_topics': {},
        'common_overlap_s': 0.0,
        'fixed_to_raw_ratio': None,
        'marker_count': 0,
        'command_message_count': 0,
        'checks': [],
        'validation_errors': [],
    }
    try:
        state_bag = BASE_VALIDATOR.read_bag_directory(
            run_directory / 'state')
    except BASE_VALIDATOR.BagReadError as error:
        result['validation_errors'].append(str(error))
        return result

    state_topics = _serializable_state_topics(state_bag)
    result['state_topics'] = state_topics
    result['checks'].append(_check(
        'base sensor capture',
        sensor_result['overall'],
        'expected=PASS, actual={}'.format(
            'PASS' if sensor_result['overall'] else 'FAIL'),
    ))

    for topic_name, expected_type in STATE_TOPIC_TYPES.items():
        item = state_topics.get(topic_name)
        actual_types = item['types'] if item is not None else []
        count = item['count'] if item is not None else 0
        result['checks'].append(_check(
            '{} type and data'.format(topic_name),
            actual_types == [expected_type] and count > 0,
            'expected={} with data, actual={} count={}'.format(
                expected_type, actual_types, count),
        ))

    raw_count = state_topics.get('/leg_odom2', {'count': 0})['count']
    fixed_count = state_topics.get(
        '/diagnostics/lite3/leg_odom_fixed', {'count': 0})['count']
    if raw_count:
        fixed_to_raw_ratio = fixed_count / raw_count
    else:
        fixed_to_raw_ratio = None
    result['fixed_to_raw_ratio'] = fixed_to_raw_ratio
    ratio_passed = (
        fixed_to_raw_ratio is not None
        and minimum_fixed_raw_ratio <= fixed_to_raw_ratio
        <= maximum_fixed_raw_ratio
    )
    result['checks'].append(_check(
        'fixed/raw odometry count ratio',
        ratio_passed,
        'expected={:.3f}..{:.3f}, actual={}'.format(
            minimum_fixed_raw_ratio,
            maximum_fixed_raw_ratio,
            (
                '{:.6f}'.format(fixed_to_raw_ratio)
                if fixed_to_raw_ratio is not None
                else 'missing'
            ),
        ),
    ))

    marker_count = state_topics.get(
        '/diagnostics/lite3/motion_test_marker', {'count': 0})['count']
    result['marker_count'] = marker_count
    result['checks'].append(_check(
        'complete motion markers',
        marker_count >= minimum_markers,
        'expected>={}, actual={}'.format(minimum_markers, marker_count),
    ))

    command_count = sum(
        state_topics.get(topic_name, {'count': 0})['count']
        for topic_name in COMMAND_TOPICS
    )
    result['command_message_count'] = command_count
    result['checks'].append(_check(
        'zero command messages',
        command_count == 0,
        'expected=0, actual={}'.format(command_count),
    ))

    overlap_s = _common_overlap_seconds(
        sensor_result['topics'], state_topics)
    result['common_overlap_s'] = overlap_s
    result['checks'].append(_check(
        'sensor/state common overlap',
        overlap_s >= minimum_overlap_s,
        'expected>={:.3f}s, actual={:.3f}s'.format(
            minimum_overlap_s, overlap_s),
    ))

    result['overall'] = (
        not result['validation_errors']
        and all(check['passed'] for check in result['checks'])
    )
    return result


def format_report(result):
    """Render a concise controlled-motion capture report."""
    lines = [BASE_VALIDATOR.format_report(result['sensor_capture'])]
    lines.append('STATE_TOPICS')
    for topic_name in sorted(result['state_topics']):
        item = result['state_topics'][topic_name]
        lines.append('{}: count={} rate={}'.format(
            topic_name,
            item['count'],
            (
                '{:.3f}Hz'.format(item['rate_hz'])
                if item['rate_hz'] is not None
                else 'n/a'
            ),
        ))
    for check in result['checks']:
        lines.append('{} {}: {}'.format(
            'PASS' if check['passed'] else 'FAIL',
            check['name'],
            check['detail'],
        ))
    for error in result['validation_errors']:
        lines.append('ERROR: {}'.format(error))
    lines.append('CONTROLLED_MOTION_CAPTURE={}'.format(
        'PASS' if result['overall'] else 'FAIL'))
    lines.append('MOTION_READY=NO')
    return '\n'.join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory')
    parser.add_argument('--json-out')
    return parser.parse_args()


def main():
    """Validate one capture and return a shell-friendly status."""
    args = parse_args()
    result = evaluate_controlled_capture(args.run_directory)
    report = format_report(result)
    print(report)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding='utf-8',
        )
    return 0 if result['overall'] else 1


if __name__ == '__main__':
    sys.exit(main())
