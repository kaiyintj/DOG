"""Tests for the standalone Lite3 header-timestamp audit."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'audit_lite3_timestamps.py'
)
SPEC = importlib.util.spec_from_file_location(
    'audit_lite3_timestamps', SCRIPT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _header(timestamp_ns, frame_id):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=timestamp_ns // 1000000000,
            nanosec=timestamp_ns % 1000000000,
        ),
        frame_id=frame_id,
    )


def _camera_info(
        k=None,
        frame_id='camera_color_optical_frame',
        width=640,
        height=480):
    if k is None:
        k = (
            600.0, 0.0, 320.0,
            0.0, 601.0, 240.0,
            0.0, 0.0, 1.0,
        )
    return SimpleNamespace(
        header=_header(1000000000, frame_id),
        width=width,
        height=height,
        distortion_model='plumb_bob',
        d=(0.1, -0.2, 0.0, 0.0, 0.0),
        k=k,
        r=(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ),
        p=(
            600.0, 0.0, 320.0, 0.0,
            0.0, 601.0, 240.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ),
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(
            x_offset=0,
            y_offset=0,
            height=0,
            width=0,
            do_rectify=False,
        ),
    )


def _transform(parent, child, x=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=x, y=0.0, z=0.0),
            rotation=SimpleNamespace(
                x=0.0,
                y=0.0,
                z=0.0,
                w=1.0,
            ),
        ),
    )


def _install_synthetic_audit(monkeypatch, excessive_gap_topic=None):
    base_ns = 1000000000
    duration_ns = 60000000000
    steps_ns = {
        '/timefix/imu': 40000000,
        '/timefix/lidar': 100000000,
        '/camera/color/image_raw': 200000000,
        '/camera/color/camera_info': 200000000,
    }
    timestamps_by_topic = {
        topic_name: list(range(
            base_ns,
            base_ns + duration_ns + 1,
            step_ns,
        ))
        for topic_name, step_ns in steps_ns.items()
    }
    if excessive_gap_topic is not None:
        timestamps = timestamps_by_topic[excessive_gap_topic]
        middle_index = len(timestamps) // 2
        removed_count = (
            2 if excessive_gap_topic == '/timefix/lidar' else 1
        )
        del timestamps[middle_index:middle_index + removed_count]

    frames_by_topic = {
        '/timefix/imu': 'imu_frame',
        '/timefix/lidar': 'lidar_frame',
        '/camera/color/image_raw': 'camera_color_optical_frame',
        '/camera/color/camera_info': 'camera_color_optical_frame',
    }
    camera_info = _camera_info()
    camera_info.d = (0.0, 0.0, 0.0, 0.0, 0.0)
    camera_info_audit = AUDIT.CameraInfoAudit()
    camera_info_audit.add(camera_info)

    def fake_audit_topic(
            unused_run_dir,
            unused_bag_name,
            topic_name,
            unused_expected_type):
        timestamps = timestamps_by_topic[topic_name]
        receipt = AUDIT.GapStatistics()
        header = AUDIT.GapStatistics()
        for timestamp in timestamps:
            receipt.add(timestamp)
            header.add(timestamp)
        timebase = AUDIT.DifferenceStatistics()
        if topic_name == '/timefix/lidar':
            for timestamp in timestamps:
                timebase.add(timestamp, timestamp)
        return {
            'receipt': receipt,
            'header': header,
            'header_timestamps': timestamps,
            'frame_ids': {
                frames_by_topic[topic_name]: len(timestamps),
            },
            'image_shapes': (
                {(640, 480): len(timestamps)}
                if topic_name == '/camera/color/image_raw'
                else {}
            ),
            'zero_header_count': 0,
            'timebase_differences': timebase,
            'camera_info_audit': (
                camera_info_audit
                if topic_name == '/camera/color/camera_info'
                else None
            ),
        }

    tf_audit = AUDIT.TfStaticAudit()
    tf_audit.add(SimpleNamespace(transforms=[
        _transform('base_link', 'lidar_frame'),
        _transform('base_link', 'camera_color_optical_frame'),
    ]))
    monkeypatch.setattr(AUDIT, '_audit_topic', fake_audit_topic)
    monkeypatch.setattr(
        AUDIT, '_audit_tf_static', lambda unused_run_dir: tf_audit)


def test_common_window_alignment_excludes_stream_edges():
    millisecond = 1000000
    result = AUDIT.common_window_timestamp_alignment(
        [
            0,
            100 * millisecond,
            200 * millisecond,
            300 * millisecond,
            400 * millisecond,
        ],
        [
            90 * millisecond,
            190 * millisecond,
            290 * millisecond,
        ],
    )

    assert result['common_start_ns'] == 90 * millisecond
    assert result['common_end_ns'] == 290 * millisecond
    assert result['reference_count'] == 2
    assert result['reference_total'] == 5
    assert result['difference_statistics'].count == 2
    assert result['absolute_percentiles_ns']['p50'] == 10 * millisecond
    assert result['coverage'][10 * millisecond]['count'] == 2
    assert result['coverage'][10 * millisecond]['ratio'] == 1.0


def test_common_window_alignment_reports_no_overlap():
    result = AUDIT.common_window_timestamp_alignment(
        [0, 10],
        [20, 30],
    )

    assert result['common_end_ns'] < result['common_start_ns']
    assert result['difference_statistics'].count == 0
    assert result['absolute_percentiles_ns']['p95'] is None


def test_image_camera_info_matching_is_one_to_one_and_inclusive():
    millisecond = 1000000
    result = AUDIT.match_timestamp_streams_within_tolerance(
        [
            0,
            10 * millisecond,
            20 * millisecond,
            30 * millisecond,
        ],
        [
            millisecond,
            11 * millisecond,
            22 * millisecond,
            31 * millisecond,
        ],
        millisecond,
    )

    assert result['matched_count'] == 3
    assert result['unmatched_reference_count'] == 1
    assert result['unmatched_comparison_count'] == 1
    assert result['reference_coverage'] == 0.75
    assert result['difference_statistics'].maximum_absolute_ns == millisecond


def test_lidar_imu_window_counts_samples_on_both_sides():
    result = AUDIT.samples_around_references(
        [0, 1000000000],
        [
            -100000000,
            0,
            100000000,
            900000000,
            1000000000,
            1100000000,
        ],
        150000000,
    )

    assert result['with_samples_count'] == 2
    assert result['without_samples_count'] == 0
    assert result['with_samples_on_both_sides_count'] == 2
    assert result['total']['minimum'] == 3
    assert result['total']['p50'] == 3
    assert result['total']['maximum'] == 3


def test_camera_info_audit_selects_modal_valid_candidate():
    audit = AUDIT.CameraInfoAudit()
    valid = _camera_info()
    audit.add(valid)
    audit.add(valid)
    audit.add(_camera_info(k=(0.0,) * 9))

    assert audit.count == 3
    assert audit.valid_count == 2
    assert audit.invalid_count == 1
    assert len(audit.fingerprints) == 2
    assert audit.valid_fingerprint_count == 1
    assert audit.candidate['k'] == valid.k
    assert audit.candidate['valid_count'] == 2
    assert audit.reason_counts['K has non-positive focal length'] == 1


def test_camera_info_fingerprint_changes_with_calibration():
    first = AUDIT._camera_info_snapshot(_camera_info())
    second = AUDIT._camera_info_snapshot(_camera_info(k=(
        610.0, 0.0, 320.0,
        0.0, 611.0, 240.0,
        0.0, 0.0, 1.0,
    )))

    assert first['valid']
    assert second['valid']
    assert first['fingerprint'] != second['fingerprint']


def test_tf_static_audit_finds_lidar_camera_chain():
    audit = AUDIT.TfStaticAudit()
    audit.add(SimpleNamespace(transforms=[
        _transform('base_link', 'lidar_frame', x=0.2),
        _transform('base_link', 'camera_color_optical_frame', x=0.1),
    ]))

    assert audit.message_count == 1
    assert audit.valid_transform_count == 2
    assert audit.path(
        'lidar_frame',
        'camera_color_optical_frame',
    ) == [
        'lidar_frame',
        'base_link',
        'camera_color_optical_frame',
    ]
    assert not audit.readiness_warnings(
        'lidar_frame',
        'camera_color_optical_frame',
    )


def test_tf_static_conflict_is_not_ready():
    audit = AUDIT.TfStaticAudit()
    audit.add(SimpleNamespace(transforms=[
        _transform('base_link', 'lidar_frame', x=0.1),
        _transform('base_link', 'lidar_frame', x=0.2),
    ]))

    warnings = audit.readiness_warnings(
        'lidar_frame',
        'camera_color_optical_frame',
    )

    assert audit.conflicting_edges == [('base_link', 'lidar_frame')]
    assert any('changes value' in warning for warning in warnings)
    assert any('no chain' in warning for warning in warnings)


def test_tf_static_cycle_is_not_ready():
    audit = AUDIT.TfStaticAudit()
    audit.add(SimpleNamespace(transforms=[
        _transform('base_link', 'lidar_frame'),
        _transform('lidar_frame', 'camera_frame'),
        _transform('camera_frame', 'base_link'),
    ]))

    assert audit.directed_cycles
    assert any(
        'directed cycle' in warning
        for warning in audit.readiness_warnings(
            'lidar_frame', 'camera_frame')
    )


def test_camera_info_rejects_non_orthonormal_rotation():
    message = _camera_info()
    message.r = (
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    )

    snapshot = AUDIT._camera_info_snapshot(message)

    assert not snapshot['valid']
    assert 'R is not a proper orthonormal rotation' in snapshot['reasons']


def test_readiness_printers_emit_candidate_and_not_ready(capsys):
    camera_audit = AUDIT.CameraInfoAudit()
    camera_audit.add(_camera_info())
    AUDIT._print_camera_info_audit(
        camera_audit,
        'different_image_frame',
    )

    tf_audit = AUDIT.TfStaticAudit()
    tf_audit.read_error = 'topic missing'
    AUDIT._print_tf_static_audit(
        tf_audit,
        'lidar_frame',
        'camera_frame',
    )
    output = capsys.readouterr().out

    assert 'candidate_camera_k: [600' in output
    assert 'NOT_READY: CameraInfo frame' in output
    assert 'NOT_READY: cannot audit /tf_static: topic missing' in output


def test_audit_run_keeps_legacy_sections_before_extended_output(
        monkeypatch, capsys, tmp_path):
    camera_info_audit = AUDIT.CameraInfoAudit()
    camera_info_audit.add(_camera_info())
    timestamps_by_topic = {
        '/timefix/imu': [
            900000000,
            1000000000,
            1100000000,
            1200000000,
        ],
        '/timefix/lidar': [1000000000, 1100000000],
        '/camera/color/image_raw': [1000000000, 1100000000],
        '/camera/color/camera_info': [1000000000, 1100000000],
    }
    frames_by_topic = {
        '/timefix/imu': 'imu_frame',
        '/timefix/lidar': 'lidar_frame',
        '/camera/color/image_raw': 'camera_color_optical_frame',
        '/camera/color/camera_info': 'camera_color_optical_frame',
    }

    def fake_audit_topic(
            unused_run_dir,
            unused_bag_name,
            topic_name,
            unused_expected_type):
        timestamps = timestamps_by_topic[topic_name]
        receipt = AUDIT.GapStatistics()
        header = AUDIT.GapStatistics()
        for timestamp in timestamps:
            receipt.add(timestamp)
            header.add(timestamp)
        timebase = AUDIT.DifferenceStatistics()
        if topic_name == '/timefix/lidar':
            for timestamp in timestamps:
                timebase.add(timestamp, timestamp)
        return {
            'receipt': receipt,
            'header': header,
            'header_timestamps': timestamps,
            'frame_ids': {
                frames_by_topic[topic_name]: len(timestamps),
            },
            'zero_header_count': 0,
            'timebase_differences': timebase,
            'camera_info_audit': (
                camera_info_audit
                if topic_name == '/camera/color/camera_info'
                else None
            ),
        }

    tf_audit = AUDIT.TfStaticAudit()
    tf_audit.add(SimpleNamespace(transforms=[
        _transform('base_link', 'lidar_frame'),
        _transform('base_link', 'camera_color_optical_frame'),
    ]))
    monkeypatch.setattr(AUDIT, '_audit_topic', fake_audit_topic)
    monkeypatch.setattr(
        AUDIT, '_audit_tf_static', lambda unused_run_dir: tf_audit)

    AUDIT.audit_run(tmp_path)
    output = capsys.readouterr().out

    legacy_heading = '=== Nearest header timestamp differences ==='
    extended_heading = '=== Common-window LiDAR/image alignment ==='
    assert 'nearest_image_minus_lidar' in output
    assert 'nearest_imu_minus_lidar' in output
    assert output.index(legacy_heading) < output.index(extended_heading)
    assert 'image_camera_info_match_le_1ms' in output
    assert 'candidate_camera_k: [600' in output
    assert 'tf_static_lidar_to_image_path' in output


def test_sensor_header_alignment_passes_healthy_header_gap_gates(
        monkeypatch, capsys, tmp_path):
    _install_synthetic_audit(monkeypatch)

    result = AUDIT.audit_run(tmp_path)
    output = capsys.readouterr().out

    expected_lines = (
        'PASS topic=/timefix/imu max_gap=0.040000000s '
        'limit=0.050000000s',
        'PASS topic=/timefix/lidar max_gap=0.100000000s '
        'limit=0.200000000s',
        'PASS topic=/camera/color/image_raw max_gap=0.200000000s '
        'limit=0.250000000s',
        'PASS topic=/camera/color/camera_info max_gap=0.200000000s '
        'limit=0.250000000s',
    )
    assert result['sensor_header_alignment_passed']
    assert output.count('header_max_gap_gate: PASS') == 4
    assert all(line in output for line in expected_lines)
    assert 'SENSOR_HEADER_ALIGNMENT=PASS' in output


@pytest.mark.parametrize(
    ('topic_name', 'maximum_gap', 'limit'),
    (
        ('/timefix/imu', '0.080000000s', '0.050000000s'),
        ('/timefix/lidar', '0.300000000s', '0.200000000s'),
        (
            '/camera/color/image_raw',
            '0.400000000s',
            '0.250000000s',
        ),
        (
            '/camera/color/camera_info',
            '0.400000000s',
            '0.250000000s',
        ),
    ),
)
def test_sensor_header_alignment_fails_each_excessive_header_gap(
        monkeypatch,
        capsys,
        tmp_path,
        topic_name,
        maximum_gap,
        limit):
    _install_synthetic_audit(monkeypatch, topic_name)

    result = AUDIT.audit_run(tmp_path)
    output = capsys.readouterr().out

    assert not result['sensor_header_alignment_passed']
    assert (
        'header_max_gap_gate: FAIL topic={} max_gap={} limit={}'.format(
            topic_name,
            maximum_gap,
            limit,
        )
    ) in output
    assert (
        'NOT_READY: {} header max gap {} exceeds the {} limit'.format(
            topic_name,
            maximum_gap,
            limit,
        )
    ) in output
    assert output.count('header_max_gap_gate: PASS') == 3
    assert 'SENSOR_HEADER_ALIGNMENT=FAIL' in output
