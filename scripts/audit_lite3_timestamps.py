#!/usr/bin/env python3
"""Audit Lite3 rosbag receipt times and sensor header timestamps."""

import argparse
import bisect
import collections
import contextlib
import datetime
import hashlib
import heapq
import math
import pathlib
import sqlite3
import sys
import urllib.parse


TOPICS = (
    (
        'imu',
        '/timefix/imu',
        'sensor_msgs/msg/Imu',
    ),
    (
        'lidar',
        '/timefix/lidar',
        'livox_ros_driver2/msg/CustomMsg',
    ),
    (
        'camera',
        '/camera/color/image_raw',
        'sensor_msgs/msg/Image',
    ),
    (
        'camera',
        '/camera/color/camera_info',
        'sensor_msgs/msg/CameraInfo',
    ),
)

TF_STATIC_TOPIC = (
    'lidar',
    '/tf_static',
    'tf2_msgs/msg/TFMessage',
)
LIDAR_IMAGE_COVERAGE_THRESHOLDS_NS = (
    10000000,
    50000000,
    100000000,
    150000000,
    200000000,
)
IMAGE_INFO_MATCH_TOLERANCE_NS = 1000000
LIDAR_IMU_HALF_WINDOW_NS = 150000000
MIN_IMAGE_INFO_MATCH_COVERAGE = 0.99
MIN_LIDAR_IMAGE_COVERAGE = 0.98
MAX_LIDAR_IMAGE_P99_NS = 100000000
MAX_LIDAR_IMAGE_MATCH_NS = 200000000
MIN_LIDAR_IMU_COVERAGE = 0.99
HEADER_MAX_GAP_THRESHOLDS_NS = {
    '/timefix/imu': 50000000,
    '/timefix/lidar': 200000000,
    '/camera/color/image_raw': 250000000,
    '/camera/color/camera_info': 250000000,
}


class AuditError(RuntimeError):
    """Report an unreadable bag or an invalid timestamp stream."""


class GapStatistics:
    """Accumulate rate and maximum-gap statistics in constant memory."""

    def __init__(self):
        """Initialize empty timestamp statistics."""
        self.count = 0
        self.first_ns = None
        self.last_ns = None
        self.previous_ns = None
        self.maximum_gap_ns = None
        self.maximum_gap_start_ns = None
        self.maximum_gap_end_ns = None
        self.non_monotonic_count = 0

    def add(self, timestamp_ns):
        """Add one timestamp in message delivery order."""
        timestamp_ns = int(timestamp_ns)
        if self.first_ns is None:
            self.first_ns = timestamp_ns
        if self.previous_ns is not None:
            difference_ns = timestamp_ns - self.previous_ns
            if difference_ns < 0:
                self.non_monotonic_count += 1
            elif (
                self.maximum_gap_ns is None
                or difference_ns > self.maximum_gap_ns
            ):
                self.maximum_gap_ns = difference_ns
                self.maximum_gap_start_ns = self.previous_ns
                self.maximum_gap_end_ns = timestamp_ns
        self.previous_ns = timestamp_ns
        self.last_ns = timestamp_ns
        self.count += 1

    @property
    def duration_s(self):
        """Return elapsed time from the first to last timestamp."""
        if (
            self.first_ns is None
            or self.last_ns is None
            or self.last_ns <= self.first_ns
        ):
            return None
        return (self.last_ns - self.first_ns) * 1e-9

    @property
    def rate_hz(self):
        """Return the endpoint-based average rate."""
        duration_s = self.duration_s
        if duration_s is None or self.count < 2:
            return None
        return (self.count - 1) / duration_s

    @property
    def maximum_gap_s(self):
        """Return the largest consecutive non-negative gap."""
        if self.maximum_gap_ns is None:
            return None
        return self.maximum_gap_ns * 1e-9


class DifferenceStatistics:
    """Accumulate signed and absolute timestamp differences."""

    def __init__(self):
        """Initialize empty difference statistics."""
        self.count = 0
        self.minimum_signed_ns = None
        self.maximum_signed_ns = None
        self.minimum_absolute_ns = None
        self.maximum_absolute_ns = None
        self.sum_signed_ns = 0
        self.sum_absolute_ns = 0
        self.sum_squared_ns = 0
        self.maximum_reference_ns = None
        self.maximum_comparison_ns = None

    def add(self, reference_ns, comparison_ns):
        """Add comparison minus reference, both expressed in nanoseconds."""
        difference_ns = int(comparison_ns) - int(reference_ns)
        absolute_ns = abs(difference_ns)
        self.count += 1
        self.sum_signed_ns += difference_ns
        self.sum_absolute_ns += absolute_ns
        self.sum_squared_ns += difference_ns * difference_ns

        if (
            self.minimum_signed_ns is None
            or difference_ns < self.minimum_signed_ns
        ):
            self.minimum_signed_ns = difference_ns
        if (
            self.maximum_signed_ns is None
            or difference_ns > self.maximum_signed_ns
        ):
            self.maximum_signed_ns = difference_ns
        if (
            self.minimum_absolute_ns is None
            or absolute_ns < self.minimum_absolute_ns
        ):
            self.minimum_absolute_ns = absolute_ns
        if (
            self.maximum_absolute_ns is None
            or absolute_ns > self.maximum_absolute_ns
        ):
            self.maximum_absolute_ns = absolute_ns
            self.maximum_reference_ns = int(reference_ns)
            self.maximum_comparison_ns = int(comparison_ns)

    @property
    def mean_signed_s(self):
        """Return the mean signed difference in seconds."""
        if not self.count:
            return None
        return self.sum_signed_ns * 1e-9 / self.count

    @property
    def mean_absolute_s(self):
        """Return the mean absolute difference in seconds."""
        if not self.count:
            return None
        return self.sum_absolute_ns * 1e-9 / self.count

    @property
    def root_mean_square_s(self):
        """Return the root-mean-square signed difference in seconds."""
        if not self.count:
            return None
        return math.sqrt(self.sum_squared_ns / self.count) * 1e-9


def _nearest_timestamp_pairs(reference_timestamps, comparison_timestamps):
    """Yield each reference and its nearest ordered comparison timestamp."""
    comparison_iterator = iter(comparison_timestamps)
    previous_comparison = None
    current_comparison = next(comparison_iterator, None)

    for reference_ns in reference_timestamps:
        while (
            current_comparison is not None
            and current_comparison < reference_ns
        ):
            previous_comparison = current_comparison
            current_comparison = next(comparison_iterator, None)

        candidates = (
            candidate
            for candidate in (previous_comparison, current_comparison)
            if candidate is not None
        )
        nearest = min(
            candidates,
            key=lambda candidate: abs(candidate - reference_ns),
            default=None,
        )
        if nearest is not None:
            yield int(reference_ns), int(nearest)


def nearest_timestamp_differences(reference_timestamps, comparison_timestamps):
    """Match each ordered reference timestamp to its nearest comparison."""
    statistics = DifferenceStatistics()
    for reference_ns, comparison_ns in _nearest_timestamp_pairs(
        reference_timestamps,
        comparison_timestamps,
    ):
        statistics.add(reference_ns, comparison_ns)

    return statistics


def _percentile(sorted_values, percentile):
    """Return a linearly interpolated percentile from ordered values."""
    if not sorted_values:
        return None
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _count_distribution(values):
    """Summarize a non-negative integer population."""
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {
            'count': 0,
            'minimum': None,
            'p1': None,
            'p50': None,
            'p95': None,
            'maximum': None,
        }
    return {
        'count': len(ordered),
        'minimum': ordered[0],
        'p1': _percentile(ordered, 1),
        'p50': _percentile(ordered, 50),
        'p95': _percentile(ordered, 95),
        'maximum': ordered[-1],
    }


def common_window_timestamp_alignment(
    reference_timestamps,
    comparison_timestamps,
    coverage_thresholds_ns=LIDAR_IMAGE_COVERAGE_THRESHOLDS_NS,
):
    """Summarize nearest matches inside the streams' common time window."""
    references = sorted(int(value) for value in reference_timestamps)
    comparisons = sorted(int(value) for value in comparison_timestamps)
    result = {
        'common_start_ns': None,
        'common_end_ns': None,
        'common_duration_s': None,
        'reference_total': len(references),
        'comparison_total': len(comparisons),
        'reference_count': 0,
        'comparison_count': 0,
        'reference_common_ratio': 0.0,
        'comparison_common_ratio': 0.0,
        'difference_statistics': DifferenceStatistics(),
        'absolute_percentiles_ns': {
            'p50': None,
            'p90': None,
            'p95': None,
            'p99': None,
            'maximum': None,
        },
        'coverage': {
            int(threshold_ns): {'count': 0, 'ratio': 0.0}
            for threshold_ns in coverage_thresholds_ns
        },
    }
    if not references or not comparisons:
        return result

    common_start_ns = max(references[0], comparisons[0])
    common_end_ns = min(references[-1], comparisons[-1])
    result['common_start_ns'] = common_start_ns
    result['common_end_ns'] = common_end_ns
    if common_end_ns < common_start_ns:
        return result

    reference_start = bisect.bisect_left(references, common_start_ns)
    reference_end = bisect.bisect_right(references, common_end_ns)
    comparison_start = bisect.bisect_left(comparisons, common_start_ns)
    comparison_end = bisect.bisect_right(comparisons, common_end_ns)
    common_references = references[reference_start:reference_end]
    common_comparisons = comparisons[comparison_start:comparison_end]

    result['common_duration_s'] = (
        common_end_ns - common_start_ns) * 1e-9
    result['reference_count'] = len(common_references)
    result['comparison_count'] = len(common_comparisons)
    result['reference_common_ratio'] = (
        len(common_references) / len(references)
    )
    result['comparison_common_ratio'] = (
        len(common_comparisons) / len(comparisons)
    )

    absolute_differences_ns = []
    statistics = result['difference_statistics']
    for reference_ns, comparison_ns in _nearest_timestamp_pairs(
        common_references,
        common_comparisons,
    ):
        statistics.add(reference_ns, comparison_ns)
        absolute_differences_ns.append(abs(comparison_ns - reference_ns))

    ordered_differences = sorted(absolute_differences_ns)
    result['absolute_percentiles_ns'] = {
        'p50': _percentile(ordered_differences, 50),
        'p90': _percentile(ordered_differences, 90),
        'p95': _percentile(ordered_differences, 95),
        'p99': _percentile(ordered_differences, 99),
        'maximum': (
            ordered_differences[-1]
            if ordered_differences
            else None
        ),
    }
    for threshold_ns in coverage_thresholds_ns:
        threshold_ns = int(threshold_ns)
        matched_count = bisect.bisect_right(
            ordered_differences, threshold_ns)
        result['coverage'][threshold_ns] = {
            'count': matched_count,
            'ratio': (
                matched_count / len(ordered_differences)
                if ordered_differences
                else 0.0
            ),
        }
    return result


def match_timestamp_streams_within_tolerance(
    reference_timestamps,
    comparison_timestamps,
    tolerance_ns,
):
    """One-to-one match two timestamp streams within an inclusive tolerance."""
    references = sorted(int(value) for value in reference_timestamps)
    comparisons = sorted(int(value) for value in comparison_timestamps)
    tolerance_ns = int(tolerance_ns)
    if tolerance_ns < 0:
        raise ValueError('tolerance_ns must be non-negative')

    reference_index = 0
    comparison_index = 0
    statistics = DifferenceStatistics()
    while (
        reference_index < len(references)
        and comparison_index < len(comparisons)
    ):
        reference_ns = references[reference_index]
        comparison_ns = comparisons[comparison_index]
        if comparison_ns < reference_ns - tolerance_ns:
            comparison_index += 1
        elif reference_ns < comparison_ns - tolerance_ns:
            reference_index += 1
        else:
            statistics.add(reference_ns, comparison_ns)
            reference_index += 1
            comparison_index += 1

    matched_count = statistics.count
    return {
        'tolerance_ns': tolerance_ns,
        'reference_count': len(references),
        'comparison_count': len(comparisons),
        'matched_count': matched_count,
        'unmatched_reference_count': len(references) - matched_count,
        'unmatched_comparison_count': len(comparisons) - matched_count,
        'reference_coverage': (
            matched_count / len(references) if references else 0.0
        ),
        'comparison_coverage': (
            matched_count / len(comparisons) if comparisons else 0.0
        ),
        'difference_statistics': statistics,
    }


def samples_around_references(
    reference_timestamps,
    sample_timestamps,
    half_window_ns,
):
    """Count samples before and after each reference within a time window."""
    references = sorted(int(value) for value in reference_timestamps)
    samples = sorted(int(value) for value in sample_timestamps)
    half_window_ns = int(half_window_ns)
    if half_window_ns < 0:
        raise ValueError('half_window_ns must be non-negative')

    total_counts = []
    before_counts = []
    after_counts = []
    references_in_sample_span = 0
    for reference_ns in references:
        lower_index = bisect.bisect_left(
            samples, reference_ns - half_window_ns)
        upper_index = bisect.bisect_right(
            samples, reference_ns + half_window_ns)
        before_end = bisect.bisect_left(samples, reference_ns)
        after_start = bisect.bisect_right(samples, reference_ns)
        total_counts.append(upper_index - lower_index)
        before_counts.append(max(0, before_end - lower_index))
        after_counts.append(max(0, upper_index - after_start))
        if (
            samples
            and samples[0] <= reference_ns <= samples[-1]
        ):
            references_in_sample_span += 1

    return {
        'half_window_ns': half_window_ns,
        'reference_count': len(references),
        'sample_count': len(samples),
        'references_in_sample_span': references_in_sample_span,
        'references_outside_sample_span': (
            len(references) - references_in_sample_span
        ),
        'with_samples_count': sum(
            count > 0 for count in total_counts),
        'without_samples_count': sum(
            count == 0 for count in total_counts),
        'with_samples_on_both_sides_count': sum(
            before > 0 and after > 0
            for before, after in zip(before_counts, after_counts)
        ),
        'total': _count_distribution(total_counts),
        'before': _count_distribution(before_counts),
        'after': _count_distribution(after_counts),
    }


def _format_seconds(value):
    if value is None:
        return 'missing'
    return '{:.9f}s'.format(value)


def _format_rate(value):
    if value is None:
        return 'missing'
    return '{:.6f}Hz'.format(value)


def _format_absolute_time(timestamp_ns):
    if timestamp_ns is None:
        return 'missing'
    seconds = timestamp_ns * 1e-9
    try:
        value = datetime.datetime.fromtimestamp(
            seconds, datetime.timezone.utc).astimezone()
        return '{} (ns={})'.format(
            value.isoformat(timespec='microseconds'),
            timestamp_ns,
        )
    except (OverflowError, OSError, ValueError):
        return 'unrepresentable (ns={})'.format(timestamp_ns)


def _format_ratio(value):
    if value is None:
        return 'missing'
    return '{:.2%}'.format(value)


def _format_nanoseconds_as_seconds(value):
    if value is None:
        return 'missing'
    return _format_seconds(float(value) * 1e-9)


def _format_number(value):
    if value is None:
        return 'missing'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return '{:.3f}'.format(value)


def _format_matrix(values):
    if values is None:
        return 'missing'
    return '[{}]'.format(', '.join(
        '{:.12g}'.format(value) for value in values))


def _normalize_frame_id(frame_id):
    return str(frame_id or '').strip().lstrip('/')


def _message_frame_id(message):
    try:
        return _normalize_frame_id(message.header.frame_id)
    except AttributeError:
        return ''


def _float_tuple(value):
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _camera_info_snapshot(message):
    """Extract stable calibration fields and their readiness issues."""
    reasons = []
    frame_id = _message_frame_id(message)
    if not frame_id:
        reasons.append('missing frame_id')

    try:
        width = int(message.width)
        height = int(message.height)
    except (AttributeError, TypeError, ValueError):
        width = 0
        height = 0
        reasons.append('missing image dimensions')
    else:
        if width <= 0 or height <= 0:
            reasons.append('non-positive image dimensions')

    k = _float_tuple(getattr(message, 'k', None))
    if k is None or len(k) != 9:
        reasons.append('K must contain 9 numeric values')
    elif not all(math.isfinite(value) for value in k):
        reasons.append('K contains non-finite values')
    else:
        if k[0] <= 0.0 or k[4] <= 0.0:
            reasons.append('K has non-positive focal length')
        if abs(k[8] - 1.0) > 1e-6:
            reasons.append('K[8] differs from one')
        if width > 0 and not 0.0 <= k[2] <= float(width):
            reasons.append('K principal point x is outside image')
        if height > 0 and not 0.0 <= k[5] <= float(height):
            reasons.append('K principal point y is outside image')

    d = _float_tuple(getattr(message, 'd', ()))
    r = _float_tuple(getattr(message, 'r', ()))
    p = _float_tuple(getattr(message, 'p', ()))
    if d is None:
        reasons.append('D contains non-numeric values')
    elif not all(math.isfinite(value) for value in d):
        reasons.append('D contains non-finite values')
    if r is None or len(r) not in (0, 9):
        reasons.append('R must be empty or contain 9 numeric values')
    elif not all(math.isfinite(value) for value in r):
        reasons.append('R contains non-finite values')
    elif len(r) == 9:
        rows = (r[0:3], r[3:6], r[6:9])
        orthogonality_error = max(
            abs(sum(
                rows[row][axis] * rows[column][axis]
                for axis in range(3)
            ) - (1.0 if row == column else 0.0))
            for row in range(3)
            for column in range(3)
        )
        determinant = (
            r[0] * (r[4] * r[8] - r[5] * r[7])
            - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6])
        )
        if orthogonality_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
            reasons.append('R is not a proper orthonormal rotation')
    if p is None or len(p) not in (0, 12):
        reasons.append('P must be empty or contain 12 numeric values')
    elif not all(math.isfinite(value) for value in p):
        reasons.append('P contains non-finite values')

    distortion_model = str(
        getattr(message, 'distortion_model', '') or '')
    binning_x = int(getattr(message, 'binning_x', 0) or 0)
    binning_y = int(getattr(message, 'binning_y', 0) or 0)
    roi = getattr(message, 'roi', None)
    roi_values = (
        int(getattr(roi, 'x_offset', 0) or 0),
        int(getattr(roi, 'y_offset', 0) or 0),
        int(getattr(roi, 'height', 0) or 0),
        int(getattr(roi, 'width', 0) or 0),
        bool(getattr(roi, 'do_rectify', False)),
    )
    fingerprint_payload = (
        frame_id,
        width,
        height,
        distortion_model,
        d,
        k,
        r,
        p,
        binning_x,
        binning_y,
        roi_values,
    )
    fingerprint = hashlib.sha256(
        repr(fingerprint_payload).encode('utf-8')
    ).hexdigest()[:16]
    return {
        'fingerprint': fingerprint,
        'valid': not reasons,
        'reasons': tuple(reasons),
        'frame_id': frame_id,
        'width': width,
        'height': height,
        'distortion_model': distortion_model,
        'd': d,
        'k': k,
        'r': r,
        'p': p,
    }


class CameraInfoAudit:
    """Track CameraInfo validity and stable calibration fingerprints."""

    def __init__(self):
        """Initialize empty calibration counters."""
        self.count = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.reason_counts = collections.Counter()
        self.fingerprints = {}

    def add(self, message):
        """Include one CameraInfo message in the audit."""
        snapshot = _camera_info_snapshot(message)
        self.count += 1
        if snapshot['valid']:
            self.valid_count += 1
        else:
            self.invalid_count += 1
            self.reason_counts.update(snapshot['reasons'])

        fingerprint = snapshot['fingerprint']
        entry = self.fingerprints.get(fingerprint)
        if entry is None:
            entry = dict(snapshot)
            entry.update({
                'count': 0,
                'valid_count': 0,
                'first_index': self.count,
            })
            self.fingerprints[fingerprint] = entry
        entry['count'] += 1
        if snapshot['valid']:
            entry['valid_count'] += 1

    @property
    def candidate(self):
        """Return the most frequently observed valid calibration."""
        candidates = [
            entry
            for entry in self.fingerprints.values()
            if entry['valid_count'] > 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda entry: (
                entry['valid_count'],
                entry['count'],
                -entry['first_index'],
            ),
        )

    @property
    def valid_fingerprint_count(self):
        """Return the number of distinct valid calibrations."""
        return sum(
            entry['valid_count'] > 0
            for entry in self.fingerprints.values()
        )


def _transform_snapshot(transform):
    """Validate one TransformStamped and return a stable edge value."""
    reasons = []
    try:
        parent = _normalize_frame_id(transform.header.frame_id)
        child = _normalize_frame_id(transform.child_frame_id)
    except AttributeError:
        parent = ''
        child = ''
    if not parent:
        reasons.append('missing parent frame')
    if not child:
        reasons.append('missing child frame')
    if parent and child and parent == child:
        reasons.append('parent and child frames are identical')

    try:
        translation = (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            float(transform.transform.translation.z),
        )
        quaternion = (
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        )
    except (AttributeError, TypeError, ValueError):
        translation = None
        quaternion = None
        reasons.append('missing numeric transform')
    else:
        if not all(math.isfinite(value) for value in translation):
            reasons.append('translation contains non-finite values')
        if not all(math.isfinite(value) for value in quaternion):
            reasons.append('quaternion contains non-finite values')
        else:
            quaternion_norm = math.sqrt(sum(
                value * value for value in quaternion))
            if quaternion_norm <= 1e-12:
                reasons.append('quaternion norm is zero')
            else:
                if abs(quaternion_norm - 1.0) > 1e-3:
                    reasons.append('quaternion norm differs from one')
                quaternion = tuple(
                    value / quaternion_norm for value in quaternion)
                for value in reversed(quaternion):
                    if abs(value) > 1e-15:
                        if value < 0:
                            quaternion = tuple(
                                -item for item in quaternion)
                        break

    return {
        'valid': not reasons,
        'reasons': tuple(reasons),
        'parent': parent,
        'child': child,
        'value': (translation, quaternion),
    }


class TfStaticAudit:
    """Audit static transform validity, uniqueness, and sensor connectivity."""

    def __init__(self):
        """Initialize empty static-transform counters."""
        self.message_count = 0
        self.transform_count = 0
        self.valid_transform_count = 0
        self.invalid_transform_count = 0
        self.reason_counts = collections.Counter()
        self.edge_values = {}
        self.parents_by_child = {}
        self.frames = set()
        self.read_error = None

    def add(self, message):
        """Include all transforms from one TFMessage."""
        self.message_count += 1
        transforms = getattr(message, 'transforms', None)
        if transforms is None:
            self.reason_counts['TFMessage has no transforms field'] += 1
            return
        for transform in transforms:
            self.transform_count += 1
            snapshot = _transform_snapshot(transform)
            if not snapshot['valid']:
                self.invalid_transform_count += 1
                self.reason_counts.update(snapshot['reasons'])
                continue

            self.valid_transform_count += 1
            parent = snapshot['parent']
            child = snapshot['child']
            self.frames.update((parent, child))
            self.edge_values.setdefault(
                (parent, child), set()).add(snapshot['value'])
            self.parents_by_child.setdefault(child, set()).add(parent)

    @property
    def conflicting_edges(self):
        """Return edges published with more than one transform value."""
        return [
            edge
            for edge, values in self.edge_values.items()
            if len(values) > 1
        ]

    @property
    def children_with_multiple_parents(self):
        """Return child frames assigned to more than one parent."""
        return {
            child: parents
            for child, parents in self.parents_by_child.items()
            if len(parents) > 1
        }

    @property
    def directed_cycles(self):
        """Return representative directed cycles in the static TF graph."""
        neighbours = collections.defaultdict(set)
        for parent, child in self.edge_values:
            neighbours[parent].add(child)

        state = {}
        stack = []
        cycles = set()

        def visit(frame):
            state[frame] = 1
            stack.append(frame)
            for child in sorted(neighbours[frame]):
                child_state = state.get(child, 0)
                if child_state == 0:
                    visit(child)
                elif child_state == 1:
                    start = stack.index(child)
                    cycle = tuple(stack[start:] + [child])
                    cycles.add(cycle)
            stack.pop()
            state[frame] = 2

        for frame in sorted(self.frames):
            if state.get(frame, 0) == 0:
                visit(frame)
        return sorted(cycles)

    def path(self, source_frame, target_frame):
        """Return an undirected static-TF path between two frames."""
        source_frame = _normalize_frame_id(source_frame)
        target_frame = _normalize_frame_id(target_frame)
        if (
            not source_frame
            or not target_frame
            or source_frame not in self.frames
            or target_frame not in self.frames
        ):
            return None
        if source_frame == target_frame:
            return [source_frame]

        neighbours = collections.defaultdict(set)
        for parent, child in self.edge_values:
            neighbours[parent].add(child)
            neighbours[child].add(parent)
        queue = collections.deque([(source_frame, [source_frame])])
        visited = {source_frame}
        while queue:
            frame, path = queue.popleft()
            for neighbour in sorted(neighbours[frame]):
                if neighbour in visited:
                    continue
                next_path = path + [neighbour]
                if neighbour == target_frame:
                    return next_path
                visited.add(neighbour)
                queue.append((neighbour, next_path))
        return None

    def readiness_warnings(self, lidar_frame_id, image_frame_id):
        """Return reasons the recorded static TF is not motion-ready."""
        warnings = []
        if self.read_error:
            warnings.append(
                'cannot audit /tf_static: {}'.format(self.read_error))
            return warnings
        if self.message_count == 0 or self.transform_count == 0:
            warnings.append('/tf_static contains no transforms')
        if self.invalid_transform_count:
            warnings.append(
                '/tf_static contains {} invalid transforms'.format(
                    self.invalid_transform_count))
        if self.conflicting_edges:
            warnings.append(
                '/tf_static changes value for {} parent-child edges'.format(
                    len(self.conflicting_edges)))
        if self.children_with_multiple_parents:
            warnings.append(
                '/tf_static assigns multiple parents to {} child '
                'frames'.format(
                    len(self.children_with_multiple_parents)))
        if self.directed_cycles:
            warnings.append(
                '/tf_static contains {} directed cycle(s)'.format(
                    len(self.directed_cycles)))

        lidar_frame_id = _normalize_frame_id(lidar_frame_id)
        image_frame_id = _normalize_frame_id(image_frame_id)
        if not lidar_frame_id:
            warnings.append('LiDAR header.frame_id is empty')
        if not image_frame_id:
            warnings.append('image header.frame_id is empty')
        if (
            lidar_frame_id
            and image_frame_id
            and lidar_frame_id == image_frame_id
        ):
            warnings.append(
                'LiDAR and image unexpectedly use the same frame_id {}'.format(
                    lidar_frame_id))
        elif (
            lidar_frame_id
            and image_frame_id
            and self.path(lidar_frame_id, image_frame_id) is None
        ):
            warnings.append(
                '/tf_static has no chain from {} to {}'.format(
                    lidar_frame_id, image_frame_id))
        return warnings


def _database_uri(path):
    absolute_path = str(path.resolve())
    quoted_path = urllib.parse.quote(absolute_path, safe='/')
    return 'file:{}?mode=ro'.format(quoted_path)


def _database_paths(run_dir, bag_name):
    bag_dir = run_dir / bag_name
    if not bag_dir.is_dir():
        raise AuditError('missing bag directory: {}'.format(bag_dir))
    database_paths = sorted(bag_dir.glob('*.db3'))
    if not database_paths:
        raise AuditError('no .db3 files found in {}'.format(bag_dir))
    return database_paths


def _message_rows(run_dir, bag_name, topic_name, expected_type):
    """Yield deserialized messages ordered by rosbag receipt timestamp."""
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise AuditError(
            'ROS 2 Python modules are unavailable; source the Foxy '
            'environment before running this script: {}'.format(error)
        ) from error

    try:
        message_class = get_message(expected_type)
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ) as error:
        raise AuditError(
            'cannot load message type {} for {}; source the Livox install '
            'setup.bash as well as ROS Foxy: {}'.format(
                expected_type, topic_name, error)
        ) from error

    database_paths = _database_paths(run_dir, bag_name)
    found_topic = False

    with contextlib.ExitStack() as stack:
        cursors = []
        for database_index, database_path in enumerate(database_paths):
            try:
                connection = sqlite3.connect(
                    _database_uri(database_path),
                    uri=True,
                )
                stack.callback(connection.close)
                connection.execute('PRAGMA query_only = ON')
                topic_rows = connection.execute(
                    'SELECT id, type FROM topics WHERE name = ?',
                    (topic_name,),
                ).fetchall()
            except sqlite3.DatabaseError as error:
                raise AuditError(
                    'cannot read {}: {}'.format(database_path, error)
                ) from error

            for topic_id, actual_type in topic_rows:
                found_topic = True
                if actual_type != expected_type:
                    raise AuditError(
                        '{} has type {} in {}, expected {}'.format(
                            topic_name,
                            actual_type,
                            database_path,
                            expected_type,
                        )
                    )
                cursor = connection.execute(
                    'SELECT timestamp, data '
                    'FROM messages '
                    'WHERE topic_id = ? '
                    'ORDER BY timestamp',
                    (topic_id,),
                )
                cursors.append((database_index, cursor, database_path))

        if not found_topic:
            raise AuditError(
                'topic {} was not found under {}'.format(
                    topic_name, run_dir / bag_name)
            )

        heap = []
        sequence = 0
        for database_index, cursor, database_path in cursors:
            row = cursor.fetchone()
            if row is not None:
                heapq.heappush(
                    heap,
                    (
                        int(row[0]),
                        sequence,
                        row[1],
                        cursor,
                        database_path,
                    ),
                )
                sequence += 1

        while heap:
            (
                receipt_ns,
                unused_sequence,
                serialized_data,
                cursor,
                database_path,
            ) = heapq.heappop(heap)
            try:
                message = deserialize_message(
                    bytes(serialized_data), message_class)
            except Exception as error:
                raise AuditError(
                    'cannot deserialize {} at receipt ns={} in {}: {}'.format(
                        topic_name,
                        receipt_ns,
                        database_path,
                        error,
                    )
                ) from error
            yield receipt_ns, message

            row = cursor.fetchone()
            if row is not None:
                heapq.heappush(
                    heap,
                    (
                        int(row[0]),
                        sequence,
                        row[1],
                        cursor,
                        database_path,
                    ),
                )
                sequence += 1


def _header_timestamp_ns(message):
    try:
        stamp = message.header.stamp
        return int(stamp.sec) * 1000000000 + int(stamp.nanosec)
    except AttributeError as error:
        raise AuditError(
            'message has no usable header.stamp: {}'.format(error)
        ) from error


def _audit_topic(run_dir, bag_name, topic_name, expected_type):
    receipt_statistics = GapStatistics()
    header_statistics = GapStatistics()
    header_timestamps = []
    frame_ids = collections.Counter()
    image_shapes = collections.Counter()
    zero_header_count = 0
    timebase_differences = DifferenceStatistics()
    camera_info_audit = (
        CameraInfoAudit()
        if topic_name == '/camera/color/camera_info'
        else None
    )
    progress_interval = (
        2000 if topic_name == '/timefix/imu' else 100
    )

    for receipt_ns, message in _message_rows(
        run_dir,
        bag_name,
        topic_name,
        expected_type,
    ):
        receipt_statistics.add(receipt_ns)
        header_ns = _header_timestamp_ns(message)
        frame_id = _message_frame_id(message)
        if frame_id:
            frame_ids[frame_id] += 1
        if topic_name == '/camera/color/image_raw':
            try:
                image_shape = (int(message.width), int(message.height))
            except (AttributeError, TypeError, ValueError):
                image_shape = None
            if (
                image_shape is not None
                and image_shape[0] > 0
                and image_shape[1] > 0
            ):
                image_shapes[image_shape] += 1
        if header_ns == 0:
            zero_header_count += 1
        else:
            header_statistics.add(header_ns)
            header_timestamps.append(header_ns)

        if camera_info_audit is not None:
            camera_info_audit.add(message)

        if topic_name == '/timefix/lidar':
            try:
                timebase_ns = int(message.timebase)
            except AttributeError as error:
                raise AuditError(
                    'Livox CustomMsg has no timebase: {}'.format(error)
                ) from error
            if header_ns and timebase_ns:
                timebase_differences.add(header_ns, timebase_ns)

        if receipt_statistics.count % progress_interval == 0:
            print(
                '  scanned {} messages...'.format(
                    receipt_statistics.count)
            )

    if receipt_statistics.count == 0:
        raise AuditError('{} contains no messages'.format(topic_name))

    return {
        'receipt': receipt_statistics,
        'header': header_statistics,
        'header_timestamps': header_timestamps,
        'frame_ids': frame_ids,
        'image_shapes': image_shapes,
        'zero_header_count': zero_header_count,
        'timebase_differences': timebase_differences,
        'camera_info_audit': camera_info_audit,
    }


def _audit_tf_static(run_dir):
    audit = TfStaticAudit()
    bag_name, topic_name, expected_type = TF_STATIC_TOPIC
    try:
        for unused_receipt_ns, message in _message_rows(
            run_dir,
            bag_name,
            topic_name,
            expected_type,
        ):
            audit.add(message)
    except AuditError as error:
        audit.read_error = str(error)
    return audit


def _dominant_frame_id(result):
    frame_ids = result.get('frame_ids', {})
    if not frame_ids:
        return ''
    return min(
        frame_ids,
        key=lambda frame_id: (-frame_ids[frame_id], frame_id),
    )


def _print_gap_statistics(label, statistics):
    print(
        '  {}: count={} duration={} rate={} max_gap={} '
        'non_monotonic={}'.format(
            label,
            statistics.count,
            _format_seconds(statistics.duration_s),
            _format_rate(statistics.rate_hz),
            _format_seconds(statistics.maximum_gap_s),
            statistics.non_monotonic_count,
        )
    )
    if statistics.maximum_gap_ns is not None:
        print(
            '    max_gap_start={}'.format(
                _format_absolute_time(
                    statistics.maximum_gap_start_ns))
        )
        print(
            '    max_gap_end={}'.format(
                _format_absolute_time(
                    statistics.maximum_gap_end_ns))
        )


def _print_header_max_gap_gate(topic_name, statistics):
    """Print and return the strict header maximum-gap gate result."""
    threshold_ns = HEADER_MAX_GAP_THRESHOLDS_NS[topic_name]
    maximum_gap_ns = statistics.maximum_gap_ns
    passed = (
        maximum_gap_ns is not None
        and maximum_gap_ns <= threshold_ns
    )
    print(
        '  header_max_gap_gate: {} topic={} max_gap={} limit={}'.format(
            'PASS' if passed else 'FAIL',
            topic_name,
            _format_nanoseconds_as_seconds(maximum_gap_ns),
            _format_nanoseconds_as_seconds(threshold_ns),
        )
    )
    if maximum_gap_ns is None:
        print(
            'NOT_READY: {} header max gap is missing; at least two '
            'non-zero header timestamps are required'.format(topic_name)
        )
    elif not passed:
        print(
            'NOT_READY: {} header max gap {} exceeds the {} limit'.format(
                topic_name,
                _format_nanoseconds_as_seconds(maximum_gap_ns),
                _format_nanoseconds_as_seconds(threshold_ns),
            )
        )
    return passed


def _print_difference_statistics(label, statistics):
    if not statistics.count:
        print('{}: MISSING'.format(label))
        return
    print(
        '{}: count={} signed_min={} signed_mean={} signed_max={} '
        'abs_min={} abs_mean={} abs_rms={} abs_max={}'.format(
            label,
            statistics.count,
            _format_seconds(statistics.minimum_signed_ns * 1e-9),
            _format_seconds(statistics.mean_signed_s),
            _format_seconds(statistics.maximum_signed_ns * 1e-9),
            _format_seconds(statistics.minimum_absolute_ns * 1e-9),
            _format_seconds(statistics.mean_absolute_s),
            _format_seconds(statistics.root_mean_square_s),
            _format_seconds(statistics.maximum_absolute_ns * 1e-9),
        )
    )
    print(
        '  max_abs_reference={}'.format(
            _format_absolute_time(statistics.maximum_reference_ns))
    )
    print(
        '  max_abs_comparison={}'.format(
            _format_absolute_time(statistics.maximum_comparison_ns))
    )


def _print_common_window_alignment(alignment):
    if (
        alignment['common_start_ns'] is None
        or alignment['common_end_ns'] is None
        or alignment['common_end_ns'] < alignment['common_start_ns']
    ):
        print('lidar_image_common_window: MISSING')
        print(
            'NOT_READY: LiDAR and image header timestamps have no '
            'common window')
        return

    print(
        'lidar_image_common_window: start={} end={} duration={} '
        'lidar={}/{} ({}) image={}/{} ({})'.format(
            _format_absolute_time(alignment['common_start_ns']),
            _format_absolute_time(alignment['common_end_ns']),
            _format_seconds(alignment['common_duration_s']),
            alignment['reference_count'],
            alignment['reference_total'],
            _format_ratio(alignment['reference_common_ratio']),
            alignment['comparison_count'],
            alignment['comparison_total'],
            _format_ratio(alignment['comparison_common_ratio']),
        )
    )
    percentiles = alignment['absolute_percentiles_ns']
    print(
        'lidar_image_abs_delta_percentiles: p50={} p90={} p95={} '
        'p99={} max={}'.format(
            _format_nanoseconds_as_seconds(percentiles['p50']),
            _format_nanoseconds_as_seconds(percentiles['p90']),
            _format_nanoseconds_as_seconds(percentiles['p95']),
            _format_nanoseconds_as_seconds(percentiles['p99']),
            _format_nanoseconds_as_seconds(percentiles['maximum']),
        )
    )
    coverage_fields = []
    for threshold_ns, item in sorted(alignment['coverage'].items()):
        coverage_fields.append(
            '<={}={}/{}({})'.format(
                _format_nanoseconds_as_seconds(threshold_ns),
                item['count'],
                alignment['difference_statistics'].count,
                _format_ratio(item['ratio']),
            )
        )
    print('lidar_image_coverage: {}'.format(
        ' '.join(coverage_fields)))
    if (
        alignment['reference_count'] == 0
        or alignment['comparison_count'] == 0
    ):
        print(
            'NOT_READY: the LiDAR/image common header window contains '
            'no samples from one stream')


def _print_image_info_matching(matching):
    print(
        'image_camera_info_match_le_1ms: matched={} image_count={} '
        'camera_info_count={} unmatched_image={} unmatched_camera_info={} '
        'image_coverage={} camera_info_coverage={}'.format(
            matching['matched_count'],
            matching['reference_count'],
            matching['comparison_count'],
            matching['unmatched_reference_count'],
            matching['unmatched_comparison_count'],
            _format_ratio(matching['reference_coverage']),
            _format_ratio(matching['comparison_coverage']),
        )
    )
    if (
        matching['reference_coverage'] < MIN_IMAGE_INFO_MATCH_COVERAGE
        or matching['comparison_coverage'] < MIN_IMAGE_INFO_MATCH_COVERAGE
    ):
        print(
            'NOT_READY: fewer than {:.0%} of image and CameraInfo '
            'header timestamps match within 1 ms'.format(
                MIN_IMAGE_INFO_MATCH_COVERAGE))


def _print_lidar_imu_windows(window_counts):
    distribution = window_counts['total']
    before = window_counts['before']
    after = window_counts['after']
    print(
        'lidar_imu_window_pm_0.15s: lidar_count={} imu_count={} '
        'with_imu={} without_imu={} both_sides={} '
        'inside_imu_span={} outside_imu_span={}'.format(
            window_counts['reference_count'],
            window_counts['sample_count'],
            window_counts['with_samples_count'],
            window_counts['without_samples_count'],
            window_counts['with_samples_on_both_sides_count'],
            window_counts['references_in_sample_span'],
            window_counts['references_outside_sample_span'],
        )
    )
    print(
        'lidar_imu_samples_per_window: min={} p1={} p50={} p95={} '
        'max={}'.format(
            _format_number(distribution['minimum']),
            _format_number(distribution['p1']),
            _format_number(distribution['p50']),
            _format_number(distribution['p95']),
            _format_number(distribution['maximum']),
        )
    )
    print(
        'lidar_imu_samples_before: min={} p50={} p95={} max={}'.format(
            _format_number(before['minimum']),
            _format_number(before['p50']),
            _format_number(before['p95']),
            _format_number(before['maximum']),
        )
    )
    print(
        'lidar_imu_samples_after: min={} p50={} p95={} max={}'.format(
            _format_number(after['minimum']),
            _format_number(after['p50']),
            _format_number(after['p95']),
            _format_number(after['maximum']),
        )
    )
    if (
        window_counts['reference_count'] == 0
        or window_counts['sample_count'] == 0
    ):
        print(
            'NOT_READY: LiDAR or IMU has no non-zero header timestamps')
    elif window_counts['without_samples_count']:
        print(
            'NOT_READY: {} LiDAR frames have no IMU within '
            '+/-0.15 s'.format(window_counts['without_samples_count'])
        )


def _print_camera_info_audit(
        audit,
        image_frame_id,
        image_shapes=None):
    print(
        'camera_info_calibration: count={} valid={} invalid={} '
        'unique_fingerprints={} valid_fingerprints={}'.format(
            audit.count,
            audit.valid_count,
            audit.invalid_count,
            len(audit.fingerprints),
            audit.valid_fingerprint_count,
        )
    )
    entries = sorted(
        audit.fingerprints.values(),
        key=lambda entry: entry['first_index'],
    )
    for entry in entries[:10]:
        print(
            '  camera_info_fingerprint={} count={} valid={} frame_id={} '
            'size={}x{}'.format(
                entry['fingerprint'],
                entry['count'],
                entry['valid_count'],
                entry['frame_id'] or 'missing',
                entry['width'],
                entry['height'],
            )
        )
    if len(entries) > 10:
        print(
            '  camera_info_fingerprints_omitted={}'.format(
                len(entries) - 10))

    candidate = audit.candidate
    if candidate is None:
        print('candidate_camera_k: MISSING')
        print('NOT_READY: no valid CameraInfo K candidate was recorded')
    else:
        print(
            'candidate_camera_k: {}'.format(
                _format_matrix(candidate['k'])))
        print(
            '  candidate_source: fingerprint={} count={} frame_id={} '
            'size={}x{}'.format(
                candidate['fingerprint'],
                candidate['valid_count'],
                candidate['frame_id'],
                candidate['width'],
                candidate['height'],
            )
        )
        normalized_image_frame = _normalize_frame_id(image_frame_id)
        if (
            normalized_image_frame
            and candidate['frame_id'] != normalized_image_frame
        ):
            print(
                'NOT_READY: CameraInfo frame {} does not match image '
                'frame {}'.format(
                    candidate['frame_id'],
                    normalized_image_frame,
                )
            )
        if image_shapes:
            dominant_image_shape = min(
                image_shapes,
                key=lambda shape: (-image_shapes[shape], shape),
            )
            candidate_shape = (
                candidate['width'],
                candidate['height'],
            )
            if candidate_shape != dominant_image_shape:
                print(
                    'NOT_READY: CameraInfo size {}x{} does not match '
                    'image size {}x{}'.format(
                        candidate_shape[0],
                        candidate_shape[1],
                        dominant_image_shape[0],
                        dominant_image_shape[1],
                    )
                )
        if candidate.get('d') and any(
                abs(value) > 1e-12 for value in candidate['d']):
            print(
                'NOT_READY: CameraInfo contains non-zero distortion but '
                'GA-BSVM currently applies a pinhole projection to the '
                'raw image')

    if audit.invalid_count:
        reasons = ', '.join(
            '{}={}'.format(reason, count)
            for reason, count in sorted(audit.reason_counts.items())
        )
        print(
            'NOT_READY: CameraInfo contains {} invalid messages ({})'.format(
                audit.invalid_count, reasons))
    if audit.valid_fingerprint_count > 1:
        print(
            'NOT_READY: CameraInfo calibration changed across {} valid '
            'fingerprints'.format(audit.valid_fingerprint_count))


def _print_tf_static_audit(audit, lidar_frame_id, image_frame_id):
    print(
        'tf_static: messages={} transforms={} valid={} invalid={} '
        'frames={} conflicting_edges={} multi_parent_children={} '
        'directed_cycles={}'.format(
            audit.message_count,
            audit.transform_count,
            audit.valid_transform_count,
            audit.invalid_transform_count,
            len(audit.frames),
            len(audit.conflicting_edges),
            len(audit.children_with_multiple_parents),
            len(audit.directed_cycles),
        )
    )
    lidar_frame_id = _normalize_frame_id(lidar_frame_id)
    image_frame_id = _normalize_frame_id(image_frame_id)
    print(
        'sensor_frame_ids: lidar={} image={}'.format(
            lidar_frame_id or 'missing',
            image_frame_id or 'missing',
        )
    )
    path = audit.path(lidar_frame_id, image_frame_id)
    print(
        'tf_static_lidar_to_image_path: {}'.format(
            ' -> '.join(path) if path else 'MISSING')
    )
    for reason, count in sorted(audit.reason_counts.items()):
        print('  tf_static_issue: {} count={}'.format(reason, count))
    for warning in audit.readiness_warnings(
        lidar_frame_id,
        image_frame_id,
    ):
        print('NOT_READY: {}'.format(warning))


def audit_run(run_dir):
    """Audit all required Lite3 timestamp streams."""
    results = {}
    header_max_gap_gates = {}
    print('RUN_DIR={}'.format(run_dir))
    print('=== Per-topic receipt and header statistics ===')
    for bag_name, topic_name, expected_type in TOPICS:
        print('\n{}'.format(topic_name))
        result = _audit_topic(
            run_dir,
            bag_name,
            topic_name,
            expected_type,
        )
        results[topic_name] = result
        _print_gap_statistics('receipt', result['receipt'])
        _print_gap_statistics('header', result['header'])
        header_max_gap_gates[topic_name] = _print_header_max_gap_gate(
            topic_name,
            result['header'],
        )
        print('  zero_header_count={}'.format(
            result['zero_header_count']))

    print('\n=== LiDAR timebase minus header.stamp ===')
    _print_difference_statistics(
        'lidar_timebase_minus_header',
        results['/timefix/lidar']['timebase_differences'],
    )

    print('\n=== Nearest header timestamp differences ===')
    lidar_to_image = nearest_timestamp_differences(
        iter(sorted(
            results['/timefix/lidar']['header_timestamps'])),
        iter(sorted(
            results[
                '/camera/color/image_raw']['header_timestamps'])),
    )
    _print_difference_statistics(
        'nearest_image_minus_lidar',
        lidar_to_image,
    )

    lidar_to_imu = nearest_timestamp_differences(
        iter(sorted(
            results['/timefix/lidar']['header_timestamps'])),
        iter(sorted(
            results['/timefix/imu']['header_timestamps'])),
    )
    _print_difference_statistics(
        'nearest_imu_minus_lidar',
        lidar_to_imu,
    )

    lidar_timestamps = results[
        '/timefix/lidar']['header_timestamps']
    image_timestamps = results[
        '/camera/color/image_raw']['header_timestamps']
    camera_info_timestamps = results[
        '/camera/color/camera_info']['header_timestamps']
    imu_timestamps = results[
        '/timefix/imu']['header_timestamps']

    lidar_image_alignment = common_window_timestamp_alignment(
        lidar_timestamps,
        image_timestamps,
    )
    print('\n=== Common-window LiDAR/image alignment ===')
    _print_common_window_alignment(lidar_image_alignment)

    image_info_matching = match_timestamp_streams_within_tolerance(
        image_timestamps,
        camera_info_timestamps,
        IMAGE_INFO_MATCH_TOLERANCE_NS,
    )
    print('\n=== Image/CameraInfo header matching ===')
    _print_image_info_matching(image_info_matching)

    lidar_imu_windows = samples_around_references(
        lidar_timestamps,
        imu_timestamps,
        LIDAR_IMU_HALF_WINDOW_NS,
    )
    print('\n=== LiDAR-centred IMU sample windows ===')
    _print_lidar_imu_windows(lidar_imu_windows)

    image_frame_id = _dominant_frame_id(
        results['/camera/color/image_raw'])
    lidar_frame_id = _dominant_frame_id(
        results['/timefix/lidar'])
    camera_info_audit = results[
        '/camera/color/camera_info']['camera_info_audit']

    print('\n=== CameraInfo calibration audit ===')
    _print_camera_info_audit(
        camera_info_audit,
        image_frame_id,
        results['/camera/color/image_raw'].get('image_shapes', {}),
    )

    tf_static_audit = _audit_tf_static(run_dir)
    print('\n=== /tf_static audit ===')
    _print_tf_static_audit(
        tf_static_audit,
        lidar_frame_id,
        image_frame_id,
    )

    percentiles = lidar_image_alignment[
        'absolute_percentiles_ns']
    lidar_image_coverage = lidar_image_alignment['coverage'].get(
        MAX_LIDAR_IMAGE_MATCH_NS,
        {'ratio': 0.0},
    )['ratio']
    imu_coverage = (
        lidar_imu_windows['with_samples_count']
        / lidar_imu_windows['reference_count']
        if lidar_imu_windows['reference_count']
        else 0.0
    )
    candidate = camera_info_audit.candidate
    image_shapes = results[
        '/camera/color/image_raw'].get('image_shapes', {})
    image_shape_matches = (
        candidate is not None
        and len(image_shapes) == 1
        and (
            candidate['width'],
            candidate['height'],
        ) in image_shapes
    )
    timestamps_are_valid = all(
        result['zero_header_count'] == 0
        and result['header'].non_monotonic_count == 0
        for result in results.values()
    )
    timebase_differences = results[
        '/timefix/lidar']['timebase_differences']
    lidar_timebase_matches = (
        timebase_differences.count
        == len(lidar_timestamps)
        and timebase_differences.maximum_absolute_ns == 0
    )
    sensor_header_passed = all([
        timestamps_are_valid,
        all(
            header_max_gap_gates.get(topic_name, False)
            for topic_name in HEADER_MAX_GAP_THRESHOLDS_NS
        ),
        lidar_timebase_matches,
        (
            lidar_image_alignment['common_duration_s'] is not None
            and lidar_image_alignment['common_duration_s'] >= 60.0
        ),
        percentiles['p99'] is not None,
        (
            percentiles['p99'] is not None
            and percentiles['p99'] <= MAX_LIDAR_IMAGE_P99_NS
        ),
        lidar_image_coverage >= MIN_LIDAR_IMAGE_COVERAGE,
        (
            image_info_matching['reference_coverage']
            >= MIN_IMAGE_INFO_MATCH_COVERAGE
        ),
        (
            image_info_matching['comparison_coverage']
            >= MIN_IMAGE_INFO_MATCH_COVERAGE
        ),
        imu_coverage >= MIN_LIDAR_IMU_COVERAGE,
        camera_info_audit.invalid_count == 0,
        camera_info_audit.valid_fingerprint_count == 1,
        image_shape_matches,
    ])

    calibration_warnings = tf_static_audit.readiness_warnings(
        lidar_frame_id,
        image_frame_id,
    )
    if (
        candidate is not None
        and candidate.get('d')
        and any(abs(value) > 1e-12 for value in candidate['d'])
    ):
        calibration_warnings.append(
            'raw image distortion is not modelled by GA-BSVM')

    print('\n=== Readiness summary ===')
    print(
        'SENSOR_HEADER_ALIGNMENT={}'.format(
            'PASS' if sensor_header_passed else 'FAIL'))
    print(
        'CALIBRATION_STRUCTURE={}'.format(
            'READY' if not calibration_warnings else 'NOT_READY'))
    print('MOTION_READY=NO')
    return {
        'sensor_header_alignment_passed': sensor_header_passed,
        'calibration_structure_ready': not calibration_warnings,
    }


def _parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=(
            'Read Lite3 rosbag SQLite files without modifying them and '
            'compare rosbag receipt times with message header timestamps.'
        )
    )
    parser.add_argument(
        'run_dir',
        type=pathlib.Path,
        help='lite3_concurrent_* directory containing imu/lidar/camera',
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the command-line timestamp audit."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    arguments = _parse_arguments(argv)
    run_dir = arguments.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(
            'ERROR: RUN_DIR is not a directory: {}'.format(run_dir),
            file=sys.stderr,
        )
        return 2
    try:
        audit_run(run_dir)
    except AuditError as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
