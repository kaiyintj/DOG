#!/usr/bin/env python3
"""Pure alignment and statistics helpers for CARLA reliability studies."""

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES, normalize_label


NORMAL_LIDAR_DTYPE = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('intensity', '<f4'),
])

SEMANTIC_LIDAR_DTYPE = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('cos_inc_angle', '<f4'),
    ('object_idx', '<u4'),
    ('object_tag', '<u4'),
])

CARLA_TAG_PROJECT_ALIASES = {
    'road': 'road',
    'roads': 'road',
    'road line': 'road',
    'road lines': 'road',
    'sidewalk': 'road',
    'sidewalks': 'road',
    'building': 'building',
    'buildings': 'building',
    'wall': 'building',
    'walls': 'building',
    'fence': 'building',
    'fences': 'building',
    'tree': 'tree',
    'trees': 'tree',
    'vegetation': 'tree',
    'person': 'person',
    'pedestrian': 'person',
    'pedestrians': 'person',
    'rider': 'person',
    'riders': 'person',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'bicycle': 'bicycle',
    'electric bicycle': 'electric bicycle',
    'motorcycle': 'motorcycle',
    'chair': 'chair',
    'bench': 'bench',
    'vehicle': 'vehicle',
    'vehicles': 'vehicle',
}

PER_POINT_COLUMNS = (
    'schema_version',
    'dataset_id',
    'frame_index',
    'lidar_frame',
    'lidar_timestamp',
    'rgb_frame',
    'rgb_timestamp',
    'requested_offset_ms',
    'actual_offset_ms',
    'timing_error_ms',
    'point_index',
    'semantic_match_index',
    'match_distance_m',
    'x_lidar',
    'y_lidar',
    'z_lidar',
    'intensity',
    'channel',
    'x_world',
    'y_world',
    'z_world',
    'u',
    'v',
    'actor_id',
    'carla_tag',
    'gt_project_id',
    'gt_class',
    'gt_supported',
    'gt_source',
    'pred_project_id',
    'pred_class',
    'correct',
    'pred_confidence',
    'gt_probability',
    'semantic_entropy',
    'range_m',
    'local_density',
    'view_radius',
    'angular_rms',
    'accel_deviation',
    'imu_status',
    'r_motion',
    'r_density',
    'r_range',
    'r_view',
    'r_semantic',
    'r_combined',
    'w_none',
    'w_semantic',
    'w_semantic_range',
    'w_semantic_range_density',
    'w_semantic_range_density_view',
    'w_full',
    'posterior_file',
    'posterior_row',
)

FACTOR_SUMMARY_COLUMNS = (
    'factor',
    'axis',
    'stratum',
    'bin_index',
    'bin_left',
    'bin_right',
    'right_closed',
    'sample_count',
    'unsupported_count',
    'correct_count',
    'accuracy',
    'error_rate',
    'mean_raw_value',
    'mean_reliability',
    'mean_pred_confidence',
    'mean_entropy',
    'nll',
    'brier',
    'reliability_gap',
)


@dataclasses.dataclass(frozen=True)
class PointMatchResult:
    """One-to-one mutual-nearest correspondence and quality diagnostics."""

    normal_indices: np.ndarray
    semantic_indices: np.ndarray
    distances_m: np.ndarray
    normal_count: int
    semantic_count: int
    invalid_normal_count: int
    invalid_semantic_count: int

    @property
    def matched_count(self):
        """Return the number of accepted point pairs."""
        return int(self.normal_indices.size)

    @property
    def matched_fraction(self):
        """Return accepted matches divided by the larger input cloud."""
        denominator = max(int(self.normal_count), int(self.semantic_count))
        return self.matched_count / denominator if denominator else 0.0

    def statistics(self):
        """Return JSON-compatible alignment statistics."""
        distances = np.asarray(self.distances_m, dtype=np.float64)
        return {
            'normal_count': int(self.normal_count),
            'semantic_count': int(self.semantic_count),
            'matched_count': self.matched_count,
            'matched_fraction': float(self.matched_fraction),
            'unmatched_normal_count': int(
                self.normal_count - self.matched_count),
            'unmatched_semantic_count': int(
                self.semantic_count - self.matched_count),
            'invalid_normal_count': int(self.invalid_normal_count),
            'invalid_semantic_count': int(self.invalid_semantic_count),
            'mean_distance_m': (
                float(np.mean(distances)) if distances.size else None),
            'p95_distance_m': (
                float(np.percentile(distances, 95))
                if distances.size else None),
            'max_distance_m': (
                float(np.max(distances)) if distances.size else None),
        }


@dataclasses.dataclass(frozen=True)
class HistoricalFrameSelection:
    """Result of a fail-closed timestamp-offset frame lookup."""

    frame: object
    frame_index: int
    target_timestamp: float
    selected_timestamp: float
    requested_offset_ms: float
    actual_offset_ms: float
    timing_error_ms: float
    accepted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class CarlaTruthLabel:
    """Project-ontology truth while preserving unsupported observations."""

    project_id: int
    class_name: str
    supported: bool
    source: str
    actor_id: int
    carla_tag: int


def decode_normal_lidar(raw_data):
    """Decode CARLA ray-cast LiDAR bytes into a copied structured array."""
    payload = memoryview(raw_data)
    if payload.nbytes % NORMAL_LIDAR_DTYPE.itemsize:
        raise ValueError(
            'normal LiDAR payload length must be a multiple of '
            f'{NORMAL_LIDAR_DTYPE.itemsize} bytes')
    return np.frombuffer(payload, dtype=NORMAL_LIDAR_DTYPE).copy()


def decode_semantic_lidar(raw_data):
    """Decode CARLA semantic LiDAR bytes without using labels as input."""
    payload = memoryview(raw_data)
    if payload.nbytes % SEMANTIC_LIDAR_DTYPE.itemsize:
        raise ValueError(
            'semantic LiDAR payload length must be a multiple of '
            f'{SEMANTIC_LIDAR_DTYPE.itemsize} bytes')
    return np.frombuffer(payload, dtype=SEMANTIC_LIDAR_DTYPE).copy()


def lidar_xyz(points):
    """Return an ``N x 3`` float64 coordinate matrix from decoded LiDAR."""
    values = np.asarray(points)
    required = {'x', 'y', 'z'}
    names = set(values.dtype.names or ())
    if not required.issubset(names):
        raise ValueError('decoded LiDAR must contain x, y and z fields')
    return np.column_stack((
        values['x'], values['y'], values['z'],
    )).astype(np.float64, copy=False)


def mutual_nearest_matches(normal_points, semantic_points, tolerance_m):
    """Match two clouds by finite, mutual-nearest, tolerance-gated pairs."""
    normal = np.asarray(normal_points, dtype=np.float64)
    semantic = np.asarray(semantic_points, dtype=np.float64)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError('normal_points must have shape (N, 3)')
    if semantic.ndim != 2 or semantic.shape[1] != 3:
        raise ValueError('semantic_points must have shape (M, 3)')
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('tolerance_m must be finite and non-negative')

    normal_valid = np.all(np.isfinite(normal), axis=1)
    semantic_valid = np.all(np.isfinite(semantic), axis=1)
    normal_original = np.flatnonzero(normal_valid)
    semantic_original = np.flatnonzero(semantic_valid)
    empty = np.empty(0, dtype=np.int64)
    empty_distance = np.empty(0, dtype=np.float64)
    if not normal_original.size or not semantic_original.size:
        return PointMatchResult(
            empty,
            empty.copy(),
            empty_distance,
            len(normal),
            len(semantic),
            int(np.count_nonzero(~normal_valid)),
            int(np.count_nonzero(~semantic_valid)),
        )

    normal_finite = normal[normal_valid]
    semantic_finite = semantic[semantic_valid]
    forward_distance, forward_index = cKDTree(semantic_finite).query(
        normal_finite, k=1)
    _, reverse_index = cKDTree(normal_finite).query(semantic_finite, k=1)
    normal_local = np.arange(normal_finite.shape[0], dtype=np.int64)
    mutual = reverse_index[forward_index] == normal_local
    accepted = mutual & (forward_distance <= tolerance)
    accepted_normal = normal_local[accepted]
    accepted_semantic = forward_index[accepted]
    return PointMatchResult(
        normal_indices=normal_original[accepted_normal],
        semantic_indices=semantic_original[accepted_semantic],
        distances_m=np.asarray(forward_distance[accepted], dtype=np.float64),
        normal_count=len(normal),
        semantic_count=len(semantic),
        invalid_normal_count=int(np.count_nonzero(~normal_valid)),
        invalid_semantic_count=int(np.count_nonzero(~semantic_valid)),
    )


def match_semantic_lidar_points(
    normal_points,
    semantic_points,
    tolerance_m,
):
    """Match decoded or ``N x 3`` normal and semantic LiDAR points."""
    normal = np.asarray(normal_points)
    semantic = np.asarray(semantic_points)
    if normal.dtype.names is not None:
        normal = lidar_xyz(normal)
    if semantic.dtype.names is not None:
        semantic = lidar_xyz(semantic)
    return mutual_nearest_matches(normal, semantic, tolerance_m)


def select_historical_frame(
    frames,
    lidar_timestamp,
    offset_ms,
    tolerance_ms,
    timestamp_key='timestamp',
):
    """Select an old RGB frame nearest ``lidar time - positive offset``."""
    lidar_time = float(lidar_timestamp)
    requested_offset = float(offset_ms)
    tolerance = float(tolerance_ms)
    if not np.isfinite(lidar_time):
        raise ValueError('lidar_timestamp must be finite')
    if not np.isfinite(requested_offset) or requested_offset < 0.0:
        raise ValueError('offset_ms must be finite and non-negative')
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('tolerance_ms must be finite and non-negative')

    target = lidar_time - requested_offset / 1000.0
    candidates = []
    for frame_index, frame in enumerate(frames):
        try:
            if isinstance(frame, dict):
                timestamp = float(frame[timestamp_key])
            else:
                timestamp = float(getattr(frame, timestamp_key))
        except (KeyError, AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(timestamp) and timestamp <= lidar_time + 1e-12:
            candidates.append((frame_index, frame, timestamp))

    if not candidates:
        return HistoricalFrameSelection(
            None, -1, target, np.nan, requested_offset, np.nan, np.nan,
            False, 'no_historical_frame')

    # Timestamp is the second key so equal-distance ties select the older RGB.
    frame_index, frame, selected = min(
        candidates,
        key=lambda item: (abs(item[2] - target), item[2], item[0]),
    )
    timing_error = abs(selected - target) * 1000.0
    actual_offset = (lidar_time - selected) * 1000.0
    accepted = timing_error <= tolerance + 1e-9
    return HistoricalFrameSelection(
        frame=frame if accepted else None,
        frame_index=int(frame_index),
        target_timestamp=float(target),
        selected_timestamp=float(selected),
        requested_offset_ms=requested_offset,
        actual_offset_ms=float(actual_offset),
        timing_error_ms=float(timing_error),
        accepted=bool(accepted),
        reason='ok' if accepted else 'outside_tolerance',
    )


def map_carla_truth(
    object_tag,
    object_idx,
    semantic_tag_ids,
    actor_fine_classes,
    supported_project_classes,
):
    """Map CARLA actor/tag truth while retaining unsupported project labels."""
    actor_id = int(object_idx)
    carla_tag = int(object_tag)
    actors = actor_fine_classes or {}
    actor_value = actors.get(actor_id)
    if actor_value is None:
        actor_value = actors.get(str(actor_id))
    if isinstance(actor_value, dict):
        actor_value = actor_value.get('benchmark_class')
    actor_label = normalize_label(actor_value) if actor_value else ''
    project_lookup = {
        normalize_label(name): index
        for index, name in enumerate(DEFAULT_CLASSES)
    }
    supported = {
        normalize_label(name) for name in supported_project_classes
    }

    if actor_label in project_lookup and actor_label != 'unknown background':
        class_name = DEFAULT_CLASSES[project_lookup[actor_label]]
        return CarlaTruthLabel(
            project_lookup[actor_label],
            class_name,
            normalize_label(class_name) in supported,
            'actor_metadata',
            actor_id,
            carla_tag,
        )

    matches = set()
    for name, tag_ids in (semantic_tag_ids or {}).items():
        try:
            contains = carla_tag in {int(value) for value in tag_ids}
        except TypeError:
            contains = False
        if contains:
            normalized_name = normalize_label(name)
            matches.add(CARLA_TAG_PROJECT_ALIASES.get(
                normalized_name,
                normalized_name,
            ))

    fine_vehicle = {
        'car', 'truck', 'bus', 'bicycle', 'electric bicycle', 'motorcycle',
    }
    if 'vehicle' in matches and matches.intersection(fine_vehicle):
        # Some CARLA releases alias Car to the generic Vehicles tag. Actor
        # metadata is required to make a fine-class claim in that situation.
        class_name = 'vehicle'
    else:
        priority = (
            'car', 'truck', 'bus', 'bicycle', 'electric bicycle',
            'motorcycle', 'person', 'road', 'building', 'tree',
        )
        class_name = next((name for name in priority if name in matches), '')

    project_id = project_lookup.get(class_name, -1)
    if project_id >= 0:
        canonical = DEFAULT_CLASSES[project_id]
        is_supported = normalize_label(canonical) in supported
    else:
        canonical = class_name or 'unmapped'
        is_supported = False
    return CarlaTruthLabel(
        project_id,
        canonical,
        bool(is_supported),
        'semantic_tag' if class_name else 'unmapped_tag',
        actor_id,
        carla_tag,
    )


def map_carla_ground_truth(
    object_tag,
    object_idx,
    semantic_tag_ids,
    actor_fine_classes,
    supported_project_classes,
):
    """Compatibility facade for mapping one CARLA semantic LiDAR return."""
    return map_carla_truth(
        object_tag,
        object_idx,
        semantic_tag_ids,
        actor_fine_classes,
        supported_project_classes,
    )


def _splitmix64(values):
    """Return stable uint64 hashes for a vector of unsigned integers."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over='ignore'):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (
            (values ^ (values >> np.uint64(30)))
            * np.uint64(0xBF58476D1CE4E5B9)
        )
        values = (
            (values ^ (values >> np.uint64(27)))
            * np.uint64(0x94D049BB133111EB)
        )
        return values ^ (values >> np.uint64(31))


def deterministic_sample_mask(
    point_count,
    sample_rate=1.0,
    seed=0,
    frame_index=0,
    max_points=0,
):
    """Return a stateless, reproducible point mask for streaming datasets."""
    count = int(point_count)
    rate = float(sample_rate)
    limit = int(max_points)
    if count < 0:
        raise ValueError('point_count must be non-negative')
    if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError('sample_rate must lie in [0, 1]')
    if limit < 0:
        raise ValueError('max_points must be non-negative')
    if count == 0:
        return np.zeros(0, dtype=bool)

    indices = np.arange(count, dtype=np.uint64)
    key = (
        np.uint64(int(seed) & ((1 << 64) - 1))
        ^ _splitmix64(np.asarray([
            np.uint64(int(frame_index) & ((1 << 64) - 1)),
        ]))[0]
    )
    hashes = _splitmix64(indices ^ key)
    threshold = int(rate * (1 << 64))
    mask = (
        np.ones(count, dtype=bool)
        if threshold >= (1 << 64)
        else hashes < np.uint64(threshold)
    )
    selected = np.flatnonzero(mask)
    if limit and selected.size > limit:
        order = np.argsort(hashes[selected], kind='stable')[:limit]
        mask[:] = False
        mask[selected[order]] = True
    return mask


def deterministic_sample_indices(
    point_count,
    sample_rate=1.0,
    seed=0,
    frame_index=0,
    max_points=0,
):
    """Return sorted indices selected by :func:`deterministic_sample_mask`."""
    return np.flatnonzero(deterministic_sample_mask(
        point_count,
        sample_rate=sample_rate,
        seed=seed,
        frame_index=frame_index,
        max_points=max_points,
    ))


def assign_bins(values, edges):
    """Assign finite values to left-closed bins with a closed final edge."""
    values = np.asarray(values, dtype=np.float64)
    boundaries = np.asarray(edges, dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError('edges must be a one-dimensional sequence of length 2+')
    if np.any(np.isnan(boundaries)) or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError('edges must be strictly increasing and not NaN')
    result = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    inside = finite & (values >= boundaries[0]) & (values <= boundaries[-1])
    bins = np.searchsorted(boundaries, values[inside], side='right') - 1
    bins[values[inside] == boundaries[-1]] = boundaries.size - 2
    result[inside] = bins
    return result


def binned_binary_statistics(
    values,
    correct,
    reliability,
    edges,
    confidence=None,
    entropy=None,
    eligible=None,
):
    """Summarize correctness and reliability in deterministic numeric bins."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if not (values.size == correct.size == reliability.size):
        raise ValueError('values, correct and reliability must have equal size')
    if confidence is None:
        confidence = np.full(values.size, np.nan)
    if entropy is None:
        entropy = np.full(values.size, np.nan)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    entropy = np.asarray(entropy, dtype=np.float64).reshape(-1)
    if confidence.size != values.size or entropy.size != values.size:
        raise ValueError('confidence and entropy must match values')
    if eligible is None:
        eligible = np.ones(values.size, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if eligible.size != values.size:
        raise ValueError('eligible must match values')

    boundaries = np.asarray(edges, dtype=np.float64)
    assignments = assign_bins(values, boundaries)
    rows = []
    for index in range(boundaries.size - 1):
        in_bin = assignments == index
        valid = (
            in_bin & eligible & np.isfinite(correct)
            & np.isfinite(reliability)
        )
        count = int(np.count_nonzero(valid))
        accuracy = float(np.mean(correct[valid])) if count else None
        mean_reliability = (
            float(np.mean(reliability[valid])) if count else None)

        def finite_mean(data):
            selected = in_bin & eligible & np.isfinite(data)
            return float(np.mean(data[selected])) if np.any(selected) else None

        rows.append({
            'bin_index': index,
            'bin_left': float(boundaries[index]),
            'bin_right': float(boundaries[index + 1]),
            'right_closed': index == boundaries.size - 2,
            'sample_count': count,
            'unsupported_count': int(np.count_nonzero(in_bin & ~eligible)),
            'correct_count': (
                int(np.sum(correct[valid] > 0.5)) if count else 0),
            'accuracy': accuracy,
            'error_rate': 1.0 - accuracy if accuracy is not None else None,
            'mean_raw_value': finite_mean(values),
            'mean_reliability': mean_reliability,
            'mean_pred_confidence': finite_mean(confidence),
            'mean_entropy': finite_mean(entropy),
            'reliability_gap': (
                abs(accuracy - mean_reliability)
                if accuracy is not None else None),
        })
    return rows


def summarize_binned(
    values,
    correct,
    reliability,
    edges,
    confidence=None,
    entropy=None,
    eligible=None,
):
    """Compatibility facade for factor-bin summary generation."""
    return binned_binary_statistics(
        values,
        correct,
        reliability,
        edges,
        confidence=confidence,
        entropy=entropy,
        eligible=eligible,
    )


def expected_calibration_error(confidence, correct, edges=None):
    """Compute top-label ECE after removing non-finite observations."""
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidence.size != correct.size:
        raise ValueError('confidence and correct must have equal size')
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    valid = (
        np.isfinite(confidence) & np.isfinite(correct)
        & (confidence >= 0.0) & (confidence <= 1.0)
    )
    if not np.any(valid):
        return None
    rows = binned_binary_statistics(
        confidence[valid],
        correct[valid],
        confidence[valid],
        edges,
    )
    total = int(np.count_nonzero(valid))
    return float(sum(
        row['sample_count'] / total * row['reliability_gap']
        for row in rows if row['sample_count']
    ))


def multiclass_nll(posterior, gt_class_ids, epsilon=1e-12):
    """Return mean categorical negative log likelihood for valid GT rows."""
    probabilities, labels, valid = _validated_classification_rows(
        posterior, gt_class_ids)
    if not np.any(valid):
        return None
    selected = probabilities[valid, labels[valid]]
    return float(np.mean(-np.log(np.clip(selected, epsilon, 1.0))))


def multiclass_brier(posterior, gt_class_ids):
    """Return unnormalized multiclass Brier score for valid GT rows."""
    probabilities, labels, valid = _validated_classification_rows(
        posterior, gt_class_ids)
    if not np.any(valid):
        return None
    selected = probabilities[valid]
    truth = np.zeros_like(selected)
    truth[np.arange(selected.shape[0]), labels[valid]] = 1.0
    return float(np.mean(np.sum(np.square(selected - truth), axis=1)))


def _validated_classification_rows(posterior, gt_class_ids):
    """Validate a posterior matrix and return its scoreable row mask."""
    probabilities = np.asarray(posterior, dtype=np.float64)
    labels = np.asarray(gt_class_ids, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError('posterior must be N x K and match gt_class_ids')
    if probabilities.shape[1] <= 0:
        raise ValueError('posterior must have at least one class')
    finite = np.all(np.isfinite(probabilities), axis=1)
    nonnegative = np.all(probabilities >= 0.0, axis=1)
    mass = np.sum(probabilities, axis=1)
    valid = (
        finite & nonnegative & (mass > 0.0)
        & (labels >= 0) & (labels < probabilities.shape[1])
    )
    normalized = np.zeros_like(probabilities)
    normalized[valid] = probabilities[valid] / mass[valid, None]
    return normalized, labels, valid


def reliability_gap(reliability, correct, edges=None):
    """Return binned accuracy-versus-reliability gap, not probability ECE."""
    reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if reliability.size != correct.size:
        raise ValueError('reliability and correct must have equal size')
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    valid = (
        np.isfinite(reliability) & np.isfinite(correct)
        & (reliability >= 0.0) & (reliability <= 1.0)
    )
    if not np.any(valid):
        return None
    rows = binned_binary_statistics(
        reliability[valid], correct[valid], reliability[valid], edges)
    total = int(np.count_nonzero(valid))
    return float(sum(
        row['sample_count'] / total * row['reliability_gap']
        for row in rows if row['sample_count']
    ))


def binary_auroc(scores, correct):
    """Compute tie-aware AUROC for reliability as a correctness score."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(correct, dtype=np.float64).reshape(-1)
    if scores.size != labels.size:
        raise ValueError('scores and correct must have equal size')
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid] > 0.5
    positive = int(np.count_nonzero(labels))
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None

    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while (
            stop < scores.size
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[labels]))
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def area_under_risk_coverage(scores, correct):
    """Return discrete AURC when retaining points from high to low score."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(correct, dtype=np.float64).reshape(-1)
    if scores.size != labels.size:
        raise ValueError('scores and correct must have equal size')
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid]
    if scores.size == 0:
        return None
    order = np.argsort(-scores, kind='stable')
    cumulative_errors = np.cumsum(labels[order] <= 0.5)
    risks = cumulative_errors / np.arange(1, scores.size + 1)
    return float(np.mean(risks))


def calibration_metrics(
    posterior,
    gt_class_ids,
    eligible=None,
    ece_edges=None,
):
    """Return primary confidence-calibration metrics for supported GT rows."""
    probabilities, labels, valid = _validated_classification_rows(
        posterior, gt_class_ids)
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool).reshape(-1)
        if eligible.size != valid.size:
            raise ValueError('eligible must match posterior rows')
        valid &= eligible
    if not np.any(valid):
        return {
            'sample_count': 0,
            'ece': None,
            'nll': None,
            'brier': None,
        }
    selected = probabilities[valid]
    selected_labels = labels[valid]
    confidence = np.max(selected, axis=1)
    prediction = np.argmax(selected, axis=1)
    correct = prediction == selected_labels
    return {
        'sample_count': int(selected.shape[0]),
        'ece': expected_calibration_error(
            confidence, correct, edges=ece_edges),
        'nll': multiclass_nll(selected, selected_labels),
        'brier': multiclass_brier(selected, selected_labels),
    }


def _json_compatible(value):
    """Convert common scientific Python values into strict JSON values."""
    if dataclasses.is_dataclass(value):
        return _json_compatible(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def strict_json_dumps(value, **kwargs):
    """Serialize JSON while rejecting NaN and infinity."""
    options = {
        'allow_nan': False,
        'ensure_ascii': False,
        'indent': 2,
    }
    options.update(kwargs)
    options['allow_nan'] = False
    return json.dumps(_json_compatible(value), **options)


def write_strict_json(path, value, **kwargs):
    """Write strict UTF-8 JSON and return the resolved output path."""
    output = Path(path).expanduser().resolve()
    output.write_text(
        strict_json_dumps(value, **kwargs) + '\n',
        encoding='utf-8',
    )
    return output
