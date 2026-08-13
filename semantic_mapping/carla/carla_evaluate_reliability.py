#!/usr/bin/env python3
"""Evaluate GA-BSVM reliability factors with synchronized CARLA ground truth."""

import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import platform
from importlib import metadata as importlib_metadata

import numpy as np
from PIL import Image
import yaml

from semantic_mapping.carla.carla_benchmark import camera_intrinsics
from semantic_mapping.carla.carla_evaluate_benchmark import SegformerEvaluator
from semantic_mapping.carla.carla_reliability import (
    FACTOR_SUMMARY_COLUMNS,
    PER_POINT_COLUMNS,
    area_under_risk_coverage,
    assign_bins,
    binary_auroc,
    expected_calibration_error,
    map_carla_ground_truth,
    match_semantic_lidar_points,
    deterministic_sample_mask,
    reliability_gap,
    select_historical_frame,
    summarize_binned,
    write_strict_json,
)
from semantic_mapping.runtime.reliability_factors import (
    combine_reliability,
    compute_density_reliability,
    compute_local_point_density,
    compute_motion_reliability,
    compute_normalized_view_radius,
    compute_range_reliability,
    compute_semantic_reliability,
    compute_view_reliability,
)
from semantic_mapping.carla.reliability_voxel_ablation import (
    ABLATION_NAMES,
    VoxelAblation,
    ablation_weights,
)
from semantic_mapping.runtime.segformer_core import find_supported_project_classes
from semantic_mapping.runtime.semantic_posterior import (
    normalize_posterior,
    posterior_probabilities_to_logits,
    sample_posterior_bilinear,
)
from semantic_mapping.runtime.semantic_projection import project_points_pinhole
from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES
from semantic_mapping.runtime.segformer_training import checkpoint_integrity


RELIABILITY_SCHEMA_VERSION = 1
CARLA_TO_OPTICAL = np.asarray([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

REQUIRED_RELIABILITY_PARAMETERS = (
    'imu_window_sec',
    'motion_angular_scale',
    'motion_accel_scale',
    'motion_min_reliability',
    'motion_missing_reliability',
    'imu_gravity',
    'density_scale',
    'range_scale_m',
    'view_edge_penalty',
    'semantic_confidence_floor',
    'voxel_size',
    'evidence_prior',
    'evidence_strength',
    'evidence_decay',
    'evidence_decay_reference_sec',
    'max_frame_evidence',
    'max_total_evidence',
    'max_observation_weight',
)

OPTIONAL_RELIABILITY_PARAMETERS = {
    'num_classes': len(DEFAULT_CLASSES),
    'uncertainty_entropy_weight': 0.7,
    'dynamic_voxel_ttl_sec': 0.0,
    'voxel_ttl_sec': 0.0,
}


def carla_lidar_to_optical_transform(lidar_to_world, camera_to_world):
    """Return the historical LiDAR-to-camera-optical homogeneous transform."""
    lidar_to_world = np.asarray(lidar_to_world, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if lidar_to_world.shape != (4, 4) or camera_to_world.shape != (4, 4):
        raise ValueError('CARLA sensor transforms must both have shape (4, 4)')
    if (
        not np.all(np.isfinite(lidar_to_world))
        or not np.all(np.isfinite(camera_to_world))
    ):
        raise ValueError('CARLA sensor transforms must be finite')
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError as exc:
        raise ValueError('camera_to_world must be invertible') from exc
    return CARLA_TO_OPTICAL @ world_to_camera @ lidar_to_world


def load_reliability_parameters(path):
    """Load the exact GA-BSVM factor and VoxelMap settings from one ROS YAML."""
    path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot read reliability parameter file {path}: {exc}')
    if not isinstance(document, dict):
        raise ValueError('Reliability parameter YAML must contain a mapping')
    node = document.get('ga_bsvm_node')
    if not isinstance(node, dict):
        raise ValueError('Reliability parameter YAML is missing ga_bsvm_node')
    values = node.get('ros__parameters')
    if not isinstance(values, dict):
        raise ValueError(
            'Reliability parameter YAML is missing '
            'ga_bsvm_node.ros__parameters')
    missing = [
        name for name in REQUIRED_RELIABILITY_PARAMETERS
        if name not in values
    ]
    if missing:
        raise ValueError(
            'Reliability parameter YAML is missing required fields: '
            + ', '.join(missing))

    result = {}
    for name in REQUIRED_RELIABILITY_PARAMETERS:
        try:
            result[name] = float(values[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Reliability parameter {name} must be numeric') from exc
    for name, default in OPTIONAL_RELIABILITY_PARAMETERS.items():
        value = values.get(name, default)
        try:
            result[name] = int(value) if name == 'num_classes' else float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Reliability parameter {name} must be numeric') from exc

    if result['num_classes'] != len(DEFAULT_CLASSES):
        raise ValueError(
            f'Benchmark ontology has {len(DEFAULT_CLASSES)} classes but '
            f'num_classes={result["num_classes"]}')
    if result['max_observation_weight'] <= 0.0:
        result['max_observation_weight'] = None
    return result


def _sample_field(sample, name):
    if isinstance(sample, dict):
        return sample[name]
    return getattr(sample, name)


def select_imu_motion(imu_samples, lidar_timestamp, params):
    """Apply the runtime's closed, plus/minus IMU window around a LiDAR time."""
    lidar_timestamp = float(lidar_timestamp)
    half_window = float(params['imu_window_sec'])
    angular_norms = []
    acceleration_norms = []
    for sample in imu_samples:
        try:
            timestamp = float(_sample_field(sample, 'timestamp_sec'))
            acceleration = np.asarray(
                _sample_field(sample, 'accelerometer_mps2'),
                dtype=np.float64,
            ).reshape(3)
            angular = np.asarray(
                _sample_field(sample, 'gyroscope_radps'),
                dtype=np.float64,
            ).reshape(3)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if (
            not np.isfinite(timestamp)
            or not np.all(np.isfinite(acceleration))
            or not np.all(np.isfinite(angular))
            or abs(timestamp - lidar_timestamp) > half_window + 1e-12
        ):
            continue
        acceleration_norms.append(float(np.linalg.norm(acceleration)))
        angular_norms.append(float(np.linalg.norm(angular)))

    if not angular_norms:
        return {
            'status': 'missing',
            'reliability': float(params['motion_missing_reliability']),
            'angular_rms': 0.0,
            'acceleration_deviation': 0.0,
            'sample_count': 0,
        }
    reliability, angular_rms, acceleration_deviation = (
        compute_motion_reliability(
            angular_norms,
            acceleration_norms,
            params['motion_angular_scale'],
            params['motion_accel_scale'],
            params['imu_gravity'],
            params['motion_min_reliability'],
        )
    )
    return {
        'status': 'ok',
        'reliability': float(reliability),
        'angular_rms': float(angular_rms),
        'acceleration_deviation': float(acceleration_deviation),
        'sample_count': len(angular_norms),
    }


def validate_reliability_manifest(manifest):
    """Reject image-benchmark or incomplete metadata before model inference."""
    if not isinstance(manifest, dict):
        raise ValueError('Reliability manifest must be a JSON object')
    if manifest.get('format') != 'carla_reliability_v1':
        raise ValueError(
            'Dataset format is not carla_reliability_v1; the image benchmark '
            'cannot be evaluated by this tool')
    if int(manifest.get('format_version', -1)) != 1:
        raise ValueError('Unsupported CARLA reliability format_version')
    required = (
        'frames', 'rgb_index', 'semantic_tag_ids', 'sensors', 'imu_csv')
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(
            'Reliability manifest is missing required fields: '
            + ', '.join(missing))
    if not isinstance(manifest['frames'], list):
        raise ValueError('Reliability manifest frames must be a list')
    if not manifest['frames']:
        raise ValueError('Reliability manifest frames list is empty')
    if not isinstance(manifest['semantic_tag_ids'], dict):
        raise ValueError('semantic_tag_ids must be a mapping')
    sensors = manifest['sensors']
    if not isinstance(sensors, dict):
        raise ValueError('sensors must be a mapping')
    sensor_names = set(sensors)
    needed_sensors = {
        'rgb',
        'lidar',
        'semantic_lidar',
        'semantic_camera',
        'instance_camera',
        'imu',
    }
    if not needed_sensors.issubset(sensor_names):
        raise ValueError(
            'Reliability manifest sensors are missing: '
            + ', '.join(sorted(needed_sensors - sensor_names)))
    return manifest


def _concatenate_chunks(chunks, name, dtype=np.float64):
    values = [np.asarray(chunk[name]).reshape(-1) for chunk in chunks]
    if not values:
        return np.empty(0, dtype=dtype)
    return np.concatenate(values).astype(dtype, copy=False)


def _summary_bin_rows(
    factor,
    axis,
    values,
    reliability,
    edges,
    correct,
    supported,
    confidence,
    entropy,
    nll,
    brier,
    stratum='all',
):
    base_rows = summarize_binned(
        values,
        correct,
        reliability,
        edges,
        confidence=confidence,
        entropy=entropy,
        eligible=supported,
    )
    assignments = assign_bins(values, edges)
    output = []
    for row in base_rows:
        in_bin = assignments == int(row['bin_index'])
        eligible = in_bin & supported & np.isfinite(correct)
        nll_mask = eligible & np.isfinite(nll)
        brier_mask = eligible & np.isfinite(brier)
        row.update({
            'factor': factor,
            'axis': axis,
            'stratum': stratum,
            'nll': float(np.mean(nll[nll_mask])) if np.any(nll_mask) else None,
            'brier': (
                float(np.mean(brier[brier_mask]))
                if np.any(brier_mask) else None),
        })
        output.append({name: row.get(name) for name in FACTOR_SUMMARY_COLUMNS})
    return output


def build_factor_summary(chunks):
    """Build all five single-factor and combined correctness relationships."""
    if not chunks:
        return []
    correct = _concatenate_chunks(chunks, 'correct')
    supported = _concatenate_chunks(chunks, 'gt_supported', bool)
    confidence = _concatenate_chunks(chunks, 'pred_confidence')
    entropy = _concatenate_chunks(chunks, 'semantic_entropy')
    nll = _concatenate_chunks(chunks, 'nll')
    brier = _concatenate_chunks(chunks, 'brier')
    offsets = _concatenate_chunks(chunks, 'requested_offset_ms')
    reliability_edges = np.linspace(0.0, 1.0, 11)
    definitions = (
        ('semantic', 'reliability', 'r_semantic', 'r_semantic',
         reliability_edges),
        ('semantic', 'entropy', 'semantic_entropy', 'r_semantic',
         np.linspace(0.0, np.log(len(DEFAULT_CLASSES)), 11)),
        ('range', 'reliability', 'r_range', 'r_range', reliability_edges),
        ('range', 'range_m', 'range_m', 'r_range',
         np.asarray([0, 5, 10, 15, 20, 30, 40, np.inf])),
        ('density', 'reliability', 'r_density', 'r_density',
         reliability_edges),
        ('density', 'local_density', 'local_density', 'r_density',
         np.asarray([0, 2, 4, 8, 16, 32, 64, np.inf])),
        ('view', 'reliability', 'r_view', 'r_view', reliability_edges),
        ('view', 'view_radius', 'view_radius', 'r_view',
         np.linspace(0.0, 1.0, 6)),
        ('motion', 'reliability', 'r_motion', 'r_motion',
         reliability_edges),
        ('motion', 'angular_rms', 'angular_rms', 'r_motion',
         np.asarray([0, 0.25, 0.5, 1, 2, 4, np.inf])),
        ('motion', 'accel_deviation', 'accel_deviation', 'r_motion',
         np.asarray([0, 0.25, 0.5, 1, 2, 4, 8, np.inf])),
        ('motion', 'time_offset_ms', 'requested_offset_ms', 'r_motion',
         np.asarray([-0.5, 10, 35, 75, 125, np.inf])),
        ('combined', 'reliability', 'r_combined', 'r_combined',
         reliability_edges),
    )
    rows = []
    for factor, axis, value_name, reliability_name, edges in definitions:
        rows.extend(_summary_bin_rows(
            factor,
            axis,
            _concatenate_chunks(chunks, value_name),
            _concatenate_chunks(chunks, reliability_name),
            edges,
            correct,
            supported,
            confidence,
            entropy,
            nll,
            brier,
        ))

    # Motion and time offset are deliberately reported jointly.  Offset does
    # not enter r_motion; these strata reveal whether that omission matters.
    motion = _concatenate_chunks(chunks, 'r_motion')
    for offset in sorted(np.unique(offsets[np.isfinite(offsets)])):
        selected = np.isclose(offsets, offset, rtol=0.0, atol=1e-9)
        rows.extend(_summary_bin_rows(
            'motion',
            'reliability',
            motion[selected],
            motion[selected],
            reliability_edges,
            correct[selected],
            supported[selected],
            confidence[selected],
            entropy[selected],
            nll[selected],
            brier[selected],
            stratum=f'offset_{offset:g}ms',
        ))
    return rows


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sensor_matrix(record):
    try:
        matrix = np.asarray(record['transform_matrix'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Sensor record has no valid transform_matrix') from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('Sensor transform_matrix must be finite 4 x 4')
    return matrix


def _transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _load_imu_samples(path):
    samples = []
    with Path(path).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            try:
                samples.append({
                    'timestamp_sec': float(row['timestamp_sec']),
                    'accelerometer_mps2': [
                        float(row['accel_x_mps2']),
                        float(row['accel_y_mps2']),
                        float(row['accel_z_mps2']),
                    ],
                    'gyroscope_radps': [
                        float(row['gyro_x_radps']),
                        float(row['gyro_y_radps']),
                        float(row['gyro_z_radps']),
                    ],
                })
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name):
    """Return an installed distribution version without failing evaluation."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _model_provenance(model_id, evaluator):
    """Return resolved revision and local-checkpoint provenance when possible."""
    model_path = Path(str(model_id)).expanduser()
    local_integrity = None
    if model_path.is_dir():
        try:
            integrity = checkpoint_integrity(model_path)
            local_integrity = {
                'algorithm': integrity['algorithm'],
                'aggregate_sha256': integrity['aggregate_sha256'],
                'file_count': len(integrity['files']),
            }
        except (OSError, ValueError) as exc:
            local_integrity = {'error': str(exc)}
    config = getattr(getattr(evaluator, 'model', None), 'config', None)
    return {
        'requested_revision': getattr(evaluator, 'requested_revision', None),
        'resolved_revision': getattr(config, '_commit_hash', None),
        'local_checkpoint_integrity': local_integrity,
    }


def _software_provenance(evaluator):
    """Return software and CUDA identifiers needed to reproduce inference."""
    torch = getattr(evaluator, 'torch', None)
    cuda_name = None
    cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
    try:
        if torch is not None and torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        cuda_name = None
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'scipy': _package_version('scipy'),
        'pillow': _package_version('Pillow'),
        'torch': getattr(torch, '__version__', None),
        'transformers': _package_version('transformers'),
        'cuda_runtime': cuda_version,
        'cuda_device': cuda_name,
    }


def git_provenance(repo):
    """Return best-effort commit and dirty-state metadata without mutation."""
    repo = Path(repo).resolve()

    def command(*arguments):
        try:
            return subprocess.run(
                ['git', *arguments],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = command('rev-parse', 'HEAD')
    status = command('status', '--porcelain')
    return {
        'commit': commit,
        'dirty': bool(status) if status is not None else None,
    }


class _PosteriorCache:
    """Bounded in-memory cache so nearby offsets share one model inference."""

    def __init__(self, evaluator, dataset, max_items=32):
        self.evaluator = evaluator
        self.dataset = Path(dataset)
        self.max_items = max(int(max_items), 1)
        self.values = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, record):
        relative = str(record['path'])
        if relative in self.values:
            self.hit_count += 1
            value = self.values.pop(relative)
            self.values[relative] = value
            return value
        self.miss_count += 1
        image_path = self.dataset / relative
        with Image.open(image_path) as source:
            image = source.convert('RGB')
            width, height = image.size
            posterior = self.evaluator.infer_project_posterior(image)
            # The online node publishes the native project posterior as a
            # 16FC<K> Image and GA-BSVM decodes it back to float32 before
            # sampling.  Reproduce that transport boundary here, including the
            # receiver-side normalization, instead of evaluating ideal float32
            # probabilities that the runtime never observes.
            posterior = normalize_posterior(
                posterior.astype(np.float16).astype(np.float32))
        value = (posterior, int(width), int(height))
        self.values[relative] = value
        while len(self.values) > self.max_items:
            self.values.popitem(last=False)
        return value


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_summary_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FACTOR_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name))
                             for name in FACTOR_SUMMARY_COLUMNS})


def _offset_token(offset_ms):
    return f'{float(offset_ms):09.3f}'.replace('-', 'm').replace('.', 'p')


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else None


def _factor_relationship(scores, correct, supported):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    supported = np.asarray(supported, dtype=bool)
    valid = supported & np.isfinite(scores) & np.isfinite(correct)
    if not np.any(valid):
        return {
            'sample_count': 0,
            'reliability_gap': None,
            'correctness_auroc': None,
            'aurc': None,
        }
    return {
        'sample_count': int(np.count_nonzero(valid)),
        'reliability_gap': reliability_gap(scores[valid], correct[valid]),
        'correctness_auroc': binary_auroc(scores[valid], correct[valid]),
        'aurc': area_under_risk_coverage(scores[valid], correct[valid]),
    }


def _weighted_accuracy(weight, correct, supported):
    weight = np.asarray(weight, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    valid = np.asarray(supported, dtype=bool)
    valid &= np.isfinite(weight) & np.isfinite(correct) & (weight >= 0.0)
    if not np.any(valid) or float(np.sum(weight[valid])) <= 1e-12:
        return {
            'sample_count': int(np.count_nonzero(valid)),
            'weight_sum': 0.0,
            'weighted_accuracy_diagnostic': None,
            'effective_sample_size': None,
        }
    selected_weight = weight[valid]
    weight_sum = float(np.sum(selected_weight))
    return {
        'sample_count': int(np.count_nonzero(valid)),
        'weight_sum': weight_sum,
        'weighted_accuracy_diagnostic': float(
            np.sum(selected_weight * correct[valid]) / weight_sum),
        'effective_sample_size': float(
            weight_sum ** 2 / max(np.sum(selected_weight ** 2), 1e-12)),
    }


def _aggregate_alignment(frame_reports, distance_sample):
    total_normal = sum(item['normal_count'] for item in frame_reports)
    total_semantic = sum(item['semantic_count'] for item in frame_reports)
    total_matched = sum(item['matched_count'] for item in frame_reports)
    accepted = [item for item in frame_reports if item['accepted']]
    distances = np.asarray(distance_sample, dtype=np.float64)
    return {
        'frame_count': len(frame_reports),
        'accepted_frame_count': len(accepted),
        'rejected_frame_count': len(frame_reports) - len(accepted),
        'total_normal_points': int(total_normal),
        'total_semantic_points': int(total_semantic),
        'total_matched_points': int(total_matched),
        'aggregate_matched_fraction': float(
            total_matched / max(total_normal, total_semantic, 1)),
        'distance_sample_count': int(distances.size),
        'mean_match_distance_m': (
            float(np.mean(distances)) if distances.size else None),
        'p95_match_distance_m': (
            float(np.percentile(distances, 95)) if distances.size else None),
        'max_match_distance_m': (
            float(np.max(distances)) if distances.size else None),
        'rejected_frames': [
            int(item['frame']) for item in frame_reports
            if not item['accepted']
        ],
    }


def _append_distance_sample(storage, distances, limit=200000):
    remaining = int(limit) - len(storage)
    if remaining <= 0:
        return
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    if len(distances) > remaining:
        positions = np.linspace(
            0, len(distances) - 1, remaining, dtype=np.int64)
        distances = distances[positions]
    storage.extend(float(value) for value in distances)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate runtime GA-BSVM reliability factors against CARLA '
            'Semantic LiDAR ground truth.'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--params-file', required=True)
    parser.add_argument(
        '--segformer-model',
        default='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    )
    parser.add_argument(
        '--segformer-revision', default='',
        help='Optional Hugging Face revision/commit for reproducible loading.')
    parser.add_argument('--segformer-temperature', type=float, default=1.0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        default='auto')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--time-offset-ms', type=float, nargs='+',
                        default=[0.0, 20.0, 50.0, 100.0, 150.0])
    parser.add_argument('--frame-tolerance-ms', type=float, default=6.0)
    parser.add_argument('--semantic-match-tolerance-m', type=float,
                        default=0.02)
    parser.add_argument('--min-match-fraction', type=float, default=0.98)
    parser.add_argument('--point-sample-rate', type=float, default=1.0)
    parser.add_argument('--max-points-per-frame', type=int, default=1000)
    parser.add_argument('--posterior-storage', choices=('npz', 'none'),
                        default='npz')
    parser.add_argument('--posterior-cache-size', type=int, default=32)
    parser.add_argument('--voxel-eval', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--limit', type=int, default=0)
    return parser


def _validate_arguments(args):
    offsets = np.asarray(args.time_offset_ms, dtype=np.float64)
    if offsets.size == 0 or not np.all(np.isfinite(offsets)):
        raise ValueError('time-offset-ms must contain finite values')
    if np.any(offsets < 0.0):
        raise ValueError('time-offset-ms must be non-negative')
    if len(np.unique(offsets)) != len(offsets):
        raise ValueError(
            'time-offset-ms must not contain duplicate values; duplicate '
            'offsets would overwrite posterior artifacts')
    if args.frame_tolerance_ms < 0.0:
        raise ValueError('frame-tolerance-ms must be non-negative')
    if args.semantic_match_tolerance_m < 0.0:
        raise ValueError('semantic-match-tolerance-m must be non-negative')
    if not 0.0 <= args.min_match_fraction <= 1.0:
        raise ValueError('min-match-fraction must lie in [0, 1]')
    if not 0.0 <= args.point_sample_rate <= 1.0:
        raise ValueError('point-sample-rate must lie in [0, 1]')
    if args.max_points_per_frame < 0:
        raise ValueError('max-points-per-frame must be non-negative')
    if args.posterior_cache_size <= 0:
        raise ValueError('posterior-cache-size must be positive')
    if args.limit < 0:
        raise ValueError('limit must be non-negative')


def evaluate(args):
    """Run the complete point-level and optional short-sequence evaluation."""
    _validate_arguments(args)
    started = time.monotonic()
    dataset = Path(args.dataset).expanduser().resolve()
    manifest = validate_reliability_manifest(
        _read_json(dataset / 'manifest.json'))
    params_path = Path(args.params_file).expanduser().resolve()
    params = load_reliability_parameters(params_path)
    frame_paths = list(manifest['frames'])
    if args.limit > 0:
        frame_paths = frame_paths[:args.limit]
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dataset / 'results' / ('reliability_' + time.strftime(
            '%Y%m%d_%H%M%S'))
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f'Output directory is not empty: {output}. Use a new path.')
    output.mkdir(parents=True, exist_ok=True)
    posterior_directory = output / 'posterior'
    if args.posterior_storage == 'npz':
        posterior_directory.mkdir(exist_ok=True)

    rgb_index = _read_json(dataset / manifest['rgb_index'])
    if not isinstance(rgb_index, list) or not rgb_index:
        raise ValueError('rgb_index must contain historical camera records')
    imu_samples = _load_imu_samples(dataset / manifest['imu_csv'])
    actor_map = manifest.get('actor_id_map', manifest.get('vehicle_actors', {}))
    semantic_tag_ids = manifest['semantic_tag_ids']

    segformer = SegformerEvaluator(
        args.segformer_model,
        args.device,
        confidence_threshold=0.0,
        use_fp16=args.fp16,
        posterior_temperature=args.segformer_temperature,
        revision=args.segformer_revision or None,
    )
    supported_classes = tuple(
        find_supported_project_classes(segformer.id2label))
    cache = _PosteriorCache(
        segformer, dataset, max_items=args.posterior_cache_size)

    offsets = [float(value) for value in args.time_offset_ms]
    voxel_evaluators = (
        {
            offset: VoxelAblation(params, class_count=len(DEFAULT_CLASSES))
            for offset in offsets
        }
        if args.voxel_eval else {}
    )
    alignment_reports = []
    distance_sample = []
    offset_flow = {
        offset: Counter() for offset in offsets
    }
    global_flow = Counter()
    unsupported_exported = Counter()
    unsupported_all_projected = Counter()
    gt_source_exported = Counter()
    summary_chunks = []

    per_point_path = output / 'per_point.csv'
    with per_point_path.open('w', newline='', encoding='utf-8') as csv_handle:
        point_writer = csv.DictWriter(
            csv_handle, fieldnames=PER_POINT_COLUMNS)
        point_writer.writeheader()

        for frame_index, relative_metadata in enumerate(frame_paths):
            metadata = _read_json(dataset / relative_metadata)
            lidar_frame = int(metadata['lidar']['frame'])
            lidar_timestamp = float(metadata['lidar']['timestamp_sec'])
            with np.load(dataset / metadata['lidar']['path']) as normal_data:
                normal_xyz = np.asarray(
                    normal_data['xyz'], dtype=np.float64).copy()
                intensity = np.asarray(
                    normal_data['intensity'], dtype=np.float32).copy()
                channels = np.asarray(
                    normal_data['channel'], dtype=np.int64).copy()
            with np.load(
                dataset / metadata['semantic_lidar']['path']
            ) as semantic_data:
                semantic_xyz = np.asarray(
                    semantic_data['xyz'], dtype=np.float64).copy()
                semantic_object_idx = np.asarray(
                    semantic_data['object_idx'], dtype=np.uint32).copy()
                semantic_object_tag = np.asarray(
                    semantic_data['object_tag'], dtype=np.uint32).copy()
            if normal_xyz.ndim != 2 or normal_xyz.shape[1] != 3:
                raise ValueError(
                    f'Normal LiDAR frame {lidar_frame} has invalid xyz shape')
            if semantic_xyz.ndim != 2 or semantic_xyz.shape[1] != 3:
                raise ValueError(
                    f'Semantic LiDAR frame {lidar_frame} has invalid xyz shape')

            normal_to_world = _sensor_matrix(metadata['lidar'])
            semantic_to_world = _sensor_matrix(metadata['semantic_lidar'])
            semantic_to_normal = (
                np.linalg.inv(normal_to_world) @ semantic_to_world)
            semantic_in_normal = _transform_points(
                semantic_xyz, semantic_to_normal)
            matches = match_semantic_lidar_points(
                normal_xyz,
                semantic_in_normal,
                args.semantic_match_tolerance_m,
            )
            match_statistics = matches.statistics()
            accepted_alignment = (
                matches.matched_fraction >= args.min_match_fraction)
            match_statistics.update({
                'frame': lidar_frame,
                'accepted': bool(accepted_alignment),
            })
            alignment_reports.append(match_statistics)
            _append_distance_sample(distance_sample, matches.distances_m)
            global_flow['normal_points'] += len(normal_xyz)
            global_flow['semantic_points'] += len(semantic_xyz)
            global_flow['matched_points'] += matches.matched_count
            if not accepted_alignment:
                global_flow['alignment_rejected_frames'] += 1
                continue

            normal_to_semantic = np.full(len(normal_xyz), -1, dtype=np.int64)
            normal_match_distance = np.full(
                len(normal_xyz), np.nan, dtype=np.float64)
            normal_to_semantic[matches.normal_indices] = (
                matches.semantic_indices)
            normal_match_distance[matches.normal_indices] = (
                matches.distances_m)
            motion = select_imu_motion(
                imu_samples, lidar_timestamp, params)
            global_flow[f'imu_{motion["status"]}_frames'] += 1

            if len(intensity) != len(normal_xyz) or len(channels) != len(
                normal_xyz
            ):
                raise ValueError(
                    f'LiDAR auxiliary arrays mismatch frame {lidar_frame}')

            for offset_index, offset in enumerate(offsets):
                flow = offset_flow[offset]
                flow['lidar_frames'] += 1
                selected_rgb = select_historical_frame(
                    rgb_index,
                    lidar_timestamp,
                    offset,
                    args.frame_tolerance_ms,
                )
                if not selected_rgb.accepted:
                    flow[f'rgb_rejected_{selected_rgb.reason}'] += 1
                    continue
                rgb_record = selected_rgb.frame
                posterior_grid, width, height = cache.get(rgb_record)
                projection_transform = carla_lidar_to_optical_transform(
                    normal_to_world,
                    _sensor_matrix(rgb_record),
                )
                camera_matrix = camera_intrinsics(
                    width,
                    height,
                    float(manifest['sensors']['rgb']['attributes']['fov']),
                )
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    valid_mask, pixel_u, pixel_v = project_points_pinhole(
                        normal_xyz,
                        projection_transform,
                        camera_matrix,
                        image_height=height,
                        image_width=width,
                    )
                normal_indices = np.flatnonzero(valid_mask)
                flow['projected_normal_points'] += len(normal_indices)
                global_flow['projected_normal_points'] += len(normal_indices)
                if not len(normal_indices):
                    flow['empty_projection_frames'] += 1
                    continue

                valid_points = normal_xyz[normal_indices]
                valid_u = pixel_u[normal_indices]
                valid_v = pixel_v[normal_indices]
                sampled_posterior = sample_posterior_bilinear(
                    posterior_grid,
                    valid_u,
                    valid_v,
                    width,
                    height,
                )
                logits = posterior_probabilities_to_logits(
                    sampled_posterior)
                probabilities, semantic_entropy, r_semantic = (
                    compute_semantic_reliability(
                        logits,
                        len(DEFAULT_CLASSES),
                        params['semantic_confidence_floor'],
                    )
                )
                local_density = compute_local_point_density(valid_points, 0.3)
                sensor_range = np.linalg.norm(valid_points, axis=1)
                view_radius = compute_normalized_view_radius(
                    valid_u, valid_v, width, height)
                r_density = compute_density_reliability(
                    local_density, params['density_scale'])
                r_range = compute_range_reliability(
                    sensor_range, params['range_scale_m'])
                r_view = compute_view_reliability(
                    view_radius, params['view_edge_penalty'])
                r_motion = np.full(
                    len(valid_points),
                    motion['reliability'],
                    dtype=np.float32,
                )
                r_combined = combine_reliability(
                    r_motion, r_density, r_range, r_view, r_semantic)
                weights = ablation_weights(
                    r_semantic, r_range, r_density, r_view, r_motion)
                world_points = _transform_points(
                    valid_points, normal_to_world).astype(np.float32)

                semantic_indices = normal_to_semantic[normal_indices]
                matched_projected = semantic_indices >= 0
                flow['matched_projected_points'] += int(
                    np.count_nonzero(matched_projected))
                global_flow['matched_projected_points'] += int(
                    np.count_nonzero(matched_projected))
                gt_ids = np.full(len(valid_points), -1, dtype=np.int64)
                gt_supported = np.zeros(len(valid_points), dtype=bool)
                gt_names = np.full(len(valid_points), 'unmatched', dtype=object)
                gt_sources = np.full(
                    len(valid_points), 'unmatched', dtype=object)
                actor_ids = np.zeros(len(valid_points), dtype=np.uint32)
                carla_tags = np.zeros(len(valid_points), dtype=np.uint32)
                truth_cache = {}
                for local_index in np.flatnonzero(matched_projected):
                    semantic_index = int(semantic_indices[local_index])
                    actor_id = int(semantic_object_idx[semantic_index])
                    carla_tag = int(semantic_object_tag[semantic_index])
                    cache_key = (carla_tag, actor_id)
                    truth = truth_cache.get(cache_key)
                    if truth is None:
                        truth = map_carla_ground_truth(
                            carla_tag,
                            actor_id,
                            semantic_tag_ids,
                            actor_map,
                            supported_classes,
                        )
                        truth_cache[cache_key] = truth
                    actor_ids[local_index] = actor_id
                    carla_tags[local_index] = carla_tag
                    gt_ids[local_index] = truth.project_id
                    gt_supported[local_index] = truth.supported
                    gt_names[local_index] = truth.class_name
                    gt_sources[local_index] = truth.source
                    if not truth.supported:
                        unsupported_all_projected[truth.class_name] += 1

                if args.voxel_eval:
                    voxel_gt = np.where(gt_supported, gt_ids, -1)
                    voxel_evaluators[offset].update(
                        world_points,
                        logits,
                        voxel_gt,
                        weights,
                        timestamp_sec=lidar_timestamp,
                    )

                candidate_local = np.flatnonzero(matched_projected)
                if not len(candidate_local):
                    flow['no_gt_projection_frames'] += 1
                    continue
                # Hash all original LiDAR point indices once with a frame-stable
                # key.  The sample-rate decision and global per-frame cap are
                # therefore independent of RGB offset.  Intersecting this fixed
                # set with each offset's visible GT candidates makes every
                # common point a paired observation; an offset can export fewer
                # than the cap when many globally selected points are invisible.
                all_point_mask = deterministic_sample_mask(
                    len(normal_xyz),
                    sample_rate=args.point_sample_rate,
                    seed=int(manifest.get('seed', 0)),
                    frame_index=lidar_frame,
                    max_points=args.max_points_per_frame,
                )
                sampled_candidate = np.flatnonzero(
                    all_point_mask[normal_indices[candidate_local]])
                selected_local = candidate_local[sampled_candidate]
                flow['exported_points'] += len(selected_local)
                global_flow['exported_points'] += len(selected_local)
                if not len(selected_local):
                    continue

                selected_probabilities = probabilities[selected_local]
                selected_gt_ids = gt_ids[selected_local]
                selected_supported = gt_supported[selected_local]
                predictions = np.argmax(
                    selected_probabilities, axis=1).astype(np.int64)
                pred_confidence = np.max(selected_probabilities, axis=1)
                correct = np.full(len(selected_local), np.nan, dtype=np.float64)
                correct[selected_supported] = (
                    predictions[selected_supported]
                    == selected_gt_ids[selected_supported]
                ).astype(np.float64)
                gt_probability = np.full(
                    len(selected_local), np.nan, dtype=np.float64)
                gt_probability[selected_supported] = selected_probabilities[
                    selected_supported,
                    selected_gt_ids[selected_supported],
                ]
                nll = np.full(len(selected_local), np.nan, dtype=np.float64)
                nll[selected_supported] = -np.log(np.clip(
                    gt_probability[selected_supported], 1e-12, 1.0))
                brier = np.full(len(selected_local), np.nan, dtype=np.float64)
                if np.any(selected_supported):
                    selected_truth = np.zeros_like(selected_probabilities)
                    supported_rows = np.flatnonzero(selected_supported)
                    selected_truth[
                        supported_rows,
                        selected_gt_ids[selected_supported],
                    ] = 1.0
                    brier[selected_supported] = np.sum(np.square(
                        selected_probabilities[selected_supported]
                        - selected_truth[selected_supported]), axis=1)

                posterior_relative = ''
                if args.posterior_storage == 'npz':
                    posterior_name = (
                        f'{lidar_frame:08d}_offset_'
                        f'{_offset_token(offset)}.npz')
                    posterior_path = posterior_directory / posterior_name
                    np.savez_compressed(
                        posterior_path,
                        posterior=selected_probabilities.astype(np.float16),
                        class_names=np.asarray(DEFAULT_CLASSES, dtype='U32'),
                        point_indices=normal_indices[selected_local].astype(
                            np.int64),
                        gt_project_ids=selected_gt_ids.astype(np.int16),
                        source_image_size=np.asarray(
                            [width, height], dtype=np.int32),
                    )
                    posterior_relative = posterior_path.relative_to(
                        output).as_posix()

                selected_weights = {
                    name: weights[name][selected_local]
                    for name in ABLATION_NAMES
                }
                selected_world = world_points[selected_local]
                for row_index, local_index in enumerate(selected_local):
                    original_index = int(normal_indices[local_index])
                    supported = bool(gt_supported[local_index])
                    if not supported:
                        unsupported_exported[str(gt_names[local_index])] += 1
                    gt_source_exported[str(gt_sources[local_index])] += 1
                    row = {
                        'schema_version': RELIABILITY_SCHEMA_VERSION,
                        'dataset_id': dataset.name,
                        'frame_index': frame_index,
                        'lidar_frame': lidar_frame,
                        'lidar_timestamp': lidar_timestamp,
                        'rgb_frame': int(rgb_record['frame']),
                        'rgb_timestamp': float(rgb_record['timestamp_sec']),
                        'requested_offset_ms': offset,
                        'actual_offset_ms': selected_rgb.actual_offset_ms,
                        'timing_error_ms': selected_rgb.timing_error_ms,
                        'point_index': original_index,
                        'semantic_match_index': int(
                            semantic_indices[local_index]),
                        'match_distance_m': float(
                            normal_match_distance[original_index]),
                        'x_lidar': float(valid_points[local_index, 0]),
                        'y_lidar': float(valid_points[local_index, 1]),
                        'z_lidar': float(valid_points[local_index, 2]),
                        'intensity': float(intensity[original_index]),
                        'channel': int(channels[original_index]),
                        'x_world': float(selected_world[row_index, 0]),
                        'y_world': float(selected_world[row_index, 1]),
                        'z_world': float(selected_world[row_index, 2]),
                        'u': int(valid_u[local_index]),
                        'v': int(valid_v[local_index]),
                        'actor_id': int(actor_ids[local_index]),
                        'carla_tag': int(carla_tags[local_index]),
                        'gt_project_id': int(gt_ids[local_index]),
                        'gt_class': str(gt_names[local_index]),
                        'gt_supported': supported,
                        'gt_source': str(gt_sources[local_index]),
                        'pred_project_id': int(predictions[row_index]),
                        'pred_class': DEFAULT_CLASSES[predictions[row_index]],
                        'correct': (
                            int(correct[row_index]) if supported else None),
                        'pred_confidence': float(pred_confidence[row_index]),
                        'gt_probability': (
                            float(gt_probability[row_index])
                            if supported else None),
                        'semantic_entropy': float(
                            semantic_entropy[local_index]),
                        'range_m': float(sensor_range[local_index]),
                        'local_density': int(local_density[local_index]),
                        'view_radius': float(view_radius[local_index]),
                        'angular_rms': motion['angular_rms'],
                        'accel_deviation': motion[
                            'acceleration_deviation'],
                        'imu_status': motion['status'],
                        'r_motion': float(r_motion[local_index]),
                        'r_density': float(r_density[local_index]),
                        'r_range': float(r_range[local_index]),
                        'r_view': float(r_view[local_index]),
                        'r_semantic': float(r_semantic[local_index]),
                        'r_combined': float(r_combined[local_index]),
                        'w_none': float(selected_weights['none'][row_index]),
                        'w_semantic': float(
                            selected_weights['semantic'][row_index]),
                        'w_semantic_range': float(
                            selected_weights['semantic_range'][row_index]),
                        'w_semantic_range_density': float(
                            selected_weights[
                                'semantic_range_density'][row_index]),
                        'w_semantic_range_density_view': float(
                            selected_weights[
                                'semantic_range_density_view'][row_index]),
                        'w_full': float(selected_weights['full'][row_index]),
                        'posterior_file': posterior_relative,
                        'posterior_row': (
                            row_index if posterior_relative else None),
                    }
                    point_writer.writerow({
                        name: _csv_value(row.get(name))
                        for name in PER_POINT_COLUMNS
                    })

                summary_chunks.append({
                    'correct': correct,
                    'gt_supported': selected_supported,
                    'gt_project_id': selected_gt_ids,
                    'pred_project_id': predictions,
                    'pred_confidence': pred_confidence,
                    'semantic_entropy': semantic_entropy[selected_local],
                    'r_motion': r_motion[selected_local],
                    'r_density': r_density[selected_local],
                    'r_range': r_range[selected_local],
                    'r_view': r_view[selected_local],
                    'r_semantic': r_semantic[selected_local],
                    'r_combined': r_combined[selected_local],
                    'range_m': sensor_range[selected_local],
                    'local_density': local_density[selected_local],
                    'view_radius': view_radius[selected_local],
                    'angular_rms': np.full(
                        len(selected_local), motion['angular_rms']),
                    'accel_deviation': np.full(
                        len(selected_local), motion[
                            'acceleration_deviation']),
                    'nll': nll,
                    'brier': brier,
                    'requested_offset_ms': np.full(
                        len(selected_local), offset),
                    **{
                        f'w_{name}': selected_weights[name]
                        for name in ABLATION_NAMES
                    },
                })

            if (frame_index + 1) % 10 == 0:
                print(
                    f'Evaluated {frame_index + 1}/{len(frame_paths)} '
                    f'LiDAR frames; exported={global_flow["exported_points"]}',
                    flush=True,
                )

    factor_rows = build_factor_summary(summary_chunks)
    factor_summary_path = output / 'factor_summary.csv'
    _write_summary_csv(factor_summary_path, factor_rows)

    if summary_chunks:
        correct = _concatenate_chunks(summary_chunks, 'correct')
        supported = _concatenate_chunks(summary_chunks, 'gt_supported', bool)
        gt_ids = _concatenate_chunks(
            summary_chunks, 'gt_project_id', np.int64)
        predictions = _concatenate_chunks(
            summary_chunks, 'pred_project_id', np.int64)
        confidence = _concatenate_chunks(
            summary_chunks, 'pred_confidence')
        nll = _concatenate_chunks(summary_chunks, 'nll')
        brier = _concatenate_chunks(summary_chunks, 'brier')
    else:
        correct = np.empty(0)
        supported = np.empty(0, dtype=bool)
        gt_ids = np.empty(0, dtype=np.int64)
        predictions = np.empty(0, dtype=np.int64)
        confidence = np.empty(0)
        nll = np.empty(0)
        brier = np.empty(0)
    primary = supported & np.isfinite(correct)
    calibration = {
        'sample_count': int(np.count_nonzero(primary)),
        'ece': (
            expected_calibration_error(confidence[primary], correct[primary])
            if np.any(primary) else None),
        'nll': _finite_mean(nll[primary]),
        'brier': _finite_mean(brier[primary]),
        'brier_definition': 'sum over project classes; not divided by K',
    }
    by_class = {}
    for class_id, class_name in enumerate(DEFAULT_CLASSES):
        selected = primary & (gt_ids == class_id)
        if not np.any(selected):
            continue
        by_class[class_name] = {
            'count': int(np.count_nonzero(selected)),
            'accuracy': float(np.mean(correct[selected])),
            'mean_confidence': float(np.mean(confidence[selected])),
        }

    factor_relationships = {}
    for name in (
            'motion', 'density', 'range', 'view', 'semantic', 'combined'):
        field = 'r_combined' if name == 'combined' else f'r_{name}'
        values = (
            _concatenate_chunks(summary_chunks, field)
            if summary_chunks else np.empty(0))
        factor_relationships[name] = _factor_relationship(
            values, correct, supported)

    ablation_report = {
        'interpretation': (
            'Single-frame argmax is unchanged. Weighted accuracy is a '
            'diagnostic ranking statistic; VoxelMap results test fusion.'),
        'point_level_diagnostic': {},
    }
    for name in ABLATION_NAMES:
        field = f'w_{name}'
        values = (
            _concatenate_chunks(summary_chunks, field)
            if summary_chunks else np.empty(0))
        ablation_report['point_level_diagnostic'][name] = (
            _weighted_accuracy(values, correct, supported))
    if args.voxel_eval:
        ablation_report['voxel_by_offset_ms'] = {
            f'{offset:g}': voxel_evaluators[offset].report()
            for offset in offsets
        }

    warnings = []
    checkpoint_unsupported = [
        name for name in DEFAULT_CLASSES[:-1]
        if name not in supported_classes
    ]
    if checkpoint_unsupported:
        warnings.append(
            'Checkpoint cannot emit project classes: '
            + ', '.join(checkpoint_unsupported)
            + '; their GT rows are retained but excluded from primary metrics.')
    if args.posterior_storage == 'none':
        warnings.append(
            'Full posterior storage was disabled; per_point.csv retains only '
            'confidence, GT probability and entropy.')
    if args.voxel_eval and (
        manifest.get('moving_targets')
        or manifest.get('ego_motion_profile') != 'stationary'
    ):
        warnings.append(
            'Voxel ablation includes a moving ego or moving targets; dynamic '
            'world-coordinate trails can confound reliability effects.')
    warnings.append(
        'CARLA IMU gravity semantics must be verified with a stationary run '
        'before interpreting motion reliability as real-robot evidence.')
    if not np.any(primary):
        warnings.append(
            'No supported point-level ground truth survived matching, '
            'projection and sampling; primary metrics are null.')

    accel_norms = np.asarray([
        np.linalg.norm(sample['accelerometer_mps2'])
        for sample in imu_samples
    ], dtype=np.float64)
    report = {
        'schema_version': RELIABILITY_SCHEMA_VERSION,
        'benchmark': 'carla_reliability',
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset': str(dataset),
        'output': str(output),
        'dataset_id': dataset.name,
        'git': git_provenance(Path(__file__).resolve().parents[2]),
        'software': _software_provenance(segformer),
        'dataset_support': {
            'format': manifest['format'],
            'format_version': manifest['format_version'],
            'carla_server_version': manifest.get('carla_server_version'),
            'carla_client_version': manifest.get('carla_client_version'),
            'town': manifest.get('town'),
            'weather_preset': manifest.get('weather_preset'),
            'weather': manifest.get('weather'),
            'seed': manifest.get('seed'),
            'ego_motion_profile': manifest.get('ego_motion_profile'),
            'moving_targets': manifest.get('moving_targets'),
            'sensors': manifest.get('sensors'),
        },
        'model': {
            'id': args.segformer_model,
            **_model_provenance(args.segformer_model, segformer),
            'device': segformer.device,
            'fp16': bool(segformer.use_fp16),
            'posterior_temperature': float(
                getattr(
                    segformer,
                    'posterior_temperature',
                    args.segformer_temperature,
                )
            ),
            'checkpoint_id2label': segformer.id2label,
            'project_class_order': list(DEFAULT_CLASSES),
            'supported_project_classes': list(supported_classes),
            'checkpoint_unsupported_project_classes': checkpoint_unsupported,
            'native_posterior_cache_hits': cache.hit_count,
            'native_posterior_cache_misses': cache.miss_count,
            'posterior_transport': (
                'native project posterior -> float16 -> float32 -> normalize'),
        },
        'reliability_parameters': {
            'source': str(params_path),
            'source_sha256': _hash_file(params_path),
            'values': params,
            'density_neighborhood_radius_m': 0.3,
            'imu_window_interpretation': 'plus/minus half-window, closed',
        },
        'experiment': {
            'time_offsets_ms': offsets,
            'frame_tolerance_ms': args.frame_tolerance_ms,
            'semantic_match_tolerance_m': (
                args.semantic_match_tolerance_m),
            'min_match_fraction': args.min_match_fraction,
            'point_sample_rate': args.point_sample_rate,
            'max_points_per_frame_per_offset': (
                args.max_points_per_frame),
            'posterior_storage': args.posterior_storage,
            'summary_scope': 'deterministically sampled exported points',
            'voxel_eval': bool(args.voxel_eval),
        },
        'alignment': _aggregate_alignment(
            alignment_reports, distance_sample),
        'offset_flow': {
            f'{offset:g}': dict(offset_flow[offset]) for offset in offsets
        },
        'sample_flow': dict(global_flow),
        'imu_audit': {
            'sample_count': len(imu_samples),
            'mean_acceleration_norm_mps2': _finite_mean(accel_norms),
            'median_acceleration_norm_mps2': (
                float(np.median(accel_norms)) if accel_norms.size else None),
            'configured_gravity_mps2': params['imu_gravity'],
        },
        'overall': {
            'exported_point_count': int(len(correct)),
            'supported_point_count': int(np.count_nonzero(primary)),
            'accuracy': (
                float(np.mean(correct[primary]))
                if np.any(primary) else None),
            'calibration': calibration,
        },
        'by_class': by_class,
        'unsupported': {
            'all_matched_projected_by_class': dict(
                unsupported_all_projected),
            'exported_by_class': dict(unsupported_exported),
            'exported_gt_source': dict(gt_source_exported),
        },
        'factor_relationships': factor_relationships,
        'ablation': ablation_report,
        'artifacts': {
            'per_point_csv': per_point_path.name,
            'factor_summary_csv': factor_summary_path.name,
            'posterior_directory': (
                posterior_directory.name
                if args.posterior_storage == 'npz' else None),
            'posterior_dtype': (
                'float16' if args.posterior_storage == 'npz' else None),
            'posterior_class_order': list(DEFAULT_CLASSES),
        },
        'warnings': warnings,
        'elapsed_seconds': float(time.monotonic() - started),
    }
    report_path = output / 'report.json'
    write_strict_json(report_path, report)
    print(json.dumps({
        'output': str(output),
        'exported_points': int(len(correct)),
        'supported_points': int(np.count_nonzero(primary)),
        'accuracy': report['overall']['accuracy'],
        'report': str(report_path),
    }, indent=2, ensure_ascii=False))
    return report


def main(argv=None):
    """Run the reliability evaluator as a ROS-installed console program."""
    args = _build_parser().parse_args(argv)
    try:
        evaluate(args)
        return 0
    except Exception as exc:
        print(f'CARLA reliability evaluation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
