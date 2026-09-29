"""
Shared checkpoint/report checks for online inference and offline training.

No training loop, image augmentation, model loading or ROS dependencies live here.
"""

import hashlib
import json
from pathlib import Path


DEFAULT_MIN_TARGET_IOU = 0.30
DEFAULT_MIN_ELECTRIC_BICYCLE_IOU = 0.35
DEFAULT_MIN_ROAD_IOU = 0.50


def checkpoint_integrity(checkpoint):
    """Return stable per-file and aggregate SHA-256 hashes for a checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f'Checkpoint directory not found: {checkpoint}')
    file_hashes = {}
    aggregate = hashlib.sha256()
    paths = sorted(path for path in checkpoint.rglob('*') if path.is_file())
    for path in paths:
        relative_path = path.relative_to(checkpoint).as_posix()
        file_digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                file_digest.update(block)
        digest = file_digest.hexdigest()
        file_hashes[relative_path] = digest
        aggregate.update(relative_path.encode('utf-8'))
        aggregate.update(b'\0')
        aggregate.update(bytes.fromhex(digest))
    if not file_hashes:
        raise ValueError(
            f'Checkpoint contains no files to hash: {checkpoint}')
    return {
        'algorithm': 'sha256',
        'aggregate_sha256': aggregate.hexdigest(),
        'files': file_hashes,
    }


def _acceptance_metrics(training_report):
    """Choose final metrics, preferring an explicitly recorded test result."""
    for key in ('acceptance_metrics', 'test_metrics', 'best_metrics'):
        metrics = training_report.get(key)
        if isinstance(metrics, dict):
            return key, metrics
    return None, {}


def find_threshold_failures(
    metrics,
    min_target_iou,
    min_electric_bicycle_iou,
    min_road_iou,
):
    """Return per-class failures for one immutable evaluation result."""
    class_ious = metrics.get('per_class_iou', {})
    failures = {}
    for class_name in ('car', 'bicycle', 'motorcycle'):
        value = float(class_ious.get(class_name, 0.0))
        if value < min_target_iou:
            failures[class_name] = value
    electric_iou = float(class_ious.get('electric_bicycle', 0.0))
    electric_minimum = max(
        float(min_target_iou), float(min_electric_bicycle_iou))
    if electric_iou < electric_minimum:
        failures['electric_bicycle'] = electric_iou
    road_iou = float(class_ious.get('road', 0.0))
    if road_iou < min_road_iou:
        failures['road'] = road_iou
    return failures


def load_training_report(model_id):
    """Load the nearest local training report for a checkpoint, if present."""
    model_path = Path(str(model_id)).expanduser()
    if not model_path.exists():
        return None, None
    candidates = (
        model_path / 'training_report.json',
        model_path.parent / 'training_report.json',
    )
    for candidate in candidates:
        if candidate.is_file():
            return (
                candidate.resolve(),
                json.loads(candidate.read_text(encoding='utf-8')),
            )
    return None, None


def training_report_meets_thresholds(
    training_report,
    min_target_iou=DEFAULT_MIN_TARGET_IOU,
    min_electric_bicycle_iou=DEFAULT_MIN_ELECTRIC_BICYCLE_IOU,
    min_road_iou=DEFAULT_MIN_ROAD_IOU,
):
    """Independently verify saved metrics against runtime safety thresholds."""
    if training_report is None:
        return False, {'training_report': 'missing'}
    if not training_report.get('accepted', False):
        return False, {'training_report': 'training command did not accept it'}
    _, metrics = _acceptance_metrics(training_report)
    failures = find_threshold_failures(
        metrics,
        min_target_iou,
        min_electric_bicycle_iou,
        min_road_iou,
    )
    return not failures, failures


def verify_local_training_checkpoint(model_id, training_report):
    """Verify that a local checkpoint is the one bound to its report."""
    if not isinstance(training_report, dict):
        return False, 'training report is missing or invalid'
    checkpoint_path = Path(str(model_id)).expanduser()
    if not checkpoint_path.is_dir():
        return False, 'checkpoint is not a local directory'
    recorded_hash = training_report.get('checkpoint_sha256')
    if not recorded_hash:
        return False, 'training report has no checkpoint_sha256'
    try:
        current_hash = checkpoint_integrity(
            checkpoint_path)['aggregate_sha256']
    except (FileNotFoundError, OSError, ValueError) as exc:
        return False, f'checkpoint hashing failed: {exc}'
    if current_hash != recorded_hash:
        return (
            False,
            f'checkpoint hash mismatch: recorded={recorded_hash}, '
            f'current={current_hash}',
        )
    return True, current_hash
