import json

import numpy as np
import pytest
from PIL import Image

from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES
from semantic_mapping.runtime.segformer_training import (
    DatasetValidationError,
    build_classifier_copy_plan,
    checkpoint_integrity,
    checkpoint_report,
    confusion_metrics,
    initialize_dataset_layout,
    supported_project_classes,
    training_report_meets_thresholds,
    validate_segformer_dataset,
)


def _write_sample(root, split, stem, labels):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 1] = 120
    Image.fromarray(image, mode='RGB').save(
        root / split / 'images' / f'{stem}.png')
    Image.fromarray(np.asarray(labels, dtype=np.uint8), mode='L').save(
        root / split / 'masks' / f'{stem}.png')


def test_dataset_layout_and_required_target_coverage(tmp_path):
    root = initialize_dataset_layout(tmp_path / 'ebike_dataset')
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    labels = [[
        class_id['road'],
        class_id['car'],
        class_id['bicycle'],
    ], [
        class_id['electric_bicycle'],
        class_id['motorcycle'],
        255,
    ]]
    _write_sample(root, 'train', 'sample_train', labels)
    _write_sample(root, 'val', 'sample_val', labels)

    report = validate_segformer_dataset(root)
    saved_labels = json.loads((root / 'labels.json').read_text())

    assert report['valid'] is True
    assert report['splits']['train']['sample_count'] == 1
    assert report['splits']['val']['class_pixels']['electric_bicycle'] == 1
    assert report['split_protocol']['formal_evaluation_ready'] is False
    assert report['split_protocol']['final_evaluation_split'] == 'val'
    assert any('NO INDEPENDENT TEST SPLIT' in item
               for item in report['warnings'])
    assert saved_labels['classes'][str(class_id['electric_bicycle'])] == (
        'electric_bicycle')


def test_dataset_supports_independent_calibration_and_test_splits(tmp_path):
    root = initialize_dataset_layout(tmp_path / 'formal_dataset')
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    labels = [[
        class_id['road'],
        class_id['car'],
        class_id['bicycle'],
    ], [
        class_id['electric_bicycle'],
        class_id['motorcycle'],
        255,
    ]]
    for split in ('train', 'val', 'calibration', 'test'):
        _write_sample(root, split, f'sample_{split}', labels)

    report = validate_segformer_dataset(root)

    assert set(report['splits']) == {
        'train', 'val', 'calibration', 'test'}
    assert report['split_protocol'] == {
        'training_split': 'train',
        'model_selection_split': 'val',
        'calibration_split': 'calibration',
        'final_evaluation_split': 'test',
        'independent_test_present': True,
        'formal_evaluation_ready': True,
    }
    assert not any('NO INDEPENDENT TEST SPLIT' in item
                   for item in report['warnings'])


def test_partially_populated_optional_split_is_rejected(tmp_path):
    root = initialize_dataset_layout(tmp_path / 'broken_optional')
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    labels = [[
        class_id['road'],
        class_id['car'],
        class_id['bicycle'],
    ], [
        class_id['electric_bicycle'],
        class_id['motorcycle'],
        255,
    ]]
    _write_sample(root, 'train', 'sample_train', labels)
    _write_sample(root, 'val', 'sample_val', labels)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    Image.fromarray(image, mode='RGB').save(
        root / 'test' / 'images' / 'missing_mask.png')

    with pytest.raises(
        DatasetValidationError,
        match='test image/mask pairing',
    ):
        validate_segformer_dataset(root)


def test_dataset_validation_rejects_missing_electric_bicycle(tmp_path):
    root = initialize_dataset_layout(tmp_path / 'missing_ebike')
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    labels = [[
        class_id['road'],
        class_id['car'],
        class_id['bicycle'],
    ], [
        class_id['motorcycle'],
        255,
        255,
    ]]
    _write_sample(root, 'train', 'sample_train', labels)
    _write_sample(root, 'val', 'sample_val', labels)

    with pytest.raises(DatasetValidationError, match='electric_bicycle'):
        validate_segformer_dataset(root)


def test_stock_classifier_initializes_electric_channel_from_two_wheelers():
    id2label = {
        13: 'car',
        17: 'motorcycle',
        18: 'bicycle',
    }
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}

    plan = build_classifier_copy_plan(id2label)

    assert plan[class_id['car']] == [13]
    assert plan[class_id['bicycle']] == [18]
    assert plan[class_id['motorcycle']] == [17]
    assert plan[class_id['electric_bicycle']] == [17, 18]
    assert supported_project_classes(id2label) == (
        'car', 'bicycle', 'motorcycle')


def test_custom_checkpoint_reports_electric_bicycle_support():
    supported = supported_project_classes({
        0: 'car',
        1: 'bicycle',
        2: 'electric_bicycle',
        3: 'motorcycle',
    })

    assert supported == (
        'car', 'bicycle', 'electric_bicycle', 'motorcycle')


def test_confusion_metrics_report_electric_bicycle_iou():
    class_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    confusion = np.zeros(
        (len(DEFAULT_CLASSES), len(DEFAULT_CLASSES)), dtype=np.int64)
    ebike = class_id['electric_bicycle']
    bicycle = class_id['bicycle']
    confusion[ebike, ebike] = 8
    confusion[ebike, bicycle] = 2
    confusion[bicycle, bicycle] = 5

    metrics = confusion_metrics(confusion)

    assert metrics['per_class_iou']['electric_bicycle'] == pytest.approx(0.8)
    assert metrics['support_pixels']['electric_bicycle'] == 10


def test_checkpoint_requires_accepted_training_metrics(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    id2label = {
        str(index): class_name
        for index, class_name in enumerate(DEFAULT_CLASSES)
    }
    config = {
        'model_type': 'segformer',
        'num_labels': len(DEFAULT_CLASSES),
        'id2label': id2label,
        'label2id': {
            class_name: index
            for index, class_name in enumerate(DEFAULT_CLASSES)
        },
    }
    (checkpoint / 'config.json').write_text(json.dumps(config))

    unverified = checkpoint_report(
        checkpoint, local_files_only=True)
    schema_only = checkpoint_report(
        checkpoint,
        local_files_only=True,
        require_training_report=False,
    )
    (tmp_path / 'training_report.json').write_text(json.dumps({
        'accepted': True,
        'best_metrics': {
            'per_class_iou': {
                'road': 0.8,
                'car': 0.7,
                'bicycle': 0.6,
                'electric_bicycle': 0.5,
                'motorcycle': 0.6,
            },
        },
    }))
    accepted = checkpoint_report(checkpoint, local_files_only=True)

    assert unverified['schema_support_valid'] is True
    assert unverified['valid_for_requested_target_distinction'] is False
    assert schema_only['valid_for_requested_target_distinction'] is True
    assert accepted['metrics_accepted'] is True
    assert accepted['valid_for_requested_target_distinction'] is True
    assert accepted['valid_for_formal_evaluation'] is False
    assert accepted['checkpoint_hash_verified'] is False
    assert any('NON-FORMAL TRAINING REPORT' in item
               for item in accepted['warnings'])


def test_checkpoint_report_verifies_formal_test_metrics_and_hash(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    config = {
        'model_type': 'segformer',
        'num_labels': len(DEFAULT_CLASSES),
        'id2label': {
            str(index): class_name
            for index, class_name in enumerate(DEFAULT_CLASSES)
        },
        'label2id': {
            class_name: index
            for index, class_name in enumerate(DEFAULT_CLASSES)
        },
    }
    (checkpoint / 'config.json').write_text(json.dumps(config))
    integrity = checkpoint_integrity(checkpoint)
    passing_iou = {
        'road': 0.8,
        'car': 0.7,
        'bicycle': 0.6,
        'electric_bicycle': 0.5,
        'motorcycle': 0.6,
    }
    (tmp_path / 'training_report.json').write_text(json.dumps({
        'accepted': True,
        'acceptance_metrics_split': 'test',
        'acceptance_metrics': {'per_class_iou': passing_iou},
        'valid_for_formal_evaluation': True,
        'checkpoint_sha256': integrity['aggregate_sha256'],
    }))

    report = checkpoint_report(checkpoint, local_files_only=True)

    assert report['checkpoint_hash_verified'] is True
    assert report['valid_for_formal_evaluation'] is True
    assert report['warnings'] == []


def test_checkpoint_report_rejects_hash_mismatch_for_formal_use(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    config = {
        'model_type': 'segformer',
        'num_labels': len(DEFAULT_CLASSES),
        'id2label': {
            str(index): class_name
            for index, class_name in enumerate(DEFAULT_CLASSES)
        },
    }
    (checkpoint / 'config.json').write_text(json.dumps(config))
    passing_iou = {
        'road': 0.8,
        'car': 0.7,
        'bicycle': 0.6,
        'electric_bicycle': 0.5,
        'motorcycle': 0.6,
    }
    (tmp_path / 'training_report.json').write_text(json.dumps({
        'accepted': True,
        'acceptance_metrics_split': 'test',
        'acceptance_metrics': {'per_class_iou': passing_iou},
        'valid_for_formal_evaluation': True,
        'checkpoint_sha256': '0' * 64,
    }))

    report = checkpoint_report(checkpoint, local_files_only=True)

    assert report['metrics_accepted'] is True
    assert report['checkpoint_hash_verified'] is False
    assert report['valid_for_formal_evaluation'] is False
    assert any('HASH MISMATCH' in item for item in report['warnings'])


def test_runtime_thresholds_reject_zero_iou_accepted_flag():
    accepted, failures = training_report_meets_thresholds({
        'accepted': True,
        'best_metrics': {
            'per_class_iou': {
                'road': 0.0,
                'car': 0.0,
                'bicycle': 0.0,
                'electric_bicycle': 0.0,
                'motorcycle': 0.0,
            },
        },
    })

    assert accepted is False
    assert set(failures) == {
        'road', 'car', 'bicycle', 'electric_bicycle', 'motorcycle'}


def test_runtime_thresholds_use_acceptance_metrics_before_validation_metrics():
    passing_iou = {
        'road': 0.8,
        'car': 0.7,
        'bicycle': 0.6,
        'electric_bicycle': 0.5,
        'motorcycle': 0.6,
    }
    failing_test_iou = dict(passing_iou, electric_bicycle=0.1)

    accepted, failures = training_report_meets_thresholds({
        'accepted': True,
        'best_metrics': {'per_class_iou': passing_iou},
        'acceptance_metrics': {'per_class_iou': failing_test_iou},
        'acceptance_metrics_split': 'test',
    })

    assert accepted is False
    assert failures == {'electric_bicycle': 0.1}
