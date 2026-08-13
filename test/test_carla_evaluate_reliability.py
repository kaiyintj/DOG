"""
Offline contracts for the CARLA point-level reliability evaluator.

These tests intentionally exercise only NumPy/YAML-style helpers.  They do
not require a CARLA server, ROS graph, or a SegFormer checkpoint, so they
guard the evaluator's dataset and formula contracts independently of a full
benchmark run.
"""

import argparse
import csv
import json

import numpy as np
import pytest
from PIL import Image

import semantic_mapping.carla.carla_evaluate_reliability as reliability_evaluator
from semantic_mapping.carla.carla_evaluate_reliability import (
    build_factor_summary,
    carla_lidar_to_optical_transform,
    evaluate,
    load_reliability_parameters,
    relative_pose_motion,
    select_imu_motion,
    validate_reliability_manifest,
)
from semantic_mapping.carla.carla_reliability import (
    FACTOR_SUMMARY_COLUMNS,
    PER_POINT_COLUMNS,
)
from semantic_mapping.runtime.reliability_factors import compute_motion_reliability
from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES


def _matrix_with_translation(x=0.0, y=0.0, z=0.0):
    """Return a homogeneous translation matrix for pose-contract tests."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _runtime_parameters():
    """Return the current GA-BSVM motion-factor subset."""
    return {
        'imu_window_sec': 0.15,
        'motion_angular_scale': 2.0,
        'motion_accel_scale': 3.0,
        'motion_rotation_scale_rad': 0.05,
        'motion_translation_scale_m': 2.0,
        'motion_min_reliability': 0.2,
        'motion_missing_reliability': 0.2,
        'imu_gravity': 9.81,
    }


def _reliability_yaml(max_observation_weight=0.0):
    """Build a minimal, valid GA-BSVM reliability configuration."""
    parameters = {
        **_runtime_parameters(),
        'density_scale': 8.0,
        'range_scale_m': 20.0,
        'view_edge_penalty': 0.4,
        'semantic_confidence_floor': 0.15,
        'voxel_size': 0.1,
        'evidence_prior': 1.0,
        'evidence_strength': 1.0,
        'evidence_decay': 0.995,
        'evidence_decay_reference_sec': 0.1,
        'max_frame_evidence': 3.0,
        'max_total_evidence': 200.0,
        'max_observation_weight': max_observation_weight,
        'dynamic_voxel_ttl_sec': 10.0,
        'voxel_ttl_sec': 300.0,
    }
    lines = ['ga_bsvm_node:', '  ros__parameters:']
    for key, value in parameters.items():
        lines.append(f'    {key}: {value}')
    return '\n'.join(lines) + '\n'


def _valid_manifest():
    """Return the smallest metadata object accepted by reliability v1."""
    return {
        'format': 'carla_reliability_v1',
        'format_version': 1,
        'benchmark': 'carla_reliability',
        'town': 'Carla/Maps/Town05',
        'fixed_delta_seconds': 0.01,
        'semantic_tag_ids': {'road': [1], 'car': [14]},
        'rgb_index': 'rgb_index.json',
        'imu_csv': 'imu/imu.csv',
        'sensors': {
            'rgb': {},
            'lidar': {},
            'semantic_lidar': {},
            'semantic_camera': {},
            'instance_camera': {},
            'imu': {},
        },
        'frames': ['metadata/00000001.json'],
    }


def test_carla_transform_changes_carla_axes_to_camera_optical_axes():
    """CARLA forward/right/up axes must become optical right/down/forward."""
    transform = carla_lidar_to_optical_transform(np.eye(4), np.eye(4))

    optical = transform @ np.asarray([1.0, 2.0, 3.0, 1.0])

    # CARLA local/world axes are x-forward, y-right, z-up.  The camera
    # projection convention is x-right, y-down, z-forward.
    np.testing.assert_allclose(optical, [2.0, -3.0, 1.0, 1.0])


def test_carla_transform_uses_the_historical_camera_pose():
    """Projection must use the selected RGB frame's historical camera pose."""
    lidar_to_world = _matrix_with_translation(x=10.0)
    historical_camera_to_world = _matrix_with_translation(x=5.0)

    transform = carla_lidar_to_optical_transform(
        lidar_to_world,
        historical_camera_to_world,
    )

    # A LiDAR-local point at x=1 is at world x=11, hence six metres ahead of
    # the older camera pose at world x=5.
    np.testing.assert_allclose(
        transform @ np.asarray([1.0, 2.0, 3.0, 1.0]),
        [2.0, -3.0, 6.0, 1.0],
    )


def test_relative_pose_motion_reports_rotation_and_translation():
    """Motion V2 uses the reference-to-historical camera pose change."""
    reference = _matrix_with_translation(x=1.0, y=2.0, z=3.0)
    historical = _matrix_with_translation(x=4.0, y=6.0, z=3.0)
    angle = np.deg2rad(30.0)
    historical[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]

    rotation, translation = relative_pose_motion(reference, historical)

    assert rotation == pytest.approx(angle)
    assert translation == pytest.approx(5.0)
    assert relative_pose_motion(reference, reference) == pytest.approx(
        (0.0, 0.0))


def test_load_parameters_reads_runtime_and_voxel_settings(tmp_path):
    """The evaluator must read factor and fusion parameters from GA-BSVM."""
    config_path = tmp_path / 'reliability.yaml'
    config_path.write_text(_reliability_yaml(), encoding='utf-8')

    parameters = load_reliability_parameters(config_path)

    assert parameters['imu_window_sec'] == pytest.approx(0.15)
    assert parameters['motion_angular_scale'] == pytest.approx(2.0)
    assert parameters['motion_rotation_scale_rad'] == pytest.approx(0.05)
    assert parameters['motion_translation_scale_m'] == pytest.approx(2.0)
    assert parameters['density_scale'] == pytest.approx(8.0)
    assert parameters['range_scale_m'] == pytest.approx(20.0)
    assert parameters['voxel_size'] == pytest.approx(0.1)
    assert parameters['max_total_evidence'] == pytest.approx(200.0)
    # GA-BSVM derives a finite EMA cap when this configuration value is zero.
    assert parameters['max_observation_weight'] is None


def test_load_parameters_rejects_missing_runtime_fields(tmp_path):
    """An incomplete parameter profile must fail before evaluation starts."""
    config_path = tmp_path / 'incomplete.yaml'
    config_path.write_text(
        'ga_bsvm_node:\n  ros__parameters:\n    imu_window_sec: 0.15\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='missing|required'):
        load_reliability_parameters(config_path)


def test_select_imu_motion_uses_closed_runtime_alignment_window():
    """Motion samples exactly on both half-window boundaries are included."""
    parameters = _runtime_parameters()
    samples = [
        {
            'timestamp_sec': 0.85,
            'accelerometer_mps2': [0.0, 0.0, 9.81],
            'gyroscope_radps': [2.0, 0.0, 0.0],
        },
        {
            'timestamp_sec': 1.00,
            'accelerometer_mps2': [0.0, 0.0, 12.81],
            'gyroscope_radps': [0.0, 0.0, 0.0],
        },
        {
            'timestamp_sec': 1.15,
            'accelerometer_mps2': [0.0, 0.0, 9.81],
            'gyroscope_radps': [0.0, 0.0, 2.0],
        },
        {
            'timestamp_sec': 1.151,
            'accelerometer_mps2': [0.0, 0.0, 9.81],
            'gyroscope_radps': [99.0, 0.0, 0.0],
        },
    ]

    selected = select_imu_motion(samples, lidar_timestamp=1.0, params=parameters)
    expected = compute_motion_reliability(
        angular_norms=[2.0, 0.0, 2.0],
        acceleration_norms=[9.81, 12.81, 9.81],
        angular_scale=parameters['motion_angular_scale'],
        acceleration_scale=parameters['motion_accel_scale'],
        gravity=parameters['imu_gravity'],
        min_reliability=parameters['motion_min_reliability'],
    )

    assert selected['status'] == 'ok'
    assert selected['sample_count'] == 3
    assert selected['reliability'] == pytest.approx(expected[0])
    assert selected['angular_rms'] == pytest.approx(expected[1])
    assert selected['acceleration_deviation'] == pytest.approx(expected[2])


def test_select_imu_motion_fails_closed_without_aligned_measurements():
    """Missing aligned IMU data uses the configured conservative weight."""
    parameters = _runtime_parameters()
    selected = select_imu_motion(
        [{
            'timestamp_sec': 1.151,
            'accelerometer_mps2': [0.0, 0.0, 9.81],
            'gyroscope_radps': [0.0, 0.0, 0.0],
        }],
        lidar_timestamp=1.0,
        params=parameters,
    )

    assert selected == {
        'status': 'missing',
        'reliability': pytest.approx(0.2),
        'angular_rms': pytest.approx(0.0),
        'acceleration_deviation': pytest.approx(0.0),
        'sample_count': 0,
    }


def test_manifest_validation_accepts_only_reliability_v1_metadata():
    """The new evaluator must reject the independent image-dataset format."""
    assert validate_reliability_manifest(_valid_manifest()) is not None

    wrong_format = _valid_manifest()
    wrong_format['format'] = 'carla_image_benchmark_v2'
    with pytest.raises(ValueError, match='format|reliability'):
        validate_reliability_manifest(wrong_format)

    missing_frames = _valid_manifest()
    missing_frames.pop('frames')
    with pytest.raises(ValueError, match='frames|required'):
        validate_reliability_manifest(missing_frames)

    non_list_frames = _valid_manifest()
    non_list_frames['frames'] = 'metadata/00000001.json'
    with pytest.raises(ValueError, match='frames'):
        validate_reliability_manifest(non_list_frames)


def test_factor_summary_preserves_empty_bins_and_excludes_unsupported():
    """Empty bins stay null and unsupported truth never enters accuracy."""
    chunks = [{
        'correct': np.asarray([1.0, 0.0, 1.0, 0.0]),
        'gt_supported': np.asarray([True, True, False, True]),
        'pred_confidence': np.asarray([0.9, 0.7, 0.2, 0.4]),
        'semantic_entropy': np.asarray([0.1, 0.5, 1.0, 0.7]),
        'r_motion': np.asarray([0.9, 0.8, 0.1, 0.4]),
        'r_motion_v1': np.asarray([1.0, 0.9, 0.5, 0.7]),
        'r_density': np.asarray([0.9, 0.7, 0.1, 0.4]),
        'r_range': np.asarray([0.9, 0.6, 0.1, 0.4]),
        'r_view': np.asarray([0.9, 0.5, 0.1, 0.4]),
        'r_semantic': np.asarray([0.9, 0.4, 0.1, 0.4]),
        'r_combined': np.asarray([0.9, 0.6, 0.1, 0.4]),
        'range_m': np.asarray([1.0, 8.0, 25.0, 15.0]),
        'local_density': np.asarray([1.0, 4.0, 12.0, 8.0]),
        'view_radius': np.asarray([0.1, 0.3, 0.8, 0.5]),
        'angular_rms': np.asarray([0.0, 0.2, 2.0, 1.0]),
        'accel_deviation': np.asarray([0.0, 0.1, 1.0, 0.3]),
        'relative_rotation_rad': np.asarray([0.0, 0.002, 0.01, 0.02]),
        'relative_translation_m': np.asarray([0.0, 0.1, 0.5, 1.0]),
        'nll': np.asarray([0.1, 0.4, 3.0, 0.9]),
        'brier': np.asarray([0.02, 0.3, 1.5, 0.8]),
        'requested_offset_ms': np.asarray([0.0, 20.0, 50.0, 100.0]),
    }]

    rows = build_factor_summary(chunks)
    combined_rows = [
        row for row in rows
        if row['factor'] == 'combined' and row['axis'] == 'reliability'
    ]

    assert combined_rows
    assert all(set(FACTOR_SUMMARY_COLUMNS).issubset(row) for row in rows)
    # The shared bin contract is left-closed: a value exactly 0.1 belongs to
    # the bin that starts at 0.1, rather than the preceding bin ending there.
    unsupported_bin = next(
        row for row in combined_rows if np.isclose(row['bin_left'], 0.1)
    )
    assert unsupported_bin['sample_count'] == 0
    assert unsupported_bin['unsupported_count'] == 1
    assert unsupported_bin['accuracy'] is None
    assert unsupported_bin['nll'] is None
    assert unsupported_bin['brier'] is None


def test_evaluate_runs_end_to_end_on_a_minimal_offline_reliability_dataset(
    tmp_path,
    monkeypatch,
):
    """Exercise dataset I/O, projection, GT, posterior and ablations offline."""
    dataset = tmp_path / 'synthetic_reliability_v1'
    for relative in (
        'rgb', 'lidar', 'semantic_lidar', 'imu', 'metadata',
    ):
        (dataset / relative).mkdir(parents=True, exist_ok=True)

    # Camera and LiDAR poses are both identity. CARLA x-forward points become
    # camera-optical z-forward points, so these returns project into a 2x2 RGB
    # image with FOV=90 degrees.
    rgb_relative = 'rgb/00000001.png'
    Image.new('RGB', (2, 2), color=(5, 10, 15)).save(dataset / rgb_relative)
    normal_relative = 'lidar/00000001.npz'
    semantic_relative = 'semantic_lidar/00000001.npz'
    points = np.asarray([[3.0, 0.0, 0.0], [3.0, 0.1, 0.0]], dtype=np.float32)
    np.savez_compressed(
        dataset / normal_relative,
        xyz=points,
        intensity=np.asarray([0.5, 0.75], dtype=np.float32),
        channel=np.asarray([0, 0], dtype=np.uint16),
    )
    np.savez_compressed(
        dataset / semantic_relative,
        xyz=points.copy(),
        object_idx=np.asarray([0, 0], dtype=np.uint32),
        object_tag=np.asarray([1, 1], dtype=np.uint32),
    )

    identity = np.eye(4, dtype=np.float64).tolist()
    rgb_record = {
        'frame': 1,
        'timestamp': 1.0,
        'timestamp_sec': 1.0,
        'path': rgb_relative,
        'transform_matrix': identity,
    }
    metadata = {
        'frame': 1,
        'rgb': rgb_record,
        'lidar': {
            'frame': 1,
            'timestamp_sec': 1.0,
            'path': normal_relative,
            'transform_matrix': identity,
        },
        'semantic_lidar': {
            'frame': 1,
            'timestamp_sec': 1.0,
            'path': semantic_relative,
            'transform_matrix': identity,
        },
    }
    (dataset / 'metadata/00000001.json').write_text(
        json.dumps(metadata), encoding='utf-8')
    (dataset / 'rgb_index.json').write_text(
        json.dumps([rgb_record]), encoding='utf-8')
    (dataset / 'imu/imu.csv').write_text(
        'frame,timestamp_sec,accel_x_mps2,accel_y_mps2,accel_z_mps2,'
        'gyro_x_radps,gyro_y_radps,gyro_z_radps,compass_rad\n'
        '1,1.0,0.0,0.0,9.81,0.0,0.0,0.0,0.0\n',
        encoding='utf-8',
    )
    manifest = {
        **_valid_manifest(),
        'seed': 7,
        'rgb_index': 'rgb_index.json',
        'imu_csv': 'imu/imu.csv',
        'sensors': {
            'rgb': {'attributes': {'fov': '90.0'}},
            'lidar': {},
            'semantic_lidar': {},
            'semantic_camera': {},
            'instance_camera': {},
            'imu': {},
        },
        'frames': ['metadata/00000001.json'],
        # Actor ID zero has no metadata: tag 1 must fall back to road.
        'semantic_tag_ids': {'road': [1], 'car': [14]},
        'actor_id_map': {},
        'ego_motion_profile': 'stationary',
        'moving_targets': False,
    }
    (dataset / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    params_path = tmp_path / 'reliability.yaml'
    params_path.write_text(_reliability_yaml(), encoding='utf-8')

    class FakeSegformerEvaluator:
        """A model double that accepts only RGB PIL input, never LiDAR GT."""

        calls = []

        def __init__(
            self,
            model_id,
            device,
            confidence_threshold,
            use_fp16,
            posterior_temperature=1.0,
            revision=None,
        ):
            assert model_id == 'fake/segformer'
            assert device == 'cpu'
            assert confidence_threshold == 0.0
            assert use_fp16 is False
            assert posterior_temperature == 1.0
            assert revision is None
            self.device = 'cpu'
            self.use_fp16 = False
            self.requested_revision = revision
            self.id2label = {0: 'road', 1: 'car'}

        def infer_project_posterior(self, image):
            assert isinstance(image, Image.Image)
            assert image.mode == 'RGB'
            assert image.size == (2, 2)
            type(self).calls.append(image.copy())
            posterior = np.full(
                (2, 2, len(DEFAULT_CLASSES)),
                0.01 / (len(DEFAULT_CLASSES) - 1),
                dtype=np.float32,
            )
            posterior[..., 0] = 0.99
            return posterior

    monkeypatch.setattr(
        reliability_evaluator,
        'SegformerEvaluator',
        FakeSegformerEvaluator,
    )
    output = tmp_path / 'evaluation'
    args = argparse.Namespace(
        dataset=str(dataset),
        output=str(output),
        params_file=str(params_path),
        segformer_model='fake/segformer',
        segformer_revision='',
        segformer_temperature=1.0,
        device='cpu',
        fp16=False,
        time_offset_ms=[0.0],
        frame_tolerance_ms=0.0,
        semantic_match_tolerance_m=0.001,
        min_match_fraction=1.0,
        point_sample_rate=1.0,
        max_points_per_frame=0,
        posterior_storage='npz',
        posterior_cache_size=1,
        voxel_eval=True,
        limit=0,
    )

    returned_report = evaluate(args)

    per_point_path = output / 'per_point.csv'
    with per_point_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == PER_POINT_COLUMNS
        rows = list(reader)
    assert len(rows) == 2
    assert all(row['correct'] == '1' for row in rows)
    assert all(float(row['r_motion']) == pytest.approx(1.0) for row in rows)
    assert all(float(row['r_motion_v1']) == pytest.approx(1.0)
               for row in rows)
    assert all(float(row['relative_rotation_rad']) == pytest.approx(0.0)
               for row in rows)
    assert all(float(row['relative_translation_m']) == pytest.approx(0.0)
               for row in rows)
    for row in rows:
        assert row['posterior_file']
        posterior_path = output / row['posterior_file']
        assert posterior_path.is_file()
        with np.load(posterior_path) as stored:
            assert tuple(stored['class_names'].tolist()) == DEFAULT_CLASSES
            posterior_row = int(row['posterior_row'])
            assert int(np.argmax(stored['posterior'][posterior_row])) == 0

    with (output / 'factor_summary.csv').open(
        newline='', encoding='utf-8'
    ) as handle:
        assert tuple(csv.DictReader(handle).fieldnames) == FACTOR_SUMMARY_COLUMNS

    report_path = output / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    # JSON deliberately stringifies checkpoint class-ID mapping keys. The
    # returned in-memory report retains their integer form, so verify the
    # strict serialized artifact rather than requiring those representations
    # to compare equal.
    assert report['model']['checkpoint_id2label'] == {'0': 'road', '1': 'car'}
    assert returned_report['model']['checkpoint_id2label'] == {
        0: 'road', 1: 'car'}
    assert report['overall']['accuracy'] == pytest.approx(1.0)
    assert report['alignment']['accepted_frame_count'] == 1
    assert report['alignment']['rejected_frame_count'] == 0
    assert report['experiment']['motion_weighting'] == (
        'temporal_relative_camera_pose_v2')
    assert report['motion_by_offset_ms']['0'][
        'mean_motion_v2_reliability'] == pytest.approx(1.0)
    assert set(report['ablation']['voxel_by_offset_ms']['0']) == {
        'none',
        'semantic',
        'semantic_range',
        'semantic_range_density',
        'semantic_range_density_view',
        'full',
    }
    # The fake model only saw PIL RGB. Semantic LiDAR is used only after model
    # inference as evaluation ground truth.
    assert len(FakeSegformerEvaluator.calls) == 1
