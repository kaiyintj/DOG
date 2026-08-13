"""Tests for CARLA reliability alignment and metric helpers."""

import json

import numpy as np
import pytest

from semantic_mapping.carla.carla_reliability import (
    CARLA_TAG_PROJECT_ALIASES,
    FACTOR_SUMMARY_COLUMNS,
    NORMAL_LIDAR_DTYPE,
    PER_POINT_COLUMNS,
    SEMANTIC_LIDAR_DTYPE,
    area_under_risk_coverage,
    assign_bins,
    binary_auroc,
    calibration_metrics,
    decode_normal_lidar,
    decode_semantic_lidar,
    deterministic_sample_indices,
    expected_calibration_error,
    map_carla_ground_truth,
    match_semantic_lidar_points,
    multiclass_brier,
    multiclass_nll,
    reliability_gap,
    select_historical_frame,
    strict_json_dumps,
    summarize_binned,
    write_strict_json,
)


def test_lidar_raw_decoders_preserve_fields_and_reject_truncation():
    normal = np.zeros(2, dtype=NORMAL_LIDAR_DTYPE)
    normal['x'] = [1.0, 2.0]
    normal['intensity'] = [0.2, 0.8]
    semantic = np.zeros(2, dtype=SEMANTIC_LIDAR_DTYPE)
    semantic['x'] = [1.0, 2.0]
    semantic['object_idx'] = [123, 456]
    semantic['object_tag'] = [10, 19]

    decoded_normal = decode_normal_lidar(normal.tobytes())
    decoded_semantic = decode_semantic_lidar(semantic.tobytes())

    np.testing.assert_array_equal(decoded_normal, normal)
    np.testing.assert_array_equal(decoded_semantic, semantic)
    assert decoded_semantic['object_idx'].tolist() == [123, 456]
    with pytest.raises(ValueError, match='normal LiDAR payload'):
        decode_normal_lidar(normal.tobytes()[:-1])
    with pytest.raises(ValueError, match='semantic LiDAR payload'):
        decode_semantic_lidar(semantic.tobytes()[:-1])


def test_mutual_nearest_matching_is_one_to_one_and_keeps_original_indices():
    normal = np.asarray([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
    ])
    semantic = np.asarray([
        [0.001, 0.0, 0.0],
        [2.005, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ])

    result = match_semantic_lidar_points(normal, semantic, tolerance_m=0.02)

    assert result.normal_indices.tolist() == [0, 2]
    assert result.semantic_indices.tolist() == [0, 1]
    np.testing.assert_allclose(result.distances_m, [0.001, 0.005])
    statistics = result.statistics()
    assert statistics['matched_count'] == 2
    assert statistics['invalid_normal_count'] == 1
    assert statistics['matched_fraction'] == pytest.approx(0.5)
    assert statistics['p95_distance_m'] > 0.004


def test_matching_accepts_decoded_structured_arrays():
    normal = np.zeros(1, dtype=NORMAL_LIDAR_DTYPE)
    semantic = np.zeros(1, dtype=SEMANTIC_LIDAR_DTYPE)
    normal['x'] = 1.0
    semantic['x'] = 1.005

    result = match_semantic_lidar_points(normal, semantic, tolerance_m=0.01)

    assert result.matched_count == 1


def test_historical_frame_selection_uses_old_rgb_and_older_tie_break():
    frames = [
        {'frame': 1, 'timestamp': 9.90},
        {'frame': 2, 'timestamp': 9.95},
        {'frame': 3, 'timestamp': 10.00},
        {'frame': 4, 'timestamp': 10.05},
    ]

    selected = select_historical_frame(
        frames,
        lidar_timestamp=10.0,
        offset_ms=25.0,
        tolerance_ms=25.1,
    )

    assert selected.accepted
    assert selected.frame['frame'] == 2
    assert selected.selected_timestamp == 9.95
    assert selected.actual_offset_ms == pytest.approx(50.0)
    assert selected.timing_error_ms == pytest.approx(25.0)
    assert frames[selected.frame_index]['frame'] == 2


def test_historical_frame_selection_fails_closed_outside_tolerance():
    selected = select_historical_frame(
        [{'timestamp': 9.8}],
        lidar_timestamp=10.0,
        offset_ms=50.0,
        tolerance_ms=10.0,
    )

    assert not selected.accepted
    assert selected.frame is None
    assert selected.reason == 'outside_tolerance'
    assert selected.selected_timestamp == 9.8


def test_historical_selection_never_uses_future_of_lidar():
    selected = select_historical_frame(
        [{'timestamp': 10.01}],
        lidar_timestamp=10.0,
        offset_ms=0.0,
        tolerance_ms=100.0,
    )

    assert not selected.accepted
    assert selected.reason == 'no_historical_frame'


def test_actor_metadata_provides_fine_class_and_unsupported_is_retained():
    label = map_carla_ground_truth(
        object_tag=10,
        object_idx=123,
        semantic_tag_ids={'vehicle': [10]},
        actor_fine_classes={123: {'benchmark_class': 'electric_bicycle'}},
        supported_project_classes={'car', 'bicycle'},
    )

    assert label.class_name == 'electric_bicycle'
    assert label.project_id == 8
    assert not label.supported
    assert label.source == 'actor_metadata'


def test_actor_metadata_accepts_json_string_actor_id_keys():
    label = map_carla_ground_truth(
        object_tag=10,
        object_idx=123,
        semantic_tag_ids={'vehicle': [10]},
        actor_fine_classes={'123': {'benchmark_class': 'truck'}},
        supported_project_classes={'truck'},
    )

    assert label.class_name == 'truck'
    assert label.project_id == 5
    assert label.supported
    assert label.source == 'actor_metadata'


def test_semantic_tag_mapping_handles_rider_and_unmapped_tags():
    rider = map_carla_ground_truth(
        object_tag=13,
        object_idx=0,
        semantic_tag_ids={'rider': [13]},
        actor_fine_classes={},
        supported_project_classes={'person'},
    )
    unmapped = map_carla_ground_truth(
        object_tag=999,
        object_idx=0,
        semantic_tag_ids={'road': [7]},
        actor_fine_classes={},
        supported_project_classes={'road'},
    )

    assert rider.class_name == 'person'
    assert rider.project_id == 3
    assert rider.supported
    assert unmapped.project_id == -1
    assert unmapped.class_name == 'unmapped'
    assert not unmapped.supported


@pytest.mark.parametrize(
    ('semantic_key', 'expected_class', 'expected_project_id'),
    [
        ('sidewalk', 'road', 0),
        ('Sidewalks', 'road', 0),
        ('wall', 'building', 1),
        ('Fences', 'building', 1),
        ('vegetation', 'tree', 2),
        ('Trees', 'tree', 2),
    ],
)
def test_carla_environment_tag_aliases_fold_into_project_ontology(
    semantic_key,
    expected_class,
    expected_project_id,
):
    label = map_carla_ground_truth(
        object_tag=42,
        object_idx=0,
        semantic_tag_ids={semantic_key: [42]},
        actor_fine_classes={},
        supported_project_classes={'road', 'building', 'tree'},
    )

    assert label.class_name == expected_class
    assert label.project_id == expected_project_id
    assert label.supported
    assert label.source == 'semantic_tag'


def test_tag_alias_table_is_explicit_and_traffic_label_stays_unsupported():
    assert CARLA_TAG_PROJECT_ALIASES['sidewalk'] == 'road'
    assert CARLA_TAG_PROJECT_ALIASES['wall'] == 'building'
    assert CARLA_TAG_PROJECT_ALIASES['fence'] == 'building'
    assert CARLA_TAG_PROJECT_ALIASES['vegetation'] == 'tree'

    traffic_light = map_carla_ground_truth(
        object_tag=7,
        object_idx=0,
        semantic_tag_ids={'traffic light': [7]},
        actor_fine_classes={},
        supported_project_classes={'road', 'building', 'tree'},
    )
    assert traffic_light.class_name == 'unmapped'
    assert traffic_light.project_id == -1
    assert not traffic_light.supported
    assert traffic_light.source == 'unmapped_tag'


def test_generic_vehicle_tag_does_not_invent_a_fine_class():
    label = map_carla_ground_truth(
        object_tag=10,
        object_idx=999,
        semantic_tag_ids={'vehicle': [10], 'car': [10]},
        actor_fine_classes={},
        supported_project_classes={'car'},
    )

    assert label.project_id == -1
    assert label.class_name == 'vehicle'
    assert not label.supported


def test_deterministic_sampling_is_stateless_bounded_and_frame_specific():
    first = deterministic_sample_indices(
        1000, sample_rate=0.4, seed=42, frame_index=7, max_points=50)
    repeated = deterministic_sample_indices(
        1000, sample_rate=0.4, seed=42, frame_index=7, max_points=50)
    other_frame = deterministic_sample_indices(
        1000, sample_rate=0.4, seed=42, frame_index=8, max_points=50)

    np.testing.assert_array_equal(first, repeated)
    assert len(first) == 50
    assert np.all(first[:-1] < first[1:])
    assert not np.array_equal(first, other_frame)
    assert deterministic_sample_indices(10, sample_rate=0.0).size == 0
    assert deterministic_sample_indices(10, sample_rate=1.0).size == 10


def test_bin_assignment_is_left_closed_and_last_bin_is_right_closed():
    values = np.asarray([0.0, 0.1, 0.2, 1.0, -0.1, np.nan])

    assigned = assign_bins(values, [0.0, 0.1, 0.2, 1.0])

    assert assigned.tolist() == [0, 1, 2, 2, -1, -1]


def test_binned_summary_retains_unsupported_and_empty_bins_as_none():
    rows = summarize_binned(
        values=[0.05, 0.15, 0.16, 0.95],
        correct=[1, 0, 1, 1],
        reliability=[0.1, 0.2, 0.4, 0.9],
        edges=[0.0, 0.1, 0.2, 0.9, 1.0],
        confidence=[0.8, 0.7, 0.6, 0.9],
        entropy=[0.1, 0.2, 0.3, 0.05],
        eligible=[True, False, True, True],
    )

    assert rows[0]['sample_count'] == 1
    assert rows[1]['sample_count'] == 1
    assert rows[1]['unsupported_count'] == 1
    assert rows[1]['accuracy'] == 1.0
    assert rows[2]['sample_count'] == 0
    assert rows[2]['accuracy'] is None
    assert rows[2]['reliability_gap'] is None


def test_calibration_metrics_match_hand_calculation():
    posterior = np.asarray([
        [0.8, 0.2],
        [0.4, 0.6],
    ])
    labels = np.asarray([0, 0])

    assert multiclass_nll(posterior, labels) == pytest.approx(
        -0.5 * (np.log(0.8) + np.log(0.4)))
    expected_brier = np.mean([
        (0.8 - 1.0) ** 2 + 0.2 ** 2,
        (0.4 - 1.0) ** 2 + 0.6 ** 2,
    ])
    assert multiclass_brier(posterior, labels) == pytest.approx(
        expected_brier)
    metrics = calibration_metrics(posterior, labels, ece_edges=[0, 0.7, 1])
    assert metrics['sample_count'] == 2
    assert metrics['nll'] == pytest.approx(multiclass_nll(posterior, labels))
    assert metrics['brier'] == pytest.approx(expected_brier)
    assert metrics['ece'] == pytest.approx(0.4)


def test_calibration_excludes_unsupported_and_handles_empty_inputs():
    posterior = np.asarray([[0.8, 0.2], [0.4, 0.6]])

    one = calibration_metrics(
        posterior, [0, 0], eligible=[True, False])
    empty = calibration_metrics(
        posterior, [0, 0], eligible=[False, False])

    assert one['sample_count'] == 1
    assert one['ece'] == pytest.approx(0.2)
    assert empty == {
        'sample_count': 0,
        'ece': None,
        'nll': None,
        'brier': None,
    }
    assert expected_calibration_error([], []) is None
    assert multiclass_nll(np.empty((0, 2)), []) is None
    assert multiclass_brier(np.empty((0, 2)), []) is None


def test_ranking_and_reliability_metrics_cover_degenerate_cases():
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [1, 1, 0, 0]

    assert binary_auroc(scores, correct) == 1.0
    assert binary_auroc([0.1, 0.2], [1, 1]) is None
    assert area_under_risk_coverage(scores, correct) == pytest.approx(
        np.mean([0.0, 0.0, 1.0 / 3.0, 0.5]))
    assert area_under_risk_coverage([], []) is None
    assert reliability_gap([0.9, 0.1], [1, 0]) == pytest.approx(0.1)


def test_output_schema_and_strict_json_are_machine_stable(tmp_path):
    assert len(PER_POINT_COLUMNS) == len(set(PER_POINT_COLUMNS))
    assert len(FACTOR_SUMMARY_COLUMNS) == len(set(FACTOR_SUMMARY_COLUMNS))
    assert {
        'gt_supported',
        'gt_source',
        'r_combined',
        'timing_error_ms',
        'intensity',
        'channel',
        'posterior_file',
        'posterior_row',
    }.issubset(
        PER_POINT_COLUMNS)
    assert PER_POINT_COLUMNS.index('actual_offset_ms') + 1 == (
        PER_POINT_COLUMNS.index('timing_error_ms'))
    assert PER_POINT_COLUMNS.index('z_lidar') + 1 == (
        PER_POINT_COLUMNS.index('intensity'))
    assert PER_POINT_COLUMNS.index('intensity') + 1 == (
        PER_POINT_COLUMNS.index('channel'))
    assert PER_POINT_COLUMNS.index('gt_supported') + 1 == (
        PER_POINT_COLUMNS.index('gt_source'))
    assert PER_POINT_COLUMNS.index('posterior_file') + 1 == (
        PER_POINT_COLUMNS.index('posterior_row'))
    assert {'sample_count', 'reliability_gap'}.issubset(
        FACTOR_SUMMARY_COLUMNS)

    rendered = strict_json_dumps({
        'count': np.int64(2),
        'values': np.asarray([0.25, 0.75]),
    })
    assert json.loads(rendered) == {'count': 2, 'values': [0.25, 0.75]}
    path = write_strict_json(tmp_path / 'report.json', {'ok': True})
    assert json.loads(path.read_text(encoding='utf-8')) == {'ok': True}
    with pytest.raises(ValueError):
        strict_json_dumps({'invalid': np.inf})
