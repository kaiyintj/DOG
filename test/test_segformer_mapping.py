import numpy as np

from semantic_mapping.runtime.ga_bsvm_node import (
    GABsvmNode,
    normalize_frame_id,
    resolve_pointcloud_frame,
    segmentation_to_logits,
)
from semantic_mapping.runtime.segformer_node import (
    build_project_lookup,
    convert_image_to_rgb,
    find_relevant_model_labels,
    find_supported_project_classes,
    raw_prediction_statistics,
    split_model_label,
    verify_local_training_checkpoint,
)
from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES
from semantic_mapping.runtime.voxel_map import VoxelMap


def test_local_training_checkpoint_hash_is_enforced(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')

    from semantic_mapping.runtime.segformer_training import checkpoint_integrity

    digest = checkpoint_integrity(checkpoint)['aggregate_sha256']
    verified, detail = verify_local_training_checkpoint(
        checkpoint, {'checkpoint_sha256': digest})
    assert verified is True
    assert detail == digest

    (checkpoint / 'config.json').write_text(
        '{"changed": true}', encoding='utf-8')
    verified, detail = verify_local_training_checkpoint(
        checkpoint, {'checkpoint_sha256': digest})
    assert verified is False
    assert 'hash mismatch' in detail


def test_local_training_checkpoint_requires_recorded_hash(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')

    verified, detail = verify_local_training_checkpoint(checkpoint, {})

    assert verified is False
    assert detail == 'training report has no checkpoint_sha256'


def test_segformer_labels_map_to_navigation_classes():
    lookup = build_project_lookup({
        0: 'wall',
        1: 'road',
        2: 'tree',
        3: 'person',
        4: 'car',
        5: 'truck',
        6: 'bus',
        7: 'bicycle',
        8: 'minibike',
        9: 'chair',
        10: 'bench',
        11: 'sky',
    })

    class_index = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    assert np.array_equal(lookup, np.asarray([
        class_index['building'],
        class_index['road'],
        class_index['tree'],
        class_index['person'],
        class_index['car'],
        class_index['truck'],
        class_index['bus'],
        class_index['bicycle'],
        class_index['motorcycle'],
        class_index['chair'],
        class_index['bench'],
        class_index['unknown background'],
    ]))


def test_cityscapes_labels_map_to_navigation_classes():
    """The 19 Cityscapes classes must reach the project vocabulary."""
    class_index = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    unknown = len(DEFAULT_CLASSES) - 1
    lookup = build_project_lookup({
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
        5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
    })

    assert lookup[0] == class_index['road']
    assert lookup[1] == class_index['road']
    assert lookup[8] == class_index['tree']
    assert lookup[11] == class_index['person']
    assert lookup[12] == class_index['person']  # rider folds into person
    assert lookup[13] == class_index['car']
    assert lookup[17] == class_index['motorcycle']
    assert lookup[18] == class_index['bicycle']
    assert lookup[10] == unknown  # sky is dropped
    assert lookup[16] == unknown  # train is dropped


def test_compound_ade20k_label_maps_to_car():
    compound_label = 'car, auto, automobile, machine, motorcar'
    lookup = build_project_lookup({
        0: 'wall, brick wall',
        1: compound_label,
        2: 'automobile',
        3: 'sky',
    })

    assert 'car' in split_model_label(compound_label)
    class_index = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    assert lookup.tolist() == [
        class_index['building'],
        class_index['car'],
        class_index['car'],
        class_index['unknown background'],
    ]


def test_custom_segformer_checkpoint_can_expose_electric_bicycle():
    class_index = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    id2label = {
        0: 'car',
        1: 'bicycle',
        2: 'electric_bicycle',
        3: 'motorcycle',
    }

    lookup = build_project_lookup(id2label)
    supported = find_supported_project_classes(id2label)

    assert lookup.tolist() == [
        class_index['car'],
        class_index['bicycle'],
        class_index['electric_bicycle'],
        class_index['motorcycle'],
    ]
    assert supported == (
        'car', 'bicycle', 'electric_bicycle', 'motorcycle')


def test_stock_cityscapes_checkpoint_reports_no_electric_bicycle_support():
    supported = find_supported_project_classes({
        13: 'car',
        17: 'motorcycle',
        18: 'bicycle',
    })

    assert supported == ('car', 'bicycle', 'motorcycle')
    assert 'electric_bicycle' not in supported


def test_relevant_model_labels_report_original_ade20k_ids():
    id2label = {
        1: 'building',
        6: 'road',
        20: 'car',
        80: 'bus',
        83: 'truck',
    }

    matches = find_relevant_model_labels(
        id2label, ['car', 'truck', 'bus', 'road', 'building'])

    assert matches == {
        'car': [(20, 'car')],
        'truck': [(83, 'truck')],
        'bus': [(80, 'bus')],
        'road': [(6, 'road')],
        'building': [(1, 'building')],
    }


def test_bgr_input_is_explicitly_converted_to_rgb():
    bgr = np.asarray([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)

    rgb = convert_image_to_rgb(bgr, 'bgr8')

    assert rgb.flags['C_CONTIGUOUS']
    assert rgb.tolist() == [[[3, 2, 1], [30, 20, 10]]]


def test_raw_prediction_statistics_report_global_and_vehicle_roi():
    id2label = {
        1: 'building',
        6: 'road',
        20: 'car',
        83: 'truck',
    }
    mask = np.asarray([
        [1, 1, 6, 6],
        [1, 20, 20, 6],
        [1, 20, 83, 6],
        [1, 1, 6, 6],
    ], dtype=np.uint8)
    confidence = np.full(mask.shape, 0.8, dtype=np.float32)

    statistics = raw_prediction_statistics(
        mask,
        confidence,
        id2label,
        top_k=3,
        watched_labels=['car', 'truck', 'road', 'building'],
        normalized_roi=[0.25, 0.25, 0.5, 0.5],
    )

    assert statistics['roi_pixels'] == (1, 1, 3, 3)
    assert statistics['global']['watched']['car']['count'] == 3
    assert statistics['roi']['watched']['car']['count'] == 3
    assert statistics['roi']['watched']['truck']['count'] == 1
    assert np.isclose(
        statistics['roi']['watched']['car']['mean_confidence'], 0.8)


def test_segmentation_confidence_becomes_categorical_probabilities():
    logits = segmentation_to_logits(
        class_ids=np.asarray([4, 7, 255]),
        confidences=np.asarray([0.8, 0.8, 0.2]),
        num_classes=len(DEFAULT_CLASSES),
        unknown_probability_floor=0.5,
    )
    probabilities = VoxelMap.softmax(logits)

    assert np.isclose(probabilities[0, 4], 0.8, atol=1e-6)
    assert np.isclose(probabilities[1, 7], 0.8, atol=1e-6)
    unknown = DEFAULT_CLASSES.index('unknown background')
    assert np.isclose(probabilities[2, unknown], 0.8, atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_pointcloud_frame_resolution_prefers_explicit_sensor_frame():
    assert normalize_frame_id('/mid360_link') == 'mid360_link'
    assert resolve_pointcloud_frame(
        '/mid360_link', 'laser_livox', 'base_link') == 'mid360_link'
    assert resolve_pointcloud_frame(
        '', '/velodyne', 'base_link') == 'velodyne'
    assert resolve_pointcloud_frame('', '', '/base_link') == 'base_link'


def test_closed_set_query_selects_segformer_car_cluster_without_features():
    node = object.__new__(GABsvmNode)
    node.voxel_map = VoxelMap(
        voxel_size=0.1,
        K=len(DEFAULT_CLASSES),
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    node.query_min_weight_sum = 2.0
    node.query_min_class_prob = 0.25
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.16
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0] * len(DEFAULT_CLASSES)
    node.query_color_weight = 0.35
    node.query_min_color_score = 0.2
    node.get_robot_position = lambda: None

    points = np.asarray([
        [0.01, 0.01, 0.01],
        [0.11, 0.01, 0.01],
        [0.21, 0.01, 0.01],
    ], dtype=np.float32)
    logits = segmentation_to_logits(
        np.full(3, 4),
        np.full(3, 0.9),
        num_classes=len(DEFAULT_CLASSES),
    )
    for _ in range(4):
        node.voxel_map.update(points, np.ones(3), logits, features=None)

    selected, candidate_count, rejected_count, color_rejected_count = (
        node.select_class_query_target(query_class_idx=4))

    assert selected is not None
    assert selected['voxel_count'] == 3
    assert selected['class_probability'] > 0.25
    assert candidate_count == 3
    assert rejected_count == 0
    assert color_rejected_count == 0
    assert all(
        voxel['feature_512'] is None
        for voxel in node.voxel_map.voxels.values()
    )


def test_closed_set_query_uses_observation_distribution_not_diluted_posterior():
    node = object.__new__(GABsvmNode)
    node.num_classes = len(DEFAULT_CLASSES)
    node.vocab = list(DEFAULT_CLASSES)
    node.voxel_map = VoxelMap(
        voxel_size=0.1,
        K=node.num_classes,
        evidence_decay=1.0,
        max_frame_evidence=3.0,
    )
    node.query_min_weight_sum = 2.0
    node.query_min_class_prob = 0.25
    node.query_require_class_argmax = True
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.16
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0] * node.num_classes
    node.query_color_weight = 0.35
    node.query_min_color_score = 0.2
    node.get_robot_position = lambda: None

    points = np.asarray([
        [0.01, 0.01, 0.01],
        [0.11, 0.01, 0.01],
        [0.21, 0.01, 0.01],
    ], dtype=np.float32)
    logits = segmentation_to_logits(
        np.full(3, 4),
        np.full(3, 0.9),
        num_classes=node.num_classes,
    )
    reliability = np.ones(3, dtype=np.float32)
    node.voxel_map.update(points, reliability, logits)
    node.voxel_map.update(points, reliability, logits)

    keys = [node.voxel_map.get_voxel_indices(point) for point in points]
    assert max(
        node.voxel_map.get_probabilities(key)[4] for key in keys
    ) < node.query_min_class_prob

    selected, candidate_count, rejected_count, _ = (
        node.select_class_query_target(4))

    assert selected is not None
    assert selected['class_probability'] > 0.89
    assert candidate_count == 3
    assert rejected_count == 0


def test_color_attribute_selects_blue_car_cluster():
    node = object.__new__(GABsvmNode)
    node.voxel_map = VoxelMap(
        voxel_size=0.1,
        K=len(DEFAULT_CLASSES),
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    node.query_min_weight_sum = 2.0
    node.query_min_class_prob = 0.25
    node.query_min_color_score = 0.2
    node.query_color_weight = 0.35
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.16
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0] * len(DEFAULT_CLASSES)
    node.get_robot_position = lambda: None

    points = np.asarray([
        [0.01, 0.01, 0.01],
        [0.11, 0.01, 0.01],
        [0.21, 0.01, 0.01],
        [3.01, 0.01, 0.01],
        [3.11, 0.01, 0.01],
        [3.21, 0.01, 0.01],
    ], dtype=np.float32)
    colors = np.asarray(
        [[20.0, 60.0, 220.0]] * 3 + [[220.0, 30.0, 20.0]] * 3,
        dtype=np.float32,
    )
    logits = segmentation_to_logits(
        np.full(6, 4),
        np.full(6, 0.9),
        num_classes=len(DEFAULT_CLASSES),
    )
    for _ in range(4):
        node.voxel_map.update(
            points,
            np.ones(6),
            logits,
            colors=colors,
        )

    selected, _, _, color_rejected_count = (
        node.select_class_query_target(4, query_color='blue'))

    assert selected is not None
    assert selected['pos'][0] < 1.0
    assert selected['color_score'] > 0.5
    assert color_rejected_count == 3


def test_cluster_color_rejects_white_accessories_on_red_truck():
    node = object.__new__(GABsvmNode)
    node.voxel_map = VoxelMap(
        voxel_size=0.1,
        K=len(DEFAULT_CLASSES),
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    node.query_min_weight_sum = 2.0
    node.query_min_class_prob = 0.25
    node.query_require_class_argmax = True
    node.query_min_color_score = 0.2
    node.query_min_color_support_ratio = 0.3
    node.query_color_weight = 0.35
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.16
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0] * len(DEFAULT_CLASSES)
    node.get_robot_position = lambda: None

    red_points = np.asarray([
        [0.01 + 0.1 * index, 0.01, 0.61]
        for index in range(12)
    ], dtype=np.float32)
    white_points = np.asarray([
        [3.01 + 0.1 * index, 0.01, 0.61]
        for index in range(10)
    ], dtype=np.float32)
    points = np.vstack([red_points, white_points])

    # A red vehicle has three white accessory voxels (headlights/plate). Under
    # per-voxel filtering those three voxels formed a false white instance.
    colors = np.asarray(
        [[235.0, 235.0, 235.0]] * 3
        + [[220.0, 30.0, 20.0]] * 9
        + [[235.0, 235.0, 235.0]] * 6
        + [[30.0, 30.0, 30.0]] * 4,
        dtype=np.float32,
    )
    logits = segmentation_to_logits(
        np.full(len(points), 5),
        np.asarray([0.97] * 12 + [0.78] * 10),
        num_classes=len(DEFAULT_CLASSES),
    )
    for _ in range(4):
        node.voxel_map.update(
            points,
            np.ones(len(points)),
            logits,
            colors=colors,
        )

    selected, _, _, low_color_voxels = (
        node.select_class_query_target(5, query_color='white'))

    assert selected is not None
    assert selected['pos'][0] > 2.5
    assert selected['color_score'] > 0.4
    assert selected['color_support_ratio'] > 0.5
    assert low_color_voxels == 13
    assert (
        node._last_cluster_selection_diagnostics['color_rejected_clusters']
        >= 1
    )


def test_color_diagnostics_say_when_class_gate_prevents_color_evaluation():
    node = object.__new__(GABsvmNode)
    node.num_classes = len(DEFAULT_CLASSES)
    node.vocab = list(DEFAULT_CLASSES)
    node.voxel_map = VoxelMap(
        voxel_size=0.1,
        K=node.num_classes,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    node.query_min_weight_sum = 2.0
    node.query_min_class_prob = 0.25
    node.query_require_class_argmax = True
    node.query_min_color_score = 0.2
    node.query_color_weight = 0.35
    node.query_max_candidates = 100
    node.query_cluster_radius_m = 0.16
    node.query_cluster_min_voxels = 3
    node.query_cluster_min_evidence = 3.0
    node.query_cluster_support_weight = 0.08
    node.query_distance_weight = 0.0
    node.query_class_max_extent_m = [100.0] * node.num_classes
    node.get_robot_position = lambda: None

    points = np.asarray([
        [0.01, 0.01, 0.01],
        [0.11, 0.01, 0.01],
        [0.21, 0.01, 0.01],
    ], dtype=np.float32)
    road_logits = segmentation_to_logits(
        np.zeros(3, dtype=np.uint8),
        np.full(3, 0.9),
        num_classes=node.num_classes,
    )
    for _ in range(4):
        node.voxel_map.update(
            points,
            np.ones(3),
            road_logits,
            colors=np.asarray([[20.0, 60.0, 220.0]] * 3),
        )

    selected, _, _, color_rejected_count = (
        node.select_class_query_target(5, query_color='blue'))
    diagnostics = node.format_class_query_diagnostics()

    assert selected is None
    assert color_rejected_count == 0
    assert '颜色筛选=未执行(类别门控无候选)' in diagnostics
