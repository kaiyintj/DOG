import numpy as np

from semantic_mapping.voxel_map import VoxelMap


def test_repeated_consistent_observations_increase_class_probability():
    voxel_map = VoxelMap(
        voxel_size=0.2,
        K=3,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
    )
    points = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
    reliability = np.ones(1, dtype=np.float32)
    logits = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)

    key = voxel_map.get_voxel_indices(points[0])
    initial_probability = voxel_map.get_probabilities(key)[2]
    for _ in range(8):
        voxel_map.update(points, reliability, logits)

    probabilities = voxel_map.get_probabilities(key)
    assert probabilities[2] > initial_probability
    assert probabilities[2] > 0.75
    assert int(np.argmax(probabilities)) == 2


def test_evidence_probabilities_exclude_symmetric_prior():
    voxel_map = VoxelMap(
        voxel_size=0.2,
        K=12,
        evidence_decay=1.0,
        max_frame_evidence=3.0,
    )
    point = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
    observed_probabilities = np.full(12, 0.1 / 11.0, dtype=np.float32)
    observed_probabilities[4] = 0.9
    voxel_map.update(
        point,
        np.asarray([1.0], dtype=np.float32),
        np.log(observed_probabilities)[None, :],
    )

    key = voxel_map.get_voxel_indices(point[0])
    assert voxel_map.get_probabilities(key)[4] < 0.25
    assert voxel_map.get_evidence_probabilities(key)[4] > 0.89


def test_one_dense_frame_has_bounded_evidence():
    dense_map = VoxelMap(K=3, evidence_decay=1.0, max_frame_evidence=1.0)
    sparse_map = VoxelMap(K=3, evidence_decay=1.0, max_frame_evidence=1.0)
    dense_points = np.tile([[0.01, 0.01, 0.01]], (100, 1)).astype(np.float32)
    sparse_points = dense_points[:1]
    logits = np.array([0.0, 4.0, 0.0], dtype=np.float32)

    dense_map.update(dense_points, np.ones(100), logits)
    sparse_map.update(sparse_points, np.ones(1), logits)

    key = dense_map.get_voxel_indices(dense_points[0])
    np.testing.assert_allclose(
        dense_map.voxels[key]['alpha'],
        sparse_map.voxels[key]['alpha'],
        rtol=1e-6,
        atol=1e-6,
    )


def test_uniform_semantics_remain_uncertain_even_with_more_evidence():
    voxel_map = VoxelMap(
        K=4,
        evidence_decay=1.0,
        max_frame_evidence=1.0,
        uncertainty_entropy_weight=0.7,
    )
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    for _ in range(20):
        voxel_map.update(points, np.ones(1), np.zeros((1, 4)))

    key = voxel_map.get_voxel_indices(points[0])
    probabilities = voxel_map.get_probabilities(key)
    _, _, uncertainty = voxel_map.get_uncertainty(key)
    np.testing.assert_allclose(probabilities, np.full(4, 0.25), atol=1e-6)
    assert uncertainty >= 0.7
    assert voxel_map.get_confidence(key) <= 0.3


def test_open_vocabulary_features_are_normalized_after_fusion():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)
    feature_a = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    feature_b = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)

    voxel_map.update(points, np.ones(1), logits, feature_a)
    voxel_map.update(points, np.ones(1), logits, feature_b)

    key = voxel_map.get_voxel_indices(points[0])
    feature = voxel_map.voxels[key]['feature_512']
    assert np.isclose(np.linalg.norm(feature), 1.0, atol=1e-6)
    assert feature[0] > 0.0
    assert feature[1] > 0.0


def test_point_colors_are_fused_per_voxel():
    voxel_map = VoxelMap(K=2, evidence_decay=1.0)
    points = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
    logits = np.array([[4.0, 0.0]], dtype=np.float32)

    voxel_map.update(
        points,
        np.ones(1),
        logits,
        colors=np.array([[0.0, 0.0, 255.0]], dtype=np.float32),
    )
    voxel_map.update(
        points,
        np.ones(1),
        logits,
        colors=np.array([[0.0, 0.0, 200.0]], dtype=np.float32),
    )

    key = voxel_map.get_voxel_indices(points[0])
    color = voxel_map.voxels[key]['color_rgb']
    assert color[2] > 0.85
    assert color[0] < 0.05
    assert color[1] < 0.05
