import numpy as np
import pytest
from scipy.spatial import cKDTree

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
from semantic_mapping.runtime.semantic_projection import (
    project_points_pinhole,
    scale_camera_matrix,
)
from semantic_mapping.runtime.voxel_map import VoxelMap


def test_motion_reliability_has_expected_analytic_values():
    stationary = compute_motion_reliability(
        [0.0], [9.81], 2.0, 3.0, 9.81, 0.2)
    angular_scale = compute_motion_reliability(
        [2.0], [9.81], 2.0, 3.0, 9.81, 0.2)
    both_scales = compute_motion_reliability(
        [2.0], [12.81], 2.0, 3.0, 9.81, 0.2)

    assert stationary == pytest.approx((1.0, 0.0, 0.0))
    assert angular_scale[0] == pytest.approx(np.exp(-0.5))
    assert both_scales[0] == pytest.approx(np.exp(-1.0))


def test_motion_reliability_clips_and_rejects_invalid_samples():
    reliability, _, _ = compute_motion_reliability(
        [100.0], [100.0], 2.0, 3.0, 9.81, 0.2)
    assert reliability == 0.2
    with pytest.raises(ValueError, match='must not be empty'):
        compute_motion_reliability([], [], 2.0, 3.0, 9.81, 0.2)
    with pytest.raises(ValueError, match='equal length'):
        compute_motion_reliability([0.0], [9.81, 9.81], 2.0, 3.0, 9.81, 0.2)


def test_density_counts_self_and_respects_three_dimensional_radius():
    points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, 0.0, 0.31],
    ])
    density = compute_local_point_density(points, 0.3)
    np.testing.assert_array_equal(density, [2, 2, 1])

    reliability = compute_density_reliability(density, 8.0)
    np.testing.assert_allclose(reliability, 1.0 - np.exp(-density / 8.0))


def test_range_reliability_has_expected_analytic_values():
    ranges = np.asarray([0.0, 20.0, 40.0])
    reliability = compute_range_reliability(ranges, 20.0)
    np.testing.assert_allclose(reliability, [1.0, np.exp(-1), np.exp(-4)])


def test_view_radius_and_reliability_match_center_edge_and_corner():
    radius = compute_normalized_view_radius(
        [320, 640, 640],
        [240, 240, 480],
        640,
        480,
    )
    np.testing.assert_allclose(radius, [0.0, 1.0 / np.sqrt(2.0), 1.0])
    reliability = compute_view_reliability(radius, 0.4)
    np.testing.assert_allclose(reliability, [1.0, 0.8, 0.6])


def test_semantic_reliability_matches_legacy_softmax_and_entropy():
    logits = np.asarray([
        [0.0, 0.0, 0.0],
        [4.0, 1.0, -2.0],
    ], dtype=np.float32)
    probabilities, entropy, reliability = compute_semantic_reliability(
        logits, 3, 0.15)
    legacy_probabilities = VoxelMap.softmax(logits)
    legacy_entropy = -np.sum(
        legacy_probabilities
        * np.log(np.clip(legacy_probabilities, 1e-10, 1.0)),
        axis=1,
    )
    legacy_reliability = 0.15 + 0.85 * (
        1.0 - legacy_entropy / np.log(3))

    np.testing.assert_allclose(probabilities, legacy_probabilities)
    np.testing.assert_allclose(entropy, legacy_entropy)
    np.testing.assert_allclose(reliability, legacy_reliability)
    assert reliability[0] == pytest.approx(0.15)


def test_combined_reliability_matches_legacy_formula_and_float32():
    density = np.asarray([0.2, 0.5], dtype=np.float64)
    range_ = np.asarray([0.9, 0.7], dtype=np.float64)
    view = np.asarray([1.0, 0.8], dtype=np.float64)
    semantic = np.asarray([0.6, 0.4], dtype=np.float64)
    actual = combine_reliability(0.75, density, range_, view, semantic)
    expected = (0.75 * density * range_ * view * semantic).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_all_point_factors_match_the_previous_inline_formulas():
    points = np.asarray([
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ], dtype=np.float32)
    pixel_u = np.asarray([320, 500, 10])
    pixel_v = np.asarray([240, 400, 20])
    logits = np.asarray([
        [3.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 1.5, 0.5],
    ], dtype=np.float32)

    density = compute_local_point_density(points, 0.3)
    density_reliability = compute_density_reliability(density, 8.0)
    ranges = np.linalg.norm(points, axis=1)
    range_reliability = compute_range_reliability(ranges, 20.0)
    radius = compute_normalized_view_radius(
        pixel_u, pixel_v, 640, 480)
    view_reliability = compute_view_reliability(radius, 0.4)
    probabilities, entropy, semantic_reliability = (
        compute_semantic_reliability(logits, 3, 0.15))
    combined = combine_reliability(
        0.75,
        density_reliability,
        range_reliability,
        view_reliability,
        semantic_reliability,
    )

    legacy_density = np.asarray([
        len(neighbors)
        for neighbors in cKDTree(points).query_ball_point(points, r=0.3)
    ])
    legacy_density_reliability = 1.0 - np.exp(-legacy_density / 8.0)
    legacy_range_reliability = np.exp(-np.square(ranges / 20.0))
    normalized_u = (pixel_u - 0.5 * 640) / max(0.5 * 640, 1.0)
    normalized_v = (pixel_v - 0.5 * 480) / max(0.5 * 480, 1.0)
    legacy_radius = np.clip(
        np.sqrt(normalized_u ** 2 + normalized_v ** 2) / np.sqrt(2.0),
        0.0,
        1.0,
    )
    legacy_view_reliability = np.clip(
        1.0 - 0.4 * legacy_radius ** 2, 0.05, 1.0)
    legacy_probabilities = VoxelMap.softmax(logits)
    legacy_entropy = -np.sum(
        legacy_probabilities
        * np.log(np.clip(legacy_probabilities, 1e-10, 1.0)),
        axis=1,
    )
    legacy_semantic_reliability = 0.15 + 0.85 * (
        1.0 - legacy_entropy / np.log(3))
    legacy_combined = (
        0.75
        * legacy_density_reliability
        * legacy_range_reliability
        * legacy_view_reliability
        * legacy_semantic_reliability
    ).astype(np.float32)

    np.testing.assert_array_equal(density, legacy_density)
    np.testing.assert_allclose(
        density_reliability, legacy_density_reliability)
    np.testing.assert_allclose(range_reliability, legacy_range_reliability)
    np.testing.assert_allclose(radius, legacy_radius)
    np.testing.assert_allclose(view_reliability, legacy_view_reliability)
    np.testing.assert_allclose(probabilities, legacy_probabilities)
    np.testing.assert_allclose(entropy, legacy_entropy)
    np.testing.assert_allclose(
        semantic_reliability, legacy_semantic_reliability)
    np.testing.assert_array_equal(combined, legacy_combined)


def test_projection_helpers_preserve_legacy_integer_pixels_and_scaling():
    camera_matrix = np.asarray([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 40.0],
        [0.0, 0.0, 1.0],
    ])
    scaled = scale_camera_matrix(camera_matrix, 100, 80, 200, 160)
    np.testing.assert_allclose(
        scaled,
        [[200.0, 0.0, 100.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]],
    )

    points = np.asarray([
        [0.015, 0.015, 1.0],
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 1.0],
    ])
    valid, pixel_u, pixel_v = project_points_pinhole(
        points,
        np.eye(4),
        camera_matrix,
        image_height=80,
        image_width=100,
    )
    np.testing.assert_array_equal(pixel_u, [51, 50, 150])
    np.testing.assert_array_equal(pixel_v, [41, 40, 40])
    np.testing.assert_array_equal(valid, [True, False, False])
