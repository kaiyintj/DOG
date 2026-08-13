import numpy as np
import pytest
from std_msgs.msg import Header

from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
from semantic_mapping.runtime.segformer_core import (
    aggregate_project_probability_tensor,
)
from semantic_mapping.runtime.semantic_posterior import (
    aggregate_project_probabilities,
    posterior_array_to_image,
    posterior_image_to_array,
    posterior_probabilities_to_logits,
    sample_posterior_bilinear,
)
from semantic_mapping.runtime.voxel_map import VoxelMap


def test_probability_aggregation_happens_before_argmax():
    # Raw road and sidewalk are individually below car, but their project
    # probability mass is larger after both map to navigation road.
    raw = np.asarray([[0.32, 0.29, 0.39]], dtype=np.float32)
    projected = aggregate_project_probabilities(
        raw,
        project_lookup=[0, 0, 1],
        num_project_classes=2,
    )

    np.testing.assert_allclose(projected, [[0.61, 0.39]], atol=1e-7)
    assert int(np.argmax(raw, axis=1)[0]) == 2
    assert int(np.argmax(projected, axis=1)[0]) == 0


def test_project_probability_aggregation_conserves_mass():
    raw = np.asarray([
        [[0.10, 0.20, 0.25, 0.15, 0.30]],
        [[0.30, 0.10, 0.05, 0.45, 0.10]],
    ], dtype=np.float32)
    projected = aggregate_project_probabilities(
        raw,
        project_lookup=[0, 0, 1, 2, 2],
        num_project_classes=4,
    )

    np.testing.assert_allclose(projected.sum(axis=-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(
        projected[0, 0], [0.30, 0.25, 0.45, 0.0], atol=1e-7)


def test_runtime_tensor_aggregation_matches_numpy_reference():
    torch = pytest.importorskip('torch')
    raw = np.asarray([[[[0.32]], [[0.29]], [[0.39]]]], dtype=np.float32)
    lookup = np.asarray([0, 0, 1], dtype=np.int64)

    projected_tensor = aggregate_project_probability_tensor(
        torch.from_numpy(raw),
        torch.from_numpy(lookup),
        num_project_classes=2,
    )
    projected_numpy = aggregate_project_probabilities(
        np.moveaxis(raw, 1, -1),
        lookup,
        num_project_classes=2,
    )

    actual = np.moveaxis(projected_tensor.numpy(), 1, -1)
    np.testing.assert_allclose(actual, projected_numpy, atol=1e-7)


def test_identity_project_mapping_preserves_custom_checkpoint_posterior():
    rng = np.random.default_rng(7)
    raw = rng.random((3, 4, 13), dtype=np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)

    projected = aggregate_project_probabilities(
        raw,
        project_lookup=np.arange(13),
        num_project_classes=13,
    )

    np.testing.assert_allclose(projected, raw, atol=1e-7)


def test_fp16_posterior_image_is_atomic_and_keeps_header():
    header = Header()
    header.stamp.sec = 12
    header.stamp.nanosec = 345
    header.frame_id = 'camera_color_optical_frame'
    posterior = np.asarray([
        [[0.61, 0.39], [0.25, 0.75]],
        [[0.50, 0.50], [0.90, 0.10]],
    ], dtype=np.float32)

    message = posterior_array_to_image(posterior, header)
    decoded = posterior_image_to_array(message, expected_classes=2)

    assert message.header.stamp.sec == 12
    assert message.header.stamp.nanosec == 345
    assert message.header.frame_id == 'camera_color_optical_frame'
    assert message.encoding == '16FC2'
    assert message.step == 8
    np.testing.assert_allclose(decoded, posterior, atol=5e-4)


def test_posterior_decoder_rejects_wrong_class_count_and_step():
    posterior = np.full((2, 3, 4), 0.25, dtype=np.float32)
    message = posterior_array_to_image(posterior, Header())

    with pytest.raises(ValueError, match='expected 13'):
        posterior_image_to_array(message, expected_classes=13)

    message.step += 2
    with pytest.raises(ValueError, match='tightly packed'):
        posterior_image_to_array(message, expected_classes=4)


def test_native_grid_bilinear_sampling_uses_source_pixel_centers():
    grid = np.asarray([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ], dtype=np.float32)
    sampled = sample_posterior_bilinear(
        grid,
        pixel_u=np.asarray([0, 3, 1.5]),
        pixel_v=np.asarray([0, 3, 1.5]),
        source_width=4,
        source_height=4,
    )

    np.testing.assert_allclose(sampled[0], [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sampled[1], [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sampled[2], [0.5, 0.5], atol=1e-7)


def test_probability_to_logits_round_trip_preserves_class_competition():
    posterior = np.asarray([
        [0.05, 0.48, 0.10, 0.35, 0.02],
        [0.60, 0.10, 0.10, 0.10, 0.10],
    ], dtype=np.float32)
    logits = posterior_probabilities_to_logits(posterior)
    recovered = VoxelMap.softmax(logits)

    np.testing.assert_allclose(recovered, posterior, atol=1e-6)
    assert int(np.argmax(recovered[0])) == 1
    assert np.isclose(recovered[0, 3], 0.35, atol=1e-6)


def test_ga_bsvm_rejects_cross_frame_posterior_and_rgb_headers():
    first = Header()
    first.stamp.sec = 5
    first.frame_id = '/camera'
    same = Header()
    same.stamp.sec = 5
    same.frame_id = 'camera'
    other = Header()
    other.stamp.sec = 6
    other.frame_id = 'camera'

    assert GABsvmNode._same_image_header(first, same)
    assert not GABsvmNode._same_image_header(first, other)
