import numpy as np
import pytest
from std_msgs.msg import Header

from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
from semantic_mapping.runtime.segformer_core import (
    aggregate_project_probability_tensor,
)
from semantic_mapping.runtime import semantic_posterior
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


def test_float32_clip_frame_is_atomic_and_keeps_header():
    header = Header()
    header.stamp.sec = 8
    header.stamp.nanosec = 42
    header.frame_id = 'camera'
    frame = np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7)

    message = semantic_posterior.float32_array_to_image(frame, header)
    decoded = semantic_posterior.float32_image_to_array(
        message, expected_channels=7)

    assert message.header == header
    assert message.encoding == '32FC7'
    np.testing.assert_array_equal(decoded, frame)


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


def test_same_image_header_normalizes_frame_ids_and_checks_stamp():
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


def _callback_observation(backend):
    """Use real ROS image encodings, without creating a running ROS node."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from cv_bridge import CvBridge
    from sensor_msgs.msg import PointCloud2

    bridge = CvBridge()
    node = object.__new__(GABsvmNode)
    node.frame_count = 0
    node.frame_stride = 1
    node.num_classes = 2
    node.feat_dim = 1
    node.grid_rows = node.grid_cols = 2
    node.semantic_capability = object()
    node.segformer_unknown_probability_floor = 0.01
    node.bridge = bridge
    logger = Mock()
    node.get_logger = lambda: logger
    node._process_semantic_frame = Mock()
    source = bridge.cv2_to_imgmsg(
        np.full((2, 2, 3), 80, dtype=np.uint8), encoding='rgb8')
    if backend == 'hard_mask':
        semantic = bridge.cv2_to_imgmsg(
            np.zeros((2, 2), dtype=np.uint8), encoding='mono8')
        confidence = bridge.cv2_to_imgmsg(
            np.full((2, 2), 0.8, dtype=np.float32), encoding='32FC1')
        products = [semantic, confidence, source]
        callback = node.segformer_sync_callback
    elif backend == 'full_posterior':
        semantic = posterior_array_to_image(
            np.full((2, 2, 2), 0.5, dtype=np.float32), Header())
        products = [semantic, source]
        callback = node.segformer_posterior_sync_callback
    else:
        semantic = semantic_posterior.float32_array_to_image(
            np.ones((2, 2, 3), dtype=np.float32), Header())
        products = [semantic, source]
        callback = node.clip_sync_callback
    for product in products:
        product.header.stamp.sec = 1
        product.header.frame_id = 'camera'
    # LiDAR is intentionally offset: approximate sensor synchronization remains valid.
    cloud = PointCloud2()
    cloud.header.stamp.sec = 1
    cloud.header.stamp.nanosec = 40_000_000
    return SimpleNamespace(
        node=node, logger=logger, cloud=cloud, products=products,
        invoke=lambda: callback(cloud, *products),
    )


@pytest.mark.parametrize('backend', ['hard_mask', 'full_posterior', 'clip'])
def test_semantic_callbacks_accept_same_image_with_offset_lidar(backend):
    observation = _callback_observation(backend)
    # Leading slash normalization must still be accepted.
    observation.products[-1].header.frame_id = '/camera'

    observation.invoke()

    observation.node._process_semantic_frame.assert_called_once()
    cloud, height, width, lookup = (
        observation.node._process_semantic_frame.call_args.args)
    assert cloud is observation.cloud
    assert (height, width) == (2, 2)
    logits, features, colors = lookup(np.array([0]), np.array([0]))
    assert logits.shape == (1, 2)
    np.testing.assert_array_equal(colors, [[80, 80, 80]])
    assert (features is None) == (backend != 'clip')
    observation.logger.error.assert_not_called()


@pytest.mark.parametrize('backend', ['hard_mask', 'full_posterior', 'clip'])
@pytest.mark.parametrize('product_index', [0, -1])
@pytest.mark.parametrize('mismatch', ['stamp', 'frame'])
def test_semantic_callbacks_reject_different_image_headers(
    backend, product_index, mismatch,
):
    observation = _callback_observation(backend)
    header = observation.products[product_index].header
    if mismatch == 'stamp':
        header.stamp.nanosec = 20_000_000  # Within supported approximate-sync slop.
    else:
        header.frame_id = 'other_camera'

    observation.invoke()

    observation.node._process_semantic_frame.assert_not_called()
    observation.logger.warn.assert_called_once()
    observation.logger.error.assert_not_called()


@pytest.mark.parametrize('mismatch', ['stamp', 'frame'])
def test_hard_mask_rejects_confidence_from_another_image(mismatch):
    observation = _callback_observation('hard_mask')
    header = observation.products[1].header
    if mismatch == 'stamp':
        header.stamp.nanosec = 20_000_000
    else:
        header.frame_id = 'other_camera'

    observation.invoke()

    observation.node._process_semantic_frame.assert_not_called()
    observation.logger.warn.assert_called_once()
    observation.logger.error.assert_not_called()
