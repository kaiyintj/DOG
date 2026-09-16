#!/usr/bin/env python3
"""Loss-minimizing transport helpers for project-space semantic posteriors."""

import re

import numpy as np
from sensor_msgs.msg import Image


_POSTERIOR_ENCODING = re.compile(r'^16FC([1-9][0-9]*)$')
_FLOAT32_ENCODING = re.compile(r'^32FC([1-9][0-9]*)$')


def float32_array_to_image(values, header):
    """Encode one finite ``H x W x C`` array as a Header-bearing Image."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] <= 0:
        raise ValueError('float32 image must have shape (height, width, channels)')
    if not np.all(np.isfinite(array)):
        raise ValueError('float32 image values must be finite')
    height, width, channel_count = array.shape
    little_endian = np.ascontiguousarray(array.astype('<f4', copy=False))
    message = Image()
    message.header = header
    message.height = int(height)
    message.width = int(width)
    message.encoding = f'32FC{channel_count}'
    message.is_bigendian = 0
    message.step = int(width * channel_count * np.dtype(np.float32).itemsize)
    message.data = little_endian.tobytes(order='C')
    return message


def float32_image_to_array(message, expected_channels=None):
    """Decode and validate one tightly packed Header-bearing float32 array."""
    match = _FLOAT32_ENCODING.fullmatch(str(message.encoding))
    if match is None:
        raise ValueError(
            f'float32 image encoding must be 32FC<C>, got {message.encoding!r}')
    channel_count = int(match.group(1))
    if (
        expected_channels is not None
        and channel_count != int(expected_channels)
    ):
        raise ValueError(
            f'float32 image has {channel_count} channels, '
            f'expected {expected_channels}')
    height = int(message.height)
    width = int(message.width)
    if height <= 0 or width <= 0:
        raise ValueError('float32 image dimensions must be positive')
    expected_step = width * channel_count * np.dtype(np.float32).itemsize
    if int(message.step) != expected_step:
        raise ValueError(
            f'float32 image step={message.step}, expected {expected_step}')
    payload = bytes(message.data)
    expected_bytes = height * expected_step
    if len(payload) != expected_bytes:
        raise ValueError(
            f'float32 image payload has {len(payload)} bytes, '
            f'expected {expected_bytes}')
    dtype = np.dtype('>f4' if int(message.is_bigendian) else '<f4')
    array = np.frombuffer(payload, dtype=dtype).reshape(
        height, width, channel_count).astype(np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError('float32 image values must be finite')
    return array


def aggregate_project_probabilities(
    raw_probabilities,
    project_lookup,
    num_project_classes,
    class_axis=-1,
):
    """
    Sum a raw checkpoint posterior into the project semantic ontology.

    Mapping probabilities before taking ``argmax`` preserves probability mass
    when several checkpoint labels map to one navigation class, for example
    Cityscapes ``road`` and ``sidewalk`` both mapping to project ``road``.
    """
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    lookup = np.asarray(project_lookup, dtype=np.int64).reshape(-1)
    if probabilities.ndim == 0:
        raise ValueError('raw_probabilities must have at least one dimension')
    axis = int(class_axis)
    if axis < 0:
        axis += probabilities.ndim
    if axis < 0 or axis >= probabilities.ndim:
        raise ValueError('class_axis is out of range')
    if probabilities.shape[axis] != lookup.size:
        raise ValueError(
            'project_lookup length must match the raw class dimension')
    if int(num_project_classes) <= 0:
        raise ValueError('num_project_classes must be positive')
    if np.any(lookup < 0) or np.any(lookup >= int(num_project_classes)):
        raise ValueError('project_lookup contains an invalid project class id')
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError('raw probabilities must be finite and non-negative')

    moved = np.moveaxis(probabilities, axis, -1)
    raw_sums = moved.sum(axis=-1, keepdims=True)
    if np.any(raw_sums <= 0.0):
        raise ValueError('raw probability vectors must have positive mass')
    normalized = moved / raw_sums
    projected = np.zeros(
        normalized.shape[:-1] + (int(num_project_classes),),
        dtype=np.float64,
    )
    for raw_class, project_class in enumerate(lookup):
        projected[..., project_class] += normalized[..., raw_class]
    projected /= np.maximum(projected.sum(axis=-1, keepdims=True), 1e-12)
    return np.moveaxis(projected.astype(np.float32), -1, axis)


def normalize_posterior(probabilities):
    """Return finite float32 categorical probabilities with unit mass."""
    posterior = np.asarray(probabilities, dtype=np.float32)
    if posterior.ndim < 1 or posterior.shape[-1] <= 0:
        raise ValueError('posterior must end with a non-empty class dimension')
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
        raise ValueError('posterior must be finite and non-negative')
    mass = posterior.sum(axis=-1, keepdims=True, dtype=np.float32)
    if np.any(mass <= 0.0):
        raise ValueError('posterior vectors must have positive mass')
    return posterior / mass


def posterior_array_to_image(probabilities, header):
    """Encode an ``H x W x K`` posterior as one Header-bearing FP16 image."""
    posterior = normalize_posterior(probabilities)
    if posterior.ndim != 3:
        raise ValueError(
            'posterior image must have shape (height, width, classes)')
    height, width, class_count = posterior.shape
    little_endian = np.ascontiguousarray(posterior.astype('<f2'))

    message = Image()
    message.header = header
    message.height = int(height)
    message.width = int(width)
    message.encoding = f'16FC{class_count}'
    message.is_bigendian = 0
    message.step = int(width * class_count * np.dtype(np.float16).itemsize)
    message.data = little_endian.tobytes(order='C')
    return message


def posterior_image_class_count(message):
    """Return the class count encoded in a semantic posterior Image."""
    match = _POSTERIOR_ENCODING.fullmatch(str(message.encoding))
    if match is None:
        raise ValueError(
            'posterior encoding must be 16FC<K>, '
            f'got {message.encoding!r}')
    return int(match.group(1))


def posterior_image_to_array(message, expected_classes=None):
    """Decode and validate one FP16 posterior Image into float32 H x W x K."""
    class_count = posterior_image_class_count(message)
    if expected_classes is not None and class_count != int(expected_classes):
        raise ValueError(
            f'posterior has {class_count} classes, '
            f'expected {expected_classes}')
    height = int(message.height)
    width = int(message.width)
    if height <= 0 or width <= 0:
        raise ValueError('posterior image dimensions must be positive')
    expected_step = width * class_count * np.dtype(np.float16).itemsize
    if int(message.step) != expected_step:
        raise ValueError(
            f'posterior step={message.step}, expected tightly packed '
            f'step={expected_step}')
    payload = bytes(message.data)
    expected_bytes = height * expected_step
    if len(payload) != expected_bytes:
        raise ValueError(
            f'posterior payload has {len(payload)} bytes, '
            f'expected {expected_bytes}')
    dtype = np.dtype('>f2' if int(message.is_bigendian) else '<f2')
    posterior = np.frombuffer(payload, dtype=dtype).reshape(
        height, width, class_count)
    return normalize_posterior(posterior.astype(np.float32))


def sample_posterior_bilinear(
    posterior,
    pixel_u,
    pixel_v,
    source_width,
    source_height,
):
    """Sample a native-resolution posterior at source-image pixel centers."""
    grid = normalize_posterior(posterior)
    if grid.ndim != 3:
        raise ValueError(
            'posterior grid must have shape (height, width, classes)')
    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError('source image dimensions must be positive')
    u = np.asarray(pixel_u, dtype=np.float64)
    v = np.asarray(pixel_v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError('pixel_u and pixel_v must have identical shapes')
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError('pixel coordinates must be finite')

    grid_height, grid_width, _ = grid.shape
    x = (u + 0.5) * grid_width / source_width - 0.5
    y = (v + 0.5) * grid_height / source_height - 0.5
    x = np.clip(x, 0.0, grid_width - 1.0)
    y = np.clip(y, 0.0, grid_height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid_width - 1)
    y1 = np.minimum(y0 + 1, grid_height - 1)
    wx = (x - x0)[..., np.newaxis]
    wy = (y - y0)[..., np.newaxis]

    top = (1.0 - wx) * grid[y0, x0] + wx * grid[y0, x1]
    bottom = (1.0 - wx) * grid[y1, x0] + wx * grid[y1, x1]
    sampled = (1.0 - wy) * top + wy * bottom
    return normalize_posterior(sampled)


def posterior_probabilities_to_logits(probabilities, epsilon=1e-6):
    """Convert probabilities to stable logits without changing their ratios."""
    posterior = normalize_posterior(probabilities)
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 1.0:
        raise ValueError('epsilon must lie in (0, 1)')
    logits = np.log(np.clip(posterior, epsilon, 1.0))
    return logits.astype(np.float32)
