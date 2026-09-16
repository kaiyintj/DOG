#!/usr/bin/env python3
"""Shared semantic classes, query aliases and color attribute scoring."""

import colorsys

import numpy as np

from semantic_mapping.runtime.semantic_profile import (
    COLOR_ALIASES,
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    DEFAULT_QUERY_MAX_EXTENTS_M,
    DEFAULT_SEMANTIC_COSTS,
    QUERY_CLASS_ALIASES,
    SEGFORMER_LABEL_ALIASES,
    normalize_label,
    parse_semantic_query,
)


__all__ = (
    'COLOR_ALIASES', 'DEFAULT_CLASSES', 'DEFAULT_CLASS_COLORS',
    'DEFAULT_QUERY_MAX_EXTENTS_M', 'DEFAULT_SEMANTIC_COSTS',
    'QUERY_CLASS_ALIASES', 'SEGFORMER_LABEL_ALIASES', 'normalize_label',
    'parse_semantic_query', 'color_membership_score', 'convert_image_to_rgb',
    'resolve_clip_model_name', 'should_run_inference',
)


def should_run_inference(now_ns, last_ns, interval_sec):
    """Return whether inference is due, rebasing after a ROS clock rewind."""
    now_ns = int(now_ns)
    interval_ns = max(0, int(float(interval_sec) * 1e9))
    if last_ns is None or now_ns < int(last_ns):
        return True, now_ns
    if now_ns - int(last_ns) < interval_ns:
        return False, int(last_ns)
    return True, now_ns


def resolve_clip_model_name(model_name, pretrained, available_models=None):
    """
    Return the CLIP variant whose activation matches the pretrained weights.

    OpenAI CLIP weights were trained with QuickGELU. ``open_clip`` defaults
    unsuffixed names such as ``ViT-B-32`` to standard GELU, so pairing them with
    ``pretrained='openai'`` silently applies the wrong non-linearity and lowers
    every similarity score. Redirect to the matching ``-quickgelu`` variant for
    the ``openai`` tag; leave GELU-trained tags (e.g. the ``laion*`` weights)
    unchanged. When an ``open_clip`` model list is supplied, only redirect if
    the matching QuickGELU architecture is actually available in that build.
    """
    name = str(model_name)
    if str(pretrained).lower() == 'openai' and not name.endswith('-quickgelu'):
        candidate = f'{name}-quickgelu'
        if available_models is None or candidate in set(available_models):
            return candidate
    return name


def convert_image_to_rgb(image, encoding):
    """Convert a CvBridge uint8 color image to contiguous RGB."""
    image = np.asarray(image)
    normalized_encoding = str(encoding).lower()
    expected_channels = 4 if normalized_encoding in {'rgba8', 'bgra8'} else 3
    if (
        image.ndim != 3
        or image.shape[2] != expected_channels
        or image.dtype != np.uint8
    ):
        raise ValueError(
            f'Expected uint8 {normalized_encoding} image with '
            f'{expected_channels} channels, got shape={image.shape}, '
            f'dtype={image.dtype}.')

    if normalized_encoding == 'rgb8':
        rgb_image = image
    elif normalized_encoding == 'bgr8':
        rgb_image = image[:, :, ::-1]
    elif normalized_encoding == 'rgba8':
        rgb_image = image[:, :, :3]
    elif normalized_encoding == 'bgra8':
        rgb_image = image[:, :, 2::-1]
    else:
        raise ValueError(
            'image_encoding must be one of rgb8, bgr8, rgba8 or bgra8; '
            f'got {encoding!r}.')
    return np.ascontiguousarray(rgb_image)


def color_membership_score(rgb, color_name):
    """Score how well one mean RGB color matches a named basic color."""
    if rgb is None or color_name not in COLOR_ALIASES:
        return 0.0

    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1)
    if rgb.size != 3 or not np.all(np.isfinite(rgb)):
        return 0.0
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb.tolist())

    if color_name == 'black':
        return float(np.clip((0.38 - value) / 0.30, 0.0, 1.0))
    if color_name == 'white':
        low_saturation = np.clip((0.30 - saturation) / 0.25, 0.0, 1.0)
        high_value = np.clip((value - 0.55) / 0.35, 0.0, 1.0)
        return float(low_saturation * high_value)
    if color_name == 'gray':
        low_saturation = np.clip((0.30 - saturation) / 0.25, 0.0, 1.0)
        mid_value = np.clip(1.0 - abs(value - 0.50) / 0.38, 0.0, 1.0)
        return float(low_saturation * mid_value)

    hue_centers = {
        'red': (0.0, 0.09),
        'orange': (30.0 / 360.0, 0.08),
        'yellow': (58.0 / 360.0, 0.10),
        'green': (125.0 / 360.0, 0.18),
        'blue': (220.0 / 360.0, 0.17),
        'purple': (285.0 / 360.0, 0.15),
        'brown': (28.0 / 360.0, 0.09),
    }
    center, tolerance = hue_centers[color_name]
    hue_distance = abs(hue - center)
    hue_distance = min(hue_distance, 1.0 - hue_distance)
    hue_score = np.clip(1.0 - hue_distance / tolerance, 0.0, 1.0)
    saturation_score = np.clip((saturation - 0.12) / 0.45, 0.0, 1.0)
    value_score = np.clip((value - 0.08) / 0.32, 0.0, 1.0)
    score = hue_score * np.sqrt(saturation_score * value_score)
    if color_name == 'brown':
        score *= np.clip((0.82 - value) / 0.35, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))
