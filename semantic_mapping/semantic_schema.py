#!/usr/bin/env python3
"""Shared semantic classes, query aliases and color attribute scoring."""

import colorsys
import re

import numpy as np


DEFAULT_CLASSES = (
    'road',
    'building',
    'tree',
    'person',
    'car',
    'truck',
    'bus',
    'bicycle',
    'motorcycle',
    'chair',
    'bench',
    'unknown background',
)

DEFAULT_CLASS_COLORS = (
    (120, 120, 120),  # road
    (210, 210, 210),  # building
    (0, 160, 0),      # tree
    (255, 0, 0),      # person
    (0, 90, 255),     # car
    (255, 140, 0),    # truck
    (150, 70, 200),   # bus
    (0, 220, 220),    # bicycle
    (255, 0, 180),    # motorcycle
    (160, 100, 40),   # chair
    (255, 220, 0),    # bench
    (50, 50, 50),     # unknown background
)

DEFAULT_SEMANTIC_COSTS = (
    0,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    -1,
)

DEFAULT_QUERY_MAX_EXTENTS_M = (
    100.0,  # road
    100.0,  # building
    100.0,  # tree
    1.5,    # person
    5.0,    # car
    9.0,    # truck
    14.0,   # bus
    3.0,    # bicycle
    3.5,    # motorcycle
    2.0,    # chair
    4.0,    # bench
    100.0,  # unknown background
)

# ADE20K and Cityscapes labels mapped into the compact navigation and
# retrieval vocabulary. Aliases are matched by name, so both label sets share
# one table: 'vegetation' and 'rider' come from Cityscapes, the plant/grass
# and windowpane terms from ADE20K. 'rider' folds cyclists into 'person' so a
# navigating robot treats them as humans to avoid.
SEGFORMER_LABEL_ALIASES = {
    0: {'road', 'sidewalk', 'path'},
    1: {'building', 'wall', 'house', 'fence', 'door', 'windowpane'},
    2: {'tree', 'plant', 'grass', 'flower', 'vegetation'},
    3: {'person', 'rider'},
    4: {'car', 'van', 'auto', 'automobile', 'motorcar'},
    5: {'truck'},
    6: {'bus'},
    7: {'bicycle'},
    8: {'minibike', 'motorcycle', 'motorbike'},
    9: {'chair', 'armchair', 'swivel chair', 'seat'},
    10: {'bench'},
}

QUERY_CLASS_ALIASES = {
    'road': {'road', 'street', 'sidewalk', 'path', '道路', '路面'},
    'building': {'building', 'wall', 'house', '建筑', '墙', '房屋'},
    'tree': {'tree', 'plant', 'vegetation', '树', '树木', '植物'},
    'person': {'person', 'people', 'human', 'pedestrian', '人', '行人'},
    'car': {
        'car', 'cars', 'automobile', 'vehicle', 'van',
        '汽车', '轿车', '车辆',
    },
    'truck': {'truck', 'trucks', 'lorry', '卡车', '货车'},
    'bus': {'bus', 'buses', 'coach', '公交车', '巴士'},
    'bicycle': {
        'bicycle', 'bicycles', 'bike', 'bikes', 'cycle',
        '自行车', '单车',
    },
    'motorcycle': {
        'motorcycle', 'motorcycles', 'motorbike', 'minibike',
        '摩托车',
    },
    'chair': {'chair', 'chairs', 'seat', 'armchair', '椅子', '座椅'},
    'bench': {'bench', 'benches', '长椅', '长凳'},
    'unknown background': {'unknown background'},
}

COLOR_ALIASES = {
    'red': {'red', '红', '红色'},
    'orange': {'orange', '橙', '橙色'},
    'yellow': {'yellow', '黄', '黄色'},
    'green': {'green', '绿', '绿色'},
    'blue': {'blue', '蓝', '蓝色'},
    'purple': {'purple', 'violet', '紫', '紫色'},
    'brown': {'brown', '棕', '棕色', '褐色'},
    'black': {'black', '黑', '黑色'},
    'white': {'white', '白', '白色'},
    'gray': {'gray', 'grey', '灰', '灰色'},
}


def resolve_clip_model_name(model_name, pretrained, available_models=None):
    """Return the CLIP variant whose activation matches the pretrained weights.

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


def normalize_label(label):
    """Normalize model labels and text queries for matching."""
    return str(label).lower().replace('-', ' ').replace('_', ' ').strip()


def _alias_matches(alias, normalized_text, tokens):
    normalized_alias = normalize_label(alias)
    if not normalized_alias:
        return False
    if re.fullmatch(r'[a-z0-9 ]+', normalized_alias):
        if ' ' in normalized_alias:
            return normalized_alias in normalized_text
        return normalized_alias in tokens
    return normalized_alias in normalized_text


def parse_semantic_query(query, vocab=DEFAULT_CLASSES):
    """Return ``(class_index, color_name)`` parsed from a text query."""
    normalized = normalize_label(query)
    tokens = set(re.findall(r'[a-z0-9]+', normalized))

    color_name = None
    for canonical_color, aliases in COLOR_ALIASES.items():
        if any(_alias_matches(alias, normalized, tokens) for alias in aliases):
            color_name = canonical_color
            break

    class_index = None
    vocab_lower = [normalize_label(name) for name in vocab]
    for index, class_name in enumerate(vocab_lower):
        aliases = set(QUERY_CLASS_ALIASES.get(class_name, ()))
        aliases.add(class_name)
        if any(_alias_matches(alias, normalized, tokens) for alias in aliases):
            class_index = index
            break

    return class_index, color_name


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
