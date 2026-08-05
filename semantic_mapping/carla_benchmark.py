#!/usr/bin/env python3
"""Pure helpers shared by the CARLA image-recognition benchmark tools."""

import colorsys
import re

import numpy as np

from semantic_mapping.semantic_schema import (
    COLOR_ALIASES,
    DEFAULT_CLASSES,
    SEGFORMER_LABEL_ALIASES,
    color_membership_score,
    normalize_label,
)


COARSE_LABELS = ('background', 'road', 'person', 'vehicle')
FINE_LABELS = (
    'background',
    'road',
    'person',
    'car',
    'truck',
    'bus',
    'bicycle',
    'motorcycle',
)
# Four-wheel benchmark vehicles. Kept as a stable name because the wheel
# filter, the coarse 'vehicle' label and the CLIP car/truck/bus crop head all
# depend on this exact set.
VEHICLE_CLASSES = ('car', 'truck', 'bus')
# Ridden two-wheelers. CARLA tags these Bicycle(19)/Motorcycle(18), separate
# from the four-wheel Vehicles tag, and usually spawns a Rider(13) mesh too.
TWO_WHEEL_CLASSES = ('bicycle', 'motorcycle')
# Every fine class the capture scheduler can balance and spawn.
SPAWN_CLASSES = VEHICLE_CLASSES + TWO_WHEEL_CLASSES

CARLA_ENUM_CANDIDATES = {
    'road': ('Roads', 'Road'),
    'person': ('Pedestrians', 'Pedestrian'),
    'car': ('Car',),
    'truck': ('Truck',),
    'bus': ('Bus',),
    'bicycle': ('Bicycle',),
    'motorcycle': ('Motorcycle',),
    'rider': ('Rider',),
    'vehicle': ('Vehicles', 'Vehicle'),
}


def split_model_label(label):
    """Return normalized model-label terms split on common separators."""
    terms = {normalize_label(label)}
    terms.update(
        normalize_label(part)
        for part in re.split(r'[,;/|]+', str(label))
    )
    return {term for term in terms if term}


def build_project_lookup(id2label):
    """Map model class IDs into the project's compact semantic vocabulary."""
    class_count = max((int(class_id) for class_id in id2label), default=-1) + 1
    unknown_index = len(DEFAULT_CLASSES) - 1
    lookup = np.full(class_count, unknown_index, dtype=np.uint8)
    normalized_aliases = {
        project_id: {normalize_label(alias) for alias in aliases}
        for project_id, aliases in SEGFORMER_LABEL_ALIASES.items()
    }
    for class_id, original_label in id2label.items():
        label_terms = split_model_label(original_label)
        for project_id, aliases in normalized_aliases.items():
            if label_terms.intersection(aliases):
                lookup[int(class_id)] = project_id
                break
    return lookup


def decode_carla_bgra(raw_data, width, height):
    """Decode CARLA BGRA bytes into an RGB uint8 image."""
    bgra = np.frombuffer(raw_data, dtype=np.uint8)
    expected = int(width) * int(height) * 4
    if bgra.size != expected:
        raise ValueError(
            f'Expected {expected} BGRA bytes, received {bgra.size}.')
    bgra = bgra.reshape((int(height), int(width), 4))
    return np.ascontiguousarray(bgra[:, :, 2::-1])


def decode_carla_semantic(raw_data, width, height):
    """Decode the semantic tag stored in the red channel of CARLA BGRA."""
    bgra = np.frombuffer(raw_data, dtype=np.uint8)
    expected = int(width) * int(height) * 4
    if bgra.size != expected:
        raise ValueError(
            f'Expected {expected} semantic bytes, received {bgra.size}.')
    return bgra.reshape((int(height), int(width), 4))[:, :, 2].copy()


def decode_carla_instance(raw_data, width, height):
    """Decode CARLA's G/B channels into the uint16 actor/object ID."""
    bgra = np.frombuffer(raw_data, dtype=np.uint8)
    expected = int(width) * int(height) * 4
    if bgra.size != expected:
        raise ValueError(
            f'Expected {expected} instance bytes, received {bgra.size}.')
    bgra = bgra.reshape((int(height), int(width), 4))
    return (
        bgra[:, :, 0].astype(np.uint16) << 8
    ) | bgra[:, :, 1].astype(np.uint16)


def actor_instance_id(actor_id):
    """Return the 16-bit ID encoded by CARLA's instance camera."""
    return int(actor_id) & 0xFFFF


def bbox_from_mask(mask):
    """Return the tight xyxy bounding box of a nonempty boolean mask."""
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return (
        int(np.min(columns)),
        int(np.min(rows)),
        int(np.max(columns)) + 1,
        int(np.max(rows)) + 1,
    )


def byte_swap_instance_ids(instance_mask):
    """Repair datasets captured with the pre-v2 G/B byte order bug."""
    values = np.asarray(instance_mask, dtype=np.uint16)
    return ((values & 0x00FF) << 8) | ((values & 0xFF00) >> 8)


def project_world_points(points_xyz, inverse_camera_matrix, camera_matrix):
    """
    Project CARLA world points into image pixels.

    CARLA uses x-forward, y-right, z-up. Camera projection uses x-right,
    y-down, z-forward, hence the axis permutation after the world transform.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected Nx3 points, got {points.shape}.')
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    sensor_points = (
        np.asarray(inverse_camera_matrix, dtype=np.float64)
        @ homogeneous.T
    ).T
    camera_points = np.column_stack((
        sensor_points[:, 1],
        -sensor_points[:, 2],
        sensor_points[:, 0],
    ))
    depth = camera_points[:, 2]
    projected = (
        np.asarray(camera_matrix, dtype=np.float64) @ camera_points.T
    ).T
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = depth > 1e-4
    pixels[valid] = projected[valid, :2] / depth[valid, None]
    return pixels, depth


def camera_intrinsics(width, height, horizontal_fov_deg):
    """Return a pinhole camera matrix matching a CARLA RGB camera."""
    width = float(width)
    height = float(height)
    focal = width / (2.0 * np.tan(
        np.deg2rad(float(horizontal_fov_deg)) / 2.0))
    return np.asarray([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def clipped_bbox(pixels, depth, width, height):
    """Return an in-frame xyxy box for projected vertices, or ``None``."""
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(pixels).all(axis=1) & (depth > 1e-4)
    if not np.any(valid):
        return None
    x0 = int(np.floor(np.min(pixels[valid, 0])))
    y0 = int(np.floor(np.min(pixels[valid, 1])))
    x1 = int(np.ceil(np.max(pixels[valid, 0])))
    y1 = int(np.ceil(np.max(pixels[valid, 1])))
    x0 = int(np.clip(x0, 0, int(width) - 1))
    y0 = int(np.clip(y0, 0, int(height) - 1))
    x1 = int(np.clip(x1, x0 + 1, int(width)))
    y1 = int(np.clip(y1, y0 + 1, int(height)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def dominant_instance_in_bbox(
    instance_mask,
    semantic_mask,
    bbox,
    accepted_semantic_tags,
):
    """Return the dominant nonzero instance ID inside an actor projection."""
    x0, y0, x1, y1 = bbox
    instances = np.asarray(instance_mask)[y0:y1, x0:x1]
    semantics = np.asarray(semantic_mask)[y0:y1, x0:x1]
    accepted = np.isin(semantics, list(accepted_semantic_tags))
    values = instances[accepted & (instances != 0)]
    if values.size == 0:
        return None, 0
    ids, counts = np.unique(values, return_counts=True)
    index = int(np.argmax(counts))
    return int(ids[index]), int(counts[index])


def parse_carla_color(color_text):
    """Parse a CARLA ``R,G,B`` blueprint color into an RGB vector."""
    values = [part.strip() for part in str(color_text).split(',')]
    if len(values) != 3:
        raise ValueError(f'Invalid CARLA color {color_text!r}.')
    rgb = np.asarray([float(value) for value in values], dtype=np.float32)
    return np.clip(rgb, 0.0, 255.0)


def canonical_color_from_rgb(rgb):
    """Assign one RGB value to the closest supported basic color."""
    scores = {
        color: color_membership_score(rgb, color)
        for color in COLOR_ALIASES
    }
    return max(scores, key=scores.get), scores


def robust_instance_color_scores(rgb_pixels):
    """
    Aggregate named-color evidence over one instance mask.

    A top-fraction score is used because windows, tyres, lamps and shadows are
    part of a vehicle instance but should not overwrite its body color.
    """
    pixels = np.asarray(rgb_pixels, dtype=np.float64).reshape(-1, 3)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if pixels.size == 0:
        return {color: 0.0 for color in COLOR_ALIASES}
    if np.max(pixels) <= 1.0:
        pixels = pixels * 255.0
    pixels = np.clip(pixels, 0.0, 255.0)

    sample_step = max(1, pixels.shape[0] // 20000)
    pixels = pixels[::sample_step]
    scores = {}
    top_count = max(1, int(np.ceil(pixels.shape[0] * 0.35)))
    for color in COLOR_ALIASES:
        values = np.asarray([
            color_membership_score(pixel, color) for pixel in pixels
        ], dtype=np.float64)
        top_values = np.partition(values, -top_count)[-top_count:]
        support = float(np.mean(values >= 0.25))
        scores[color] = float(np.mean(top_values) * np.sqrt(support))
    return scores


def classify_instance_color(rgb_pixels):
    """Return ``(best_color, scores)`` for RGB pixels of one object."""
    scores = robust_instance_color_scores(rgb_pixels)
    return max(scores, key=scores.get), scores


def make_coarse_masks(carla_tags, project_mask, semantic_tag_ids):
    """Convert CARLA truth and project predictions to four coarse classes."""
    carla_tags = np.asarray(carla_tags)
    project_mask = np.asarray(project_mask)
    if carla_tags.shape != project_mask.shape:
        raise ValueError(
            f'GT and prediction shapes differ: {carla_tags.shape} vs '
            f'{project_mask.shape}.')

    truth = np.zeros(carla_tags.shape, dtype=np.uint8)
    prediction = np.zeros(project_mask.shape, dtype=np.uint8)
    for key in ('road',):
        truth[np.isin(carla_tags, semantic_tag_ids.get(key, ()))] = 1
    for key in ('person', 'rider'):
        truth[np.isin(carla_tags, semantic_tag_ids.get(key, ()))] = 2
    vehicle_ids = []
    for key in ('vehicle',) + SPAWN_CLASSES:
        vehicle_ids.extend(semantic_tag_ids.get(key, ()))
    truth[np.isin(carla_tags, vehicle_ids)] = 3

    class_indices = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    prediction[project_mask == class_indices['road']] = 1
    prediction[project_mask == class_indices['person']] = 2
    prediction[np.isin(project_mask, [
        class_indices[name] for name in SPAWN_CLASSES
    ])] = 3
    return truth, prediction


def confusion_matrix(truth, prediction, class_count):
    """Return an integer confusion matrix with rows=truth, columns=prediction."""
    truth = np.asarray(truth, dtype=np.int64).ravel()
    prediction = np.asarray(prediction, dtype=np.int64).ravel()
    valid = (
        (truth >= 0)
        & (truth < int(class_count))
        & (prediction >= 0)
        & (prediction < int(class_count))
    )
    encoded = truth[valid] * int(class_count) + prediction[valid]
    return np.bincount(
        encoded, minlength=int(class_count) ** 2
    ).reshape((int(class_count), int(class_count)))


def metrics_from_confusion(matrix, labels):
    """Compute per-class precision, recall, F1 and IoU."""
    matrix = np.asarray(matrix, dtype=np.float64)
    result = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        false_positive = np.sum(matrix[:, index]) - true_positive
        false_negative = np.sum(matrix[index, :]) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
        iou = true_positive / max(
            true_positive + false_positive + false_negative, 1.0)
        result[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'support_pixels': int(np.sum(matrix[index, :])),
        }
    return result


def grid_cell_truth(
    semantic_tags,
    semantic_tag_ids,
    grid_rows,
    grid_cols,
    min_fraction=0.02,
):
    """Return the set of benchmark labels present in every image grid cell."""
    semantic_tags = np.asarray(semantic_tags)
    height, width = semantic_tags.shape
    result = []
    person_ids = set(semantic_tag_ids.get('person', ()))
    person_ids.update(semantic_tag_ids.get('rider', ()))
    label_ids = {
        'road': set(semantic_tag_ids.get('road', ())),
        'person': person_ids,
        **{
            name: set(semantic_tag_ids.get(name, ()))
            for name in SPAWN_CLASSES
        },
    }
    common_vehicle_ids = set(semantic_tag_ids.get('vehicle', ()))
    patch_h = height // int(grid_rows)
    patch_w = width // int(grid_cols)
    for row in range(int(grid_rows)):
        for column in range(int(grid_cols)):
            y0 = row * patch_h
            y1 = height if row == int(grid_rows) - 1 else (row + 1) * patch_h
            x0 = column * patch_w
            x1 = width if column == int(grid_cols) - 1 else (
                column + 1) * patch_w
            patch = semantic_tags[y0:y1, x0:x1]
            labels = set()
            fractions = {}
            for label, ids in label_ids.items():
                fraction = float(np.mean(np.isin(patch, list(ids))))
                fractions[label] = fraction
                if fraction >= float(min_fraction):
                    labels.add(label)
            vehicle_fraction = float(np.mean(
                np.isin(patch, list(common_vehicle_ids))))
            fractions['vehicle'] = vehicle_fraction
            if vehicle_fraction >= float(min_fraction):
                labels.add('vehicle')
            result.append({
                'row': row,
                'column': column,
                'labels': sorted(labels),
                'fractions': fractions,
            })
    return result


def augment_grid_truth_with_instances(
    grid_truth,
    instance_mask,
    visible_instances,
    grid_rows,
    grid_cols,
    min_pixels=20,
):
    """
    Add exact actor classes to grid truth using CARLA instance IDs.

    Some CARLA releases expose only a generic ``Vehicles`` semantic tag.
    Actor metadata still provides exact vehicle identity, so instance masks
    are the authoritative source for fine-grained grid labels, including
    bicycles and motorcycles.
    """
    instance_mask = np.asarray(instance_mask)
    height, width = instance_mask.shape
    patch_h = height // int(grid_rows)
    patch_w = width // int(grid_cols)
    for item in visible_instances:
        instance_id = int(item.get('instance_id', 0))
        label = normalize_label(item.get('benchmark_class', ''))
        if instance_id <= 0 or label not in SPAWN_CLASSES:
            continue
        object_mask = instance_mask == instance_id
        for row in range(int(grid_rows)):
            for column in range(int(grid_cols)):
                y0 = row * patch_h
                y1 = (
                    height if row == int(grid_rows) - 1
                    else (row + 1) * patch_h
                )
                x0 = column * patch_w
                x1 = (
                    width if column == int(grid_cols) - 1
                    else (column + 1) * patch_w
                )
                if np.count_nonzero(
                    object_mask[y0:y1, x0:x1]
                ) < int(min_pixels):
                    continue
                index = row * int(grid_cols) + column
                labels = set(grid_truth[index]['labels'])
                labels.update((label, 'vehicle'))
                grid_truth[index]['labels'] = sorted(labels)
    return grid_truth


def rgb_to_hsv_summary(rgb):
    """Return a human-readable HSV triple for diagnostics."""
    values = np.asarray(rgb, dtype=np.float64)
    if np.max(values) > 1.0:
        values = values / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(
        *np.clip(values, 0.0, 1.0).tolist())
    return {
        'hue_deg': float(hue * 360.0),
        'saturation': float(saturation),
        'value': float(value),
    }
