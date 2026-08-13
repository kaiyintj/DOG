#!/usr/bin/env python3
"""ROS-independent SegFormer label mapping shared by runtime tools."""

import re

import numpy as np

from semantic_mapping.runtime.semantic_schema import (
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    SEGFORMER_LABEL_ALIASES,
    normalize_label,
)


PROJECT_CLASSES = list(DEFAULT_CLASSES)
PROJECT_COLORS = np.asarray(DEFAULT_CLASS_COLORS, dtype=np.uint8)
LABEL_ALIASES = SEGFORMER_LABEL_ALIASES


def aggregate_project_probability_tensor(
    raw_probabilities,
    project_lookup,
    num_project_classes,
):
    """Sum a BxCxHxW tensor into project classes without hard decisions."""
    if raw_probabilities.ndim != 4:
        raise ValueError('raw probability tensor must have shape BxCxHxW')
    if project_lookup.ndim != 1:
        raise ValueError('project lookup tensor must be one-dimensional')
    if raw_probabilities.shape[1] != project_lookup.shape[0]:
        raise ValueError('project lookup must cover every raw class')
    if int(num_project_classes) <= 0:
        raise ValueError('num_project_classes must be positive')
    projected = raw_probabilities.new_zeros(
        (
            raw_probabilities.shape[0],
            int(num_project_classes),
            raw_probabilities.shape[2],
            raw_probabilities.shape[3],
        )
    )
    projected.index_add_(1, project_lookup, raw_probabilities)
    return projected / projected.sum(dim=1, keepdim=True).clamp_min(1e-12)


def split_model_label(label):
    """Return a model label and each comma/slash-separated synonym."""
    terms = {normalize_label(label)}
    terms.update(
        normalize_label(part)
        for part in re.split(r'[,;/|]+', str(label))
    )
    return {term for term in terms if term}


def build_project_lookup(id2label):
    """Map a model label set into the compact project classes."""
    class_count = max(int(class_id) for class_id in id2label) + 1
    unknown_index = len(PROJECT_CLASSES) - 1
    lookup = np.full(class_count, unknown_index, dtype=np.uint8)
    for class_id, original_label in id2label.items():
        label_terms = split_model_label(original_label)
        for project_id, aliases in LABEL_ALIASES.items():
            normalized_aliases = {normalize_label(alias) for alias in aliases}
            if label_terms.intersection(normalized_aliases):
                lookup[int(class_id)] = project_id
                break
    return lookup


def find_supported_project_classes(id2label):
    """Return navigation classes directly emitted by one model checkpoint."""
    lookup = build_project_lookup(id2label)
    supported_ids = {int(class_id) for class_id in lookup.tolist()}
    unknown_index = len(PROJECT_CLASSES) - 1
    return tuple(
        class_name
        for class_id, class_name in enumerate(PROJECT_CLASSES)
        if class_id != unknown_index and class_id in supported_ids
    )
