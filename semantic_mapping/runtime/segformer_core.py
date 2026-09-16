#!/usr/bin/env python3
"""ROS-independent SegFormer label mapping shared by runtime tools."""

import numpy as np

from semantic_mapping.runtime.semantic_schema import (
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    SEGFORMER_LABEL_ALIASES,
)

from semantic_mapping.runtime.semantic_profile import (
    aggregate_project_probability_tensor,
    open_profile,
    split_model_label,
)


__all__ = (
    'PROJECT_CLASSES', 'PROJECT_COLORS', 'LABEL_ALIASES',
    'aggregate_project_probability_tensor', 'split_model_label',
    'build_project_lookup', 'find_supported_project_classes',
)


PROJECT_CLASSES = list(DEFAULT_CLASSES)
PROJECT_COLORS = np.asarray(DEFAULT_CLASS_COLORS, dtype=np.uint8)
LABEL_ALIASES = SEGFORMER_LABEL_ALIASES


def build_project_lookup(id2label, profile_id='outdoor13'):
    """Map actual model labels into the selected fixed project channels."""
    return np.asarray(open_profile(profile_id, id2label).raw_lookup, dtype=np.uint8)


def find_supported_project_classes(id2label, profile_id='outdoor13'):
    """Return project classes directly backed by checkpoint output labels."""
    profile = open_profile(profile_id, id2label)
    return tuple(
        name for index, name in enumerate(profile.classes)
        if index in profile.supported_ids
    )
