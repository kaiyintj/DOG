#!/usr/bin/env python3
"""
Voxel-fusion ablations for the CARLA reliability benchmark.

This module deliberately contains no ROS or CARLA dependency.  It is a thin
adapter around :class:`semantic_mapping.runtime.voxel_map.VoxelMap`: all six
ablations
therefore use the same Dirichlet update, continuous-time decay, and evidence
caps as the runtime semantic map.  Only the per-point reliability supplied to
``VoxelMap.update`` changes.
"""

from collections import defaultdict

import numpy as np

from semantic_mapping.runtime.semantic_schema import (
    DEFAULT_CLASS_COLORS,
    DEFAULT_CLASSES,
)
from semantic_mapping.runtime.voxel_map import VoxelMap


ABLATION_NAMES = (
    'none',
    'semantic',
    'semantic_range',
    'semantic_range_density',
    'semantic_range_density_view',
    'full',
)


def _reliability_array(values, size, name):
    """Return one finite, clipped point-wise reliability array."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != size:
        raise ValueError(f'{name} must have length {size}, got {len(values)}')
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(values, 0.0, 1.0)


def ablation_weights(r_semantic, r_range, r_density, r_view, r_motion):
    """
    Return the six point-wise reliability products used for ablation.

    Inputs are the already-computed GA-BSVM factors.  Keeping multiplication in
    one place makes the benchmark's ``full`` condition exactly the operational
    product while each preceding condition adds one factor at a time.
    """
    r_semantic = np.asarray(r_semantic, dtype=np.float32).reshape(-1)
    count = len(r_semantic)
    r_semantic = _reliability_array(r_semantic, count, 'r_semantic')
    r_range = _reliability_array(r_range, count, 'r_range')
    r_density = _reliability_array(r_density, count, 'r_density')
    r_view = _reliability_array(r_view, count, 'r_view')
    r_motion = _reliability_array(r_motion, count, 'r_motion')

    semantic_range = r_semantic * r_range
    semantic_range_density = semantic_range * r_density
    semantic_range_density_view = semantic_range_density * r_view
    return {
        'none': np.ones(count, dtype=np.float32),
        'semantic': r_semantic,
        'semantic_range': semantic_range.astype(np.float32, copy=False),
        'semantic_range_density': semantic_range_density.astype(
            np.float32, copy=False),
        'semantic_range_density_view': semantic_range_density_view.astype(
            np.float32, copy=False),
        'full': (semantic_range_density_view * r_motion).astype(
            np.float32, copy=False),
    }


def _normalise_ros_parameters(params):
    """Accept direct ROS parameters or a conventional YAML node mapping."""
    if params is None:
        return {}
    params = dict(params)
    if 'ros__parameters' in params:
        return dict(params['ros__parameters'])
    node_mapping = params.get('ga_bsvm_node')
    if isinstance(node_mapping, dict) and 'ros__parameters' in node_mapping:
        return dict(node_mapping['ros__parameters'])
    return params


def voxel_map_kwargs_from_params(
    params,
    *,
    class_count=None,
    class_colors=None,
):
    """
    Extract runtime-equivalent ``VoxelMap`` constructor arguments.

    ``max_observation_weight <= 0`` has the same meaning as in GA-BSVM: let
    ``VoxelMap`` derive its finite cap from ``max_total_evidence``.
    """
    params = _normalise_ros_parameters(params)
    if class_count is None:
        class_count = params.get(
            'num_classes', params.get('K', len(DEFAULT_CLASSES)))
    configured_observation_weight = params.get('max_observation_weight', 0.0)
    if configured_observation_weight is not None:
        configured_observation_weight = float(configured_observation_weight)
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS
    return {
        'voxel_size': float(params.get('voxel_size', 0.1)),
        'K': int(class_count),
        'evidence_prior': float(params.get('evidence_prior', 1.0)),
        'evidence_strength': float(params.get('evidence_strength', 1.0)),
        'evidence_decay': float(params.get('evidence_decay', 0.995)),
        'evidence_decay_reference_sec': float(
            params.get('evidence_decay_reference_sec', 0.1)),
        'max_frame_evidence': float(params.get('max_frame_evidence', 3.0)),
        'max_total_evidence': float(params.get('max_total_evidence', 200.0)),
        'max_observation_weight': (
            configured_observation_weight
            if configured_observation_weight is not None
            and configured_observation_weight > 0.0 else None
        ),
        'uncertainty_entropy_weight': float(
            params.get('uncertainty_entropy_weight', 0.7)),
        'class_colors': class_colors,
    }


def _binary_auroc(labels, scores):
    """
    Compute binary AUROC without scipy/sklearn.

    labels:
        1 = wrong voxel
        0 = correct voxel

    scores:
        voxel uncertainty

    AUROC > 0.5 means larger uncertainty tends to identify errors.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]

    positive = labels == 1
    negative = labels == 0

    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))

    if n_pos == 0 or n_neg == 0:
        return None

    # Rank scores with average ranks for ties.
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1

        while (
            j < len(scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        # 1-based average rank for [i, j)
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank

        i = j

    rank_sum_pos = float(np.sum(ranks[positive]))

    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def _binary_score_correlation(labels, scores):
    """
    Pearson correlation between voxel error and uncertainty.

    Positive value:
        wrong voxels tend to have higher uncertainty.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]

    if len(labels) < 2:
        return None

    if np.std(labels) <= 1e-12 or np.std(scores) <= 1e-12:
        return None

    return float(np.corrcoef(labels, scores)[0, 1])


class VoxelAblation:
    """
    Fuse identical observations into six runtime-equivalent voxel maps.

    Ground truth is accumulated separately as an unweighted majority vote in
    the same voxel lattice.  Labels below zero (and labels outside ``K``) are
    ignored; this represents unsupported or unmatched semantic-LiDAR points.
    """

    def __init__(self, params=None, *, class_count=None, class_colors=None):
        """Create one identically configured map for every ablation name."""
        self.map_kwargs = voxel_map_kwargs_from_params(
            params,
            class_count=class_count,
            class_colors=class_colors,
        )
        self.class_count = int(self.map_kwargs['K'])
        self.maps = {
            name: VoxelMap(**self.map_kwargs) for name in ABLATION_NAMES
        }
        self._ground_truth_counts = defaultdict(
            lambda: np.zeros(self.class_count, dtype=np.int64))

    @property
    def voxel_size(self):
        """Return the common voxel size used by all ablation maps."""
        return self.map_kwargs['voxel_size']

    def update(
        self,
        points_world,
        logits,
        gt_ids,
        weights,
        timestamp_sec=None,
    ):
        """
        Fuse one frame and update the supported ground-truth voxel majority.

        Args:
            points_world: World-frame points shaped ``(N, 3)``.
            logits: Project-ontology logits shaped ``(N, K)``.
            gt_ids: Integer ground-truth project labels; negative means ignore.
            weights: Mapping returned by :func:`ablation_weights`.
            timestamp_sec: Timestamp passed unchanged to each ``VoxelMap``.

        Returns
        -------
            A mapping from ablation name to the ``VoxelMap.update`` return
            tuple ``(new_count, updated_count)``.

        """
        points_world = np.asarray(points_world, dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        gt_ids = np.asarray(gt_ids, dtype=np.int64).reshape(-1)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError('points_world must have shape (N, 3)')
        if (
            logits.ndim != 2
            or logits.shape != (len(points_world), self.class_count)
        ):
            raise ValueError(
                'logits must have shape (N, K) matching points_world and '
                'class_count')
        if len(gt_ids) != len(points_world):
            raise ValueError('gt_ids must match points_world length')
        missing = set(ABLATION_NAMES).difference(weights)
        if missing:
            raise ValueError(f'weights missing ablations: {sorted(missing)}')

        finite_points = np.all(np.isfinite(points_world), axis=1)
        finite_logits = np.all(np.isfinite(logits), axis=1)
        usable = finite_points & finite_logits
        gt_usable = usable & (gt_ids >= 0) & (gt_ids < self.class_count)
        reference_map = self.maps['none']
        for point, gt_id in zip(points_world[gt_usable], gt_ids[gt_usable]):
            key = reference_map.get_voxel_indices(point)
            self._ground_truth_counts[key][gt_id] += 1

        results = {}
        for name in ABLATION_NAMES:
            reliability = _reliability_array(
                weights[name], len(points_world), name)
            results[name] = self.maps[name].update(
                points_world[usable],
                reliability[usable],
                logits[usable],
                timestamp_sec=timestamp_sec,
            )
        return results

    def report(self):
        """
        Return voxel-level accuracy and uncertainty for every ablation.

        ``all_gt_accuracy`` uses all ground-truth voxels as denominator and
        treats a GT voxel missing from an ablation map as incorrect.

        ``covered_accuracy`` is conditional on a map voxel existing.

        Uncertainty-error diagnostics are evaluated only on covered voxels,
        because missing voxels have no posterior uncertainty.
        """
        gt_voxels = dict(self._ground_truth_counts)
        gt_count = len(gt_voxels)

        result = {}

        for name in ABLATION_NAMES:
            voxel_map = self.maps[name]

            covered = 0
            correct = 0

            entropies = []
            uncertainties = []

            correct_entropies = []
            wrong_entropies = []

            correct_uncertainties = []
            wrong_uncertainties = []

            error_labels = []
            uncertainty_scores = []

            for key, label_counts in gt_voxels.items():
                if key not in voxel_map.voxels:
                    continue

                covered += 1

                gt_id = int(np.argmax(label_counts))

                predicted_id = int(
                    np.argmax(voxel_map.get_probabilities(key))
                )

                is_correct = predicted_id == gt_id

                if is_correct:
                    correct += 1

                entropy, _epistemic, uncertainty = (
                    voxel_map.get_uncertainty(key)
                )

                entropy = float(entropy)
                uncertainty = float(uncertainty)

                entropies.append(entropy)
                uncertainties.append(uncertainty)

                if is_correct:
                    correct_entropies.append(entropy)
                    correct_uncertainties.append(uncertainty)
                else:
                    wrong_entropies.append(entropy)
                    wrong_uncertainties.append(uncertainty)

                # Error is the positive class.
                error_labels.append(0 if is_correct else 1)
                uncertainty_scores.append(uncertainty)

            missing = gt_count - covered
            wrong = covered - correct

            mean_entropy_correct = (
                float(np.mean(correct_entropies))
                if correct_entropies
                else None
            )

            mean_entropy_wrong = (
                float(np.mean(wrong_entropies))
                if wrong_entropies
                else None
            )

            mean_uncertainty_correct = (
                float(np.mean(correct_uncertainties))
                if correct_uncertainties
                else None
            )

            mean_uncertainty_wrong = (
                float(np.mean(wrong_uncertainties))
                if wrong_uncertainties
                else None
            )

            entropy_gap = (
                float(mean_entropy_wrong - mean_entropy_correct)
                if (
                    mean_entropy_wrong is not None
                    and mean_entropy_correct is not None
                )
                else None
            )

            uncertainty_gap = (
                float(
                    mean_uncertainty_wrong
                    - mean_uncertainty_correct
                )
                if (
                    mean_uncertainty_wrong is not None
                    and mean_uncertainty_correct is not None
                )
                else None
            )

            uncertainty_error_auroc = _binary_auroc(
                error_labels,
                uncertainty_scores,
            )

            uncertainty_error_correlation = (
                _binary_score_correlation(
                    error_labels,
                    uncertainty_scores,
                )
            )

            result[name] = {
                'gt_voxel_count': int(gt_count),

                'covered_voxel_count': int(covered),
                'missing_voxel_count': int(missing),

                'correct_voxel_count': int(correct),
                'wrong_voxel_count': int(wrong),

                # Short aliases retained for compatibility.
                'covered': int(covered),
                'missing': int(missing),

                'coverage': (
                    float(covered / gt_count)
                    if gt_count
                    else None
                ),

                'covered_accuracy': (
                    float(correct / covered)
                    if covered
                    else None
                ),

                'all_gt_accuracy': (
                    float(correct / gt_count)
                    if gt_count
                    else None
                ),

                # Overall uncertainty statistics.
                'mean_entropy': (
                    float(np.mean(entropies))
                    if entropies
                    else None
                ),

                'mean_uncertainty': (
                    float(np.mean(uncertainties))
                    if uncertainties
                    else None
                ),

                # Correct-vs-wrong voxel diagnostics.
                'mean_entropy_correct': (
                    mean_entropy_correct
                ),

                'mean_entropy_wrong': (
                    mean_entropy_wrong
                ),

                'entropy_gap_wrong_minus_correct': (
                    entropy_gap
                ),

                'mean_uncertainty_correct': (
                    mean_uncertainty_correct
                ),

                'mean_uncertainty_wrong': (
                    mean_uncertainty_wrong
                ),

                'uncertainty_gap_wrong_minus_correct': (
                    uncertainty_gap
                ),

                # Error detection metrics.
                'uncertainty_error_auroc': (
                    uncertainty_error_auroc
                ),

                'uncertainty_error_correlation': (
                    uncertainty_error_correlation
                ),
            }

        return result
