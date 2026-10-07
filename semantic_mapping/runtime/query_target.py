"""
ROS-independent query-target planning primitives.

The ROS node owns message transport and map adaptation.  This module owns the
spatial-instance ranking and the decision that turns a ranked, read-only
candidate snapshot into the first candidate with a usable approach.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from semantic_mapping.runtime.candidate_observation import summarize_observations


@dataclass(frozen=True)
class QueryTargetSnapshot:
    """
    Ranked candidates captured for one query attempt.

    Candidate mappings are treated as read-only by the planner.  The node may
    keep richer diagnostic fields on them, while the planner only iterates and
    returns the selected mapping.
    """

    candidates: Tuple[Mapping[str, Any], ...]
    map_revision: int = 0

    def __post_init__(self):
        """Normalize the candidate sequence and revision value."""
        frozen_candidates = []
        for candidate in self.candidates:
            values = dict(candidate)
            for field in ('pos', 'color_rgb'):
                value = values.get(field)
                if isinstance(value, np.ndarray):
                    value = np.array(value, copy=True)
                    value.setflags(write=False)
                    values[field] = value
            frozen_candidates.append(MappingProxyType(values))
        object.__setattr__(self, 'candidates', tuple(frozen_candidates))
        object.__setattr__(self, 'map_revision', int(self.map_revision))


@dataclass(frozen=True)
class QueryApproachVoxel:
    """One traversable voxel in a read-only approach snapshot."""

    key: Any
    position: np.ndarray
    confidence: float

    def __post_init__(self):
        """Copy the position so callers cannot mutate the snapshot."""
        position = np.array(self.position, dtype=np.float32, copy=True)
        position.setflags(write=False)
        object.__setattr__(self, 'position', position)
        object.__setattr__(self, 'confidence', float(self.confidence))


@dataclass(frozen=True)
class QueryApproachSnapshot:
    """Map-derived approach evidence shared by all candidates in one query."""

    voxels: Tuple[QueryApproachVoxel, ...]
    robot_position: Optional[np.ndarray]
    voxel_size: float
    total_voxels: int
    filtered_diagnostics: Mapping[str, int]
    map_revision: int = 0

    def __post_init__(self):
        """Normalize snapshot containers and scalar metadata."""
        object.__setattr__(self, 'voxels', tuple(self.voxels))
        robot_position = self.robot_position
        if robot_position is not None:
            robot_position = np.array(
                robot_position, dtype=np.float32, copy=True)
            robot_position.setflags(write=False)
        object.__setattr__(self, 'robot_position', robot_position)
        object.__setattr__(self, 'total_voxels', int(self.total_voxels))
        object.__setattr__(self, 'voxel_size', float(self.voxel_size))
        object.__setattr__(
            self,
            'filtered_diagnostics',
            MappingProxyType(dict(self.filtered_diagnostics)),
        )
        object.__setattr__(self, 'map_revision', int(self.map_revision))


@dataclass(frozen=True)
class ApproachPolicy:
    """Configuration for selecting an approach voxel."""

    min_distance_m: float
    max_distance_m: float
    desired_distance_m: float
    robot_distance_weight: float = 0.1
    prefer_robot_side: bool = True
    require_robot_side: bool = False
    same_side_min_cosine: float = 0.0
    require_robot_pose: bool = False
    require_safe: bool = True


@dataclass(frozen=True)
class ApproachDecision:
    """Result of one pure approach search."""

    key: Any
    position: Optional[np.ndarray]
    safe: bool
    diagnostics: Mapping[str, Any]
    fallback_used: bool = False
    reason: str = ''

    @property
    def succeeded(self):
        """Whether the decision contains a usable position."""
        return self.position is not None

    def __bool__(self):
        """Treat a usable position as a successful approach."""
        return self.succeeded


@dataclass(frozen=True)
class QueryTargetDecision:
    """Result of trying a bounded number of ranked query candidates."""

    candidate: Optional[Mapping[str, Any]]
    approach: Optional[ApproachDecision]
    attempts: int
    succeeded: bool
    reason: str = ''


def _voxel_key(position, voxel_size):
    return tuple(np.floor(np.asarray(position) / voxel_size).astype(int))


def select_approach_goal(
    snapshot: QueryApproachSnapshot,
    object_pos,
    policy: ApproachPolicy,
    *,
    is_traversable_query=False,
):
    """Choose a safe approach from one immutable map-derived snapshot."""
    object_pos = np.asarray(object_pos, dtype=np.float32).reshape(-1)
    diagnostics = dict(snapshot.filtered_diagnostics)
    diagnostics.update({
        'total_voxels': snapshot.total_voxels,
        'distance_rejected': 0,
        'traversable_candidates': 0,
        'robot_side_rejected': 0,
        'selected': 0,
    })

    if is_traversable_query:
        key = _voxel_key(object_pos, snapshot.voxel_size)
        diagnostics['selected'] = 1
        return ApproachDecision(
            key=key,
            position=object_pos,
            safe=True,
            diagnostics=diagnostics,
        )

    robot_position = snapshot.robot_position
    if robot_position is None and policy.require_robot_pose:
        diagnostics['robot_pose_missing'] = 1
        return ApproachDecision(
            key=None,
            position=None,
            safe=False,
            diagnostics=diagnostics,
            reason='robot_pose_missing',
        )

    traversable_candidates = []
    for voxel in snapshot.voxels:
        distance_to_object = float(
            np.linalg.norm(voxel.position[:2] - object_pos[:2]))
        if not (
            policy.min_distance_m
            <= distance_to_object
            <= policy.max_distance_m
        ):
            diagnostics['distance_rejected'] += 1
            continue
        traversable_candidates.append((voxel, distance_to_object))
    diagnostics['traversable_candidates'] = len(traversable_candidates)

    best = None
    best_rank = None
    for voxel, distance_to_object in traversable_candidates:
        position = voxel.position
        robot_distance = 0.0
        height_cost = 0.0
        same_side = True
        if robot_position is not None:
            robot_distance = float(
                np.linalg.norm(position[:2] - robot_position[:2]))
            height_cost = abs(float(position[2] - robot_position[2]))
            robot_direction = robot_position[:2] - object_pos[:2]
            candidate_direction = position[:2] - object_pos[:2]
            direction_norm = float(
                np.linalg.norm(robot_direction)
                * np.linalg.norm(candidate_direction)
            )
            if direction_norm > 1e-9:
                side_cosine = float(np.dot(
                    robot_direction,
                    candidate_direction,
                ) / direction_norm)
                same_side = side_cosine >= policy.same_side_min_cosine
            if policy.require_robot_side and not same_side:
                diagnostics['robot_side_rejected'] += 1
                continue
        cost = (
            abs(distance_to_object - policy.desired_distance_m)
            + policy.robot_distance_weight * robot_distance
            + 0.2 * height_cost
            - 0.1 * float(voxel.confidence)
        )
        preference_rank = (
            int(policy.prefer_robot_side and not same_side),
            cost,
        )
        if best_rank is None or preference_rank < best_rank:
            best_rank = preference_rank
            best = (voxel, same_side)

    if best is not None:
        voxel, same_side = best
        diagnostics['selected'] = 1
        diagnostics['same_side'] = bool(same_side)
        return ApproachDecision(
            key=voxel.key,
            position=voxel.position,
            safe=True,
            diagnostics=diagnostics,
        )

    if policy.require_safe:
        return ApproachDecision(
            key=None,
            position=None,
            safe=False,
            diagnostics=diagnostics,
            reason='no_safe_approach',
        )

    if robot_position is None:
        return ApproachDecision(
            key=None,
            position=None,
            safe=False,
            diagnostics=diagnostics,
            reason='robot_pose_missing_for_fallback',
        )
    fallback_position = np.array(object_pos, dtype=np.float32, copy=True)
    direction = robot_position[:2] - object_pos[:2]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return ApproachDecision(
            key=None,
            position=None,
            safe=False,
            diagnostics=diagnostics,
            reason='robot_at_target',
        )
    fallback_position[:2] = (
        object_pos[:2]
        + direction / direction_norm * policy.desired_distance_m
    )
    fallback_position[2] = robot_position[2]
    diagnostics['fallback_used'] = 1
    return ApproachDecision(
        key=_voxel_key(fallback_position, snapshot.voxel_size),
        position=fallback_position,
        safe=False,
        diagnostics=diagnostics,
        fallback_used=True,
        reason='simulation_standoff_fallback',
    )


def plan_query_target(
    snapshot: QueryTargetSnapshot,
    max_attempts: int,
    evaluate: Callable[[Mapping[str, Any]], ApproachDecision],
):
    """Evaluate ranked candidates without publishing or mutating map state."""
    attempt_limit = max(0, int(max_attempts))
    candidates = snapshot.candidates[:attempt_limit]
    if not candidates:
        return QueryTargetDecision(
            candidate=None,
            approach=None,
            attempts=0,
            succeeded=False,
            reason='no_candidates',
        )

    for attempt, candidate in enumerate(candidates, start=1):
        result = evaluate(candidate)
        if result.succeeded:
            return QueryTargetDecision(
                candidate=candidate,
                approach=result,
                attempts=attempt,
                succeeded=True,
            )
    return QueryTargetDecision(
        candidate=None,
        approach=result,
        attempts=len(candidates),
        succeeded=False,
        reason='no_approachable_candidate',
    )


@dataclass(frozen=True)
class QueryClusterPolicy:
    """Evidence, extent, color and distance rules for ranking spatial instances."""

    query_max_candidates: int
    query_cluster_radius_m: float
    query_cluster_min_voxels: int
    query_cluster_min_evidence: float
    query_class_max_extent_m: tuple
    query_cluster_support_weight: float
    query_distance_weight: float
    query_min_color_score: float
    query_color_weight: float = 0.35
    query_min_color_support_ratio: float = 0.3


def rank_query_clusters(
    candidates, policy, robot_position=None, query_class_idx=None, query_color=None,
    map_time_sec=None, recent_observation_sec=60.0,
):
    """Cluster semantic evidence before applying color and return ranked instances."""
    diagnostics = {
        'cluster_count': 0,
        'color_rejected_clusters': 0,
        'max_cluster_color_score': 0.0,
        'max_cluster_color_support_ratio': 0.0,
    }
    if not candidates:
        return [], diagnostics

    color_weight = policy.query_color_weight

    def candidate_rank(item):
        semantic_score = float(item['score'])
        if query_color is None:
            return semantic_score
        return (
            (1.0 - color_weight) * semantic_score
            + color_weight * float(item.get('color_score', 0.0))
        )

    candidates = sorted(
        candidates, key=candidate_rank, reverse=True
    )[:max(policy.query_max_candidates, 1)]
    positions = np.asarray([item['pos'][:2] for item in candidates])
    tree = cKDTree(positions)
    neighborhoods = tree.query_ball_point(positions, policy.query_cluster_radius_m)
    visited = np.zeros(len(candidates), dtype=bool)
    clusters = []

    for seed in range(len(candidates)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        indices = []
        while stack:
            current = stack.pop()
            indices.append(current)
            for neighbor in neighborhoods[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        clusters.append(indices)

    diagnostics['cluster_count'] = len(clusters)
    ranked_clusters = []
    for indices in clusters:
        cluster = [candidates[index] for index in indices]
        total_evidence = sum(item['evidence'] for item in cluster)
        if len(cluster) < policy.query_cluster_min_voxels:
            continue
        if total_evidence < policy.query_cluster_min_evidence:
            continue

        cluster_positions = np.asarray([item['pos'] for item in cluster])
        horizontal_extent = float(np.max(
            np.ptp(cluster_positions[:, :2], axis=0)))
        if (
            query_class_idx is not None
            and query_class_idx < len(policy.query_class_max_extent_m)
            and horizontal_extent > policy.query_class_max_extent_m[query_class_idx]
        ):
            continue

        evidence_weights = np.asarray([
            min(item['evidence'], 10.0) for item in cluster
        ], dtype=np.float64)
        scores = np.asarray([item['score'] for item in cluster])
        semantic_score = 0.8 * float(np.average(scores, weights=evidence_weights))
        semantic_score += 0.2 * float(np.max(scores))
        selection_score = semantic_score
        cluster_color_score = 0.0
        color_support_ratio = 0.0
        mean_color_rgb = None
        if query_color is not None:
            color_scores = np.asarray([
                float(item.get('color_score', 0.0)) for item in cluster
            ], dtype=np.float64)
            cluster_color_score = float(np.average(
                color_scores, weights=evidence_weights))
            color_support_ratio = float(np.average(
                color_scores >= policy.query_min_color_score,
                weights=evidence_weights,
            ))
            diagnostics['max_cluster_color_score'] = max(
                diagnostics['max_cluster_color_score'],
                cluster_color_score,
            )
            diagnostics['max_cluster_color_support_ratio'] = max(
                diagnostics['max_cluster_color_support_ratio'],
                color_support_ratio,
            )
            if (
                cluster_color_score < policy.query_min_color_score
                or color_support_ratio
                < policy.query_min_color_support_ratio
            ):
                diagnostics['color_rejected_clusters'] += 1
                continue
            selection_score = (
                (1.0 - color_weight) * semantic_score
                + color_weight * cluster_color_score
            )

            valid_colors = []
            valid_color_weights = []
            for item, item_weight in zip(cluster, evidence_weights):
                color_rgb = item.get('color_rgb')
                if color_rgb is None:
                    continue
                color_rgb = np.asarray(color_rgb, dtype=np.float64).reshape(-1)
                if color_rgb.size == 3 and np.all(np.isfinite(color_rgb)):
                    valid_colors.append(color_rgb)
                    valid_color_weights.append(item_weight)
            if valid_colors:
                mean_color_rgb = np.average(
                    np.asarray(valid_colors),
                    axis=0,
                    weights=np.asarray(valid_color_weights),
                ).astype(np.float32)

        support = min(np.log1p(total_evidence) / np.log(31.0), 1.0)
        centroid = np.average(
            cluster_positions,
            axis=0,
            weights=evidence_weights,
        )
        distance = 0.0
        if robot_position is not None:
            distance = float(np.linalg.norm(centroid[:2] - robot_position[:2]))
        utility = (
            selection_score
            + policy.query_cluster_support_weight * support
            - policy.query_distance_weight * distance
        )
        representative = max(cluster, key=lambda item: item['score'])
        cluster_result = {
            'key': representative['key'],
            'pos': centroid.astype(np.float32),
            'score': selection_score,
            'semantic_score': semantic_score,
            'similarity': max(item['similarity'] for item in cluster),
            'class_probability': max(
                item['class_probability'] for item in cluster),
            'color_score': cluster_color_score,
            'color_support_ratio': color_support_ratio,
            'color_rgb': mean_color_rgb,
            'voxel_count': len(cluster),
            'evidence': total_evidence,
            'utility': utility,
            'voxel_keys': tuple(item['key'] for item in cluster),
            'class_support_mean': float(np.average(
                [item['class_probability'] for item in cluster], weights=evidence_weights)),
            'uncertainty_mean': float(np.average(
                [item.get('uncertainty', 1.0) for item in cluster], weights=evidence_weights)),
        }
        cluster_result.update(summarize_observations(
            cluster, evidence_weights, map_time_sec, recent_observation_sec))
        ranked_clusters.append(cluster_result)
    ranked_clusters.sort(key=lambda item: item['utility'], reverse=True)
    return ranked_clusters, diagnostics
