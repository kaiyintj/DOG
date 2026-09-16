"""
ROS-independent query-target planning primitives.

The ROS node owns message transport and map adaptation.  This module owns the
small decision that turns a ranked, read-only candidate snapshot into the
first candidate with a usable approach.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np


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
