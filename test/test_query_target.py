import numpy as np

from semantic_mapping.runtime.query_target import (
    ApproachPolicy,
    ApproachDecision,
    QueryApproachSnapshot,
    QueryApproachVoxel,
    QueryTargetSnapshot,
    plan_query_target,
    select_approach_goal,
)


def test_query_target_planner_stops_at_two_attempts():
    snapshot = QueryTargetSnapshot(
        candidates=(
            {'id': 'first'},
            {'id': 'second'},
            {'id': 'third'},
        ),
        map_revision=7,
    )
    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate['id'])
        success = candidate['id'] == 'second'
        return ApproachDecision(
            key=None, position=np.zeros(3) if success else None,
            safe=success, diagnostics={},
        )

    decision = plan_query_target(snapshot, max_attempts=2, evaluate=evaluate)

    assert decision.succeeded
    assert decision.candidate['id'] == 'second'
    assert decision.attempts == 2
    assert evaluated == ['first', 'second']


def test_approach_selection_reuses_one_snapshot_and_prefers_robot_side():
    snapshot = QueryApproachSnapshot(
        voxels=(
            QueryApproachVoxel(
                key=(20, 0, 0),
                position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
                confidence=0.3,
            ),
            QueryApproachVoxel(
                key=(-20, 0, 0),
                position=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
                confidence=1.0,
            ),
        ),
        robot_position=np.array([4.0, 0.0, 0.0], dtype=np.float32),
        voxel_size=0.1,
        total_voxels=2,
        filtered_diagnostics={
            'low_weight_rejected': 0,
            'non_traversable_rejected': 0,
            'low_confidence_rejected': 0,
        },
        map_revision=3,
    )
    policy = ApproachPolicy(
        min_distance_m=1.5,
        max_distance_m=2.5,
        desired_distance_m=2.0,
        require_robot_side=True,
    )

    decision = select_approach_goal(
        snapshot,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        policy,
    )

    assert decision.succeeded
    assert decision.safe
    assert decision.key == (20, 0, 0)
    assert decision.diagnostics['selected'] == 1


def test_planner_does_not_try_third_candidate_after_two_failures():
    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate['id'])
        return ApproachDecision(
            key=None, position=None, safe=False, diagnostics={})

    decision = plan_query_target(
        QueryTargetSnapshot(tuple({'id': i} for i in range(3))),
        2, evaluate,
    )
    assert not decision.succeeded
    assert decision.attempts == 2
    assert evaluated == [0, 1]
