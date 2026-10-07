"""Adapter tests cover shared turn budgets, TF loss and replaced requests."""

from types import SimpleNamespace

import numpy as np
from tf2_ros import TransformException

from semantic_mapping.runtime.confirmation_ros import TargetConfirmationAdapter
from semantic_mapping.runtime.target_confirmation import ConfirmationPolicy, ConfirmationSession


def adapter(monkeypatch):
    monkeypatch.setattr('semantic_mapping.runtime.confirmation_ros.time.monotonic', lambda: 1.)
    published = []
    value = TargetConfirmationAdapter.__new__(TargetConfirmationAdapter)
    value.policy = ConfirmationPolicy(mode='always')
    value.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=101_000_000_000)),
        get_robot_position=lambda: np.zeros(3), _points_transform_matrix=lambda *args: np.eye(4),
        select_class_query_target=lambda *args: None, query_retry_pending=False, odom_frame='odom',
        _publish_query_goal=lambda *args: published.append(args))
    candidate = {'pos': np.array([0., 0, 2.]), 'voxel_keys': ((0, 0, 20),),
                 'class_support_mean': .9, 'uncertainty_mean': .1, 'evidence': 10.,
                 'observed_age_sec': 1.}
    value.node._last_query_clusters = [candidate]
    value.status = SimpleNamespace(publish=lambda *args: None)
    value.motion = SimpleNamespace(publish=lambda message: published.append(message))
    value.motion_owned = True
    value.geometry = {'source_frame': 'lidar', 'lidar_to_camera': np.eye(4)}
    value.session = ConfirmationSession(value.policy, 0, 100, 0, 0)
    value.class_index, value.color = 0, None
    value.latest_map_sec, value.latest_source_ns = 101., 101_000_000_000
    value.latest_view = None
    value.waiting_for_input = False
    value.rejected_regions = []
    value.last_heading, value.probe_heading, value.turn_used = .2, None, .4
    return value, published


def test_candidate_switch_preserves_realized_turn_budget(monkeypatch):
    value, _ = adapter(monkeypatch)
    value.select_next()
    assert value.turn_used == .4 and value.last_heading == .2
    value.session.status = 'rejected'
    value.select_next()
    assert value.turn_used == .4 and value.session.attempts == 2


def test_tf_loss_stops_confirmation_and_prevents_goal_release(monkeypatch):
    value, published = adapter(monkeypatch)
    value.select_next()

    def unavailable(*args):
        raise TransformException('historical view unavailable')

    value.node._points_transform_matrix = unavailable
    value.tick()
    assert value.session.status == 'input_invalid'
    assert not value.motion_owned
    assert len(published) == 1 and published[0].angular.z == 0


def test_new_unresolved_query_cancels_old_pending_confirmation(monkeypatch):
    value, _ = adapter(monkeypatch)
    value.select_next()
    value.cancel('query_replaced')
    assert value.session.status == 'input_invalid'
    assert not value.motion_owned


def test_last_source_rejection_is_terminal_before_new_candidate_selection(monkeypatch):
    value, published = adapter(monkeypatch)
    value.select_next()
    value.session.sources_used = value.policy.max_sources
    value.session.status = 'rejected'
    value.tick()
    assert value.session.status == 'unconfirmed'
    assert value.session.attempts == 1
    assert len(published) == 1 and published[0].angular.z == 0


def test_wait_expiry_emits_candidate_snapshot_before_switching(monkeypatch):
    value, _ = adapter(monkeypatch)
    events = []
    value.status.publish = lambda message: events.append(message.data)
    value.select_next()
    value.session.candidate_start_sec = 80.
    value.session.sources_used = 3
    value.tick()
    import json
    rejected = [json.loads(event) for event in events
                if json.loads(event)['event'] == 'CANDIDATE_REJECTED']
    assert len(rejected) == 1
    assert rejected[0]['sources_used'] == 3
    assert rejected[0]['reason'] == 'candidate_wait_exhausted'


def test_motion_waits_for_two_postrequest_sources_before_turning(monkeypatch):
    value, published = adapter(monkeypatch)
    value.turn_used, value.last_heading = 0., None
    value.select_next()
    value.navigation_busy = False
    value.view = lambda: {'bearing_error_rad': 0.}
    value.tick()
    assert published[-1].angular.z == 0
    value.session.sources_used = 2
    value.tick()
    assert published[-1].angular.z > 0


def test_transient_fusion_gap_waits_without_resetting_budget_or_releasing_ready_goal(monkeypatch):
    value, published = adapter(monkeypatch)
    value.select_next()
    value.session.status = 'confirmed'
    value.session.sources_used = 2
    value.session.valid_sources = 2
    value.latest_map_sec = 98.
    value.tick()
    assert value.session.status == 'observing' and value.waiting_for_input
    assert value.session.min_valid_sources == 3
    assert value.session.sources_used == 2 and value.session.start_map_sec == 100
    assert not value.motion_owned
    assert len(published) == 1 and published[0].angular.z == 0
    value.latest_map_sec = 101.
    value.navigation_busy = True
    value.view = lambda: {'bearing_error_rad': 0.}
    value.tick()
    assert not value.waiting_for_input and value.session.status == 'observing'
    assert value.session.sources_used == 2
    value.latest_map_sec = 98.
    value.session.start_wall_sec = -90.
    value.tick()
    assert value.session.status == 'unconfirmed'
