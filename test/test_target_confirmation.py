"""Confirmation budgets count real new sources and remain cumulative across candidates."""

import numpy as np
import pytest

from semantic_mapping.runtime.candidate_observation import summarize_observations
from semantic_mapping.runtime.target_confirmation import ConfirmationPolicy, ConfirmationSession


def candidate(age=90, support=.9):
    return {'pos': np.array([2., 0, .4]), 'voxel_keys': ((20, 0, 4), (21, 0, 4)),
            'observed_age_sec': age, 'class_support_mean': support, 'uncertainty_mean': .1,
            'evidence': 30., 'recent_class_support': support}


def contributions(probabilities=(.9, .1)):
    return [((20, 0, 4), .5, np.asarray(probabilities), np.array([2, 0, .4])),
            ((21, 0, 4), .5, np.asarray(probabilities), np.array([2.1, 0, .4]))]


def test_newest_isolated_voxel_cannot_make_whole_candidate_fresh():
    cluster = [{'last_effective_observed_at_sec': value} for value in (10, 10, 99)]
    summary = summarize_observations(cluster, [10, 10, 1], 100, recent_sec=60)
    assert summary['observed_age_sec'] == 90
    assert summary['fresh_support_ratio'] == pytest.approx(1 / 21)


@pytest.mark.parametrize('mode,expected', [
    ('none', False), ('always', True), ('age', True), ('age_quality', False)])
def test_four_policies_distinguish_age_from_static_record_quality(mode, expected):
    assert ConfirmationPolicy(mode=mode).needs_review(candidate()) is expected


def test_quality_policy_reviews_fresh_weak_or_unknown_age_records():
    policy = ConfirmationPolicy(mode='age_quality')
    assert policy.needs_review(candidate(age=5, support=.3))
    assert policy.needs_review(candidate(age=None))


def test_new_supported_sources_confirm_without_counting_copied_or_prequery_images():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 100_000_000_000)
    session.select(candidate(), 100)
    session.observe(('camera', 99_000_000_000), 101, 1, contributions(), True)
    assert session.sources_used == 0
    for stamp in (101, 102):
        source = ('camera', stamp * 1_000_000_000)
        session.observe(source, stamp, stamp - 100, contributions(), True)
        session.observe(source, stamp, stamp - 100, contributions(), True)
    assert session.status == 'confirmed'
    assert session.sources_used == session.valid_sources == 2


def test_changing_candidate_never_restarts_time_or_source_budget():
    session = ConfirmationSession(ConfirmationPolicy(mode='always', max_sources=3), 0, 100, 0, 0)
    session.select(candidate(), 100)
    session.observe(('camera', 101), 101, 1, [], True)
    session.select(candidate(), 102)
    assert session.sources_used == 1 and session.start_map_sec == 100
    session.tick(131, 31)
    assert session.status == 'unconfirmed'


def test_unknown_is_not_a_claim_that_static_target_disappeared():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 0, unknown_index=1)
    session.select(candidate(), 100)
    for stamp in (101, 102):
        session.observe(('camera', stamp), stamp, stamp - 100, contributions((.1, .9)), True)
    assert session.status == 'observing'
    session.tick(113, 13)
    assert session.status == 'rejected' and session.reason == 'candidate_wait_exhausted'


def test_wall_budget_works_when_simulation_is_paused():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 0)
    session.select(candidate(), 100)
    session.tick(100, 91)
    assert session.status == 'unconfirmed'


def test_rejection_at_last_source_cannot_bypass_budget_with_strong_next_candidate():
    policy = ConfirmationPolicy(mode='age_quality', max_sources=2)
    session = ConfirmationSession(policy, 0, 100, 0, 0)
    session.select(candidate(age=None), 100)
    for stamp in (101, 102):
        session.observe(('camera', stamp), stamp, stamp - 100, contributions((.1, .9)), True)
    assert session.status == 'rejected'
    session.select(candidate(age=1), 102)
    assert session.status == 'unconfirmed'
    assert session.attempts == 1


def test_last_permitted_source_can_confirm_but_publication_still_checks_time():
    session = ConfirmationSession(ConfirmationPolicy(mode='always', max_sources=2), 0, 100, 0, 0)
    session.select(candidate(), 100)
    for stamp in (101, 102):
        session.observe(('camera', stamp), stamp, stamp - 100, contributions(), True)
    session.tick(102, 2)
    assert session.status == 'confirmed'
    session.tick(131, 31)
    assert session.status == 'unconfirmed'


def test_regional_new_sources_without_known_depth_remain_unconfirmed():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 0)
    session.select(candidate(), 100)
    for stamp in (101, 102, 103):
        session.observe(('camera', stamp), stamp, stamp - 100, contributions(), False)
    assert session.status == 'observing'
    assert session.no_progress_sources == 0


def test_regional_evidence_excludes_unrelated_nearby_floor_but_keeps_changed_old_support():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 0)
    session.select(candidate(), 100)
    assert session.matches_regional_update((20, 0, 4), [.1, .9], [2., 0, .4])
    assert not session.matches_regional_update((22, 0, 0), [.1, .9], [2.2, 0, 0])
    assert session.matches_regional_update((22, 0, 4), [.8, .2], [2.2, 0, .4])
    assert not session.matches_regional_update((30, 0, 4), [.8, .2], [3., 0, .4])


def test_ready_goal_after_input_gap_requires_another_effective_source():
    session = ConfirmationSession(ConfirmationPolicy(mode='always'), 0, 100, 0, 0)
    session.select(candidate(), 100)
    for stamp in (101, 102):
        session.observe(('camera', stamp), stamp, stamp - 100, contributions(), True)
    assert session.status == 'confirmed'
    session.status, session.min_valid_sources = 'observing', 3  # Adapter's input-gap transition.
    session.observe(('camera', 103), 103, 3, [], True)
    assert session.status == 'observing' and session.valid_sources == 2
    session.observe(('camera', 104), 104, 4, contributions(), True)
    assert session.status == 'confirmed' and session.sources_used == 4
