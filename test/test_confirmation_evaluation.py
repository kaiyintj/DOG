"""Reference association stays fixed and unknown/bypass results stay explicit."""

from types import SimpleNamespace

import numpy as np

from semantic_mapping.offline.confirmation_evaluation import evaluate_confirmation


def recording(before=(0, 0, 0), after=(2, 0, 0), policy='always'):
    pose = {'stamp': 1., 'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}
    event = {'event': 'CANDIDATE_SELECTED' if policy != 'none' else 'GOAL_RELEASED',
             'policy': policy, 'candidate_position': before,
             'candidate_before': {'class_support_mean': .9}}
    ending = {**event, 'event': 'GOAL_RELEASED', 'state': 'executing',
              'observed_position': after, 'new_class_distribution': [0, 0, 0, .8, .2, 0, 0, 0],
              'confirmation_sim_sec': 2., 'confirmation_wall_sec': 3., 'sources_used': 2,
              'reason': 'new_sources_support_candidate'}
    if policy == 'none':
        ending.update(observed_position=None, new_class_distribution=None,
                      reason='feedback_disabled', sources_used=0)
        events = [ending]
    else:
        events = [event, ending]
    return {'alignment': {'truth': {**pose, 'frame': 'world'},
                          'estimate': {**pose, 'frame': 'odom'}}, 'confirmation_events': events}


def references(monkeypatch):
    def reference(name, x):
        return SimpleNamespace(instance_id=name, class_name='chair', measure=lambda position: {
            'reference_instance_id': name,
            'visual_surface_distance_3d_m': float(np.linalg.norm(position - [x, 0, 0]))})
    monkeypatch.setattr('semantic_mapping.offline.confirmation_evaluation.load_target_references',
                        lambda case: {'A': reference('A', 0), 'B': reference('B', 2)})


def test_after_measurement_never_rebinds_to_nearer_instance(monkeypatch):
    references(monkeypatch)
    result = evaluate_confirmation({'target_class': 'chair', 'ontology_profile': 'indoor7'},
                                   recording())['attempts'][0]
    assert result['reference_instance_id'] == 'A'
    assert result['after_fixed_geometry']['reference_instance_id'] == 'A'
    assert result['after_fixed_geometry']['visual_surface_distance_3d_m'] == 2


def test_ambiguous_or_far_initial_region_stays_unknown(monkeypatch):
    references(monkeypatch)
    result = evaluate_confirmation({'target_class': 'chair', 'ontology_profile': 'indoor7'},
                                   recording(before=(1, 0, 0)))['attempts'][0]
    assert result['reference_instance_id'] is None
    assert result['after_fixed_geometry'] is None


def test_no_feedback_does_not_claim_new_confirmation_or_position_improvement(monkeypatch):
    references(monkeypatch)
    result = evaluate_confirmation({'target_class': 'chair', 'ontology_profile': 'indoor7'},
                                   recording(policy='none'))['attempts'][0]
    assert not result['fresh_confirmation_obtained']
    assert result['position_change_m'] is None
    assert result['class_support_change'] is None


def test_confirmed_but_unapproachable_candidate_still_records_confirmation(monkeypatch):
    references(monkeypatch)
    values = recording()
    decision = {**values['confirmation_events'][-1], 'event': 'OBSERVATION_DECISION',
                'state': 'confirmed'}
    values['confirmation_events'].insert(1, decision)
    values['confirmation_events'][-1].update(
        event='CANDIDATE_REJECTED', state='rejected', reason='no_usable_approach')
    result = evaluate_confirmation({'target_class': 'chair', 'ontology_profile': 'indoor7'},
                                   values)['attempts'][0]
    assert result['fresh_confirmation_obtained']
    assert result['ending_state'] == 'rejected'
