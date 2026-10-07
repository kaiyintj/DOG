"""Static confirmation diagnostics; fixed geometry association is an offline proxy."""

import numpy as np

from semantic_mapping.gazebo.indoor_benchmark import _pose_matrix
from semantic_mapping.offline.static_geometry import load_target_references
from semantic_mapping.runtime.semantic_profile import open_profile


def evaluate_confirmation(case, recording):
    events = recording.get('confirmation_events', [])
    selected = [(index, value) for index, value in enumerate(events)
                if value['event'] == 'CANDIDATE_SELECTED'
                or (value['event'] == 'GOAL_RELEASED' and value['policy'] == 'none')]
    result = {'association': 'offline_visual_mesh_proximity_proxy',
              'binding_surface_distance_m': .15, 'binding_ambiguity_gap_m': .10,
              'attempts': []}
    if not selected or not recording.get('alignment'):
        return result
    alignment = recording['alignment']
    world_from_odom = _pose_matrix(alignment['truth'], 'world') @ np.linalg.inv(
        _pose_matrix(alignment['estimate'], 'odom'))
    references = [value for value in load_target_references(case).values()
                  if value.class_name == case['target_class']]

    def world(position):
        return world_from_odom[:3, :3] @ np.asarray(position) + world_from_odom[:3, 3]

    for item, (index, event) in enumerate(selected):
        stop = selected[item + 1][0] if item + 1 < len(selected) else len(events)
        ending = events[stop - 1]
        before = world(event['candidate_position'])
        measured = sorted(((reference.measure(before), reference) for reference in references),
                          key=lambda value: value[0]['visual_surface_distance_3d_m'])
        reference = None
        reason = 'no_same_class_reference'
        if measured:
            distance = measured[0][0]['visual_surface_distance_3d_m']
            gap = (measured[1][0]['visual_surface_distance_3d_m'] - distance
                   if len(measured) > 1 else float('inf'))
            if distance > result['binding_surface_distance_m']:
                reason = 'initial_candidate_too_far_from_visual_surface'
            elif gap < result['binding_ambiguity_gap_m']:
                reason = 'initial_geometry_association_ambiguous'
            else:
                reference, reason = measured[0][1], 'initial_geometry_proxy_bound'
        observed = ending.get('observed_position')
        after = None if observed is None else world(observed)
        baseline = event['candidate_before']
        distribution = ending.get('new_class_distribution')
        class_index = open_profile(case['ontology_profile']).classes.index(case['target_class'])
        new_support = None if distribution is None else distribution[class_index]
        result['attempts'].append({
            'reference_instance_id': None if reference is None else reference.instance_id,
            'binding_reason': reason, 'before_world_xyz': before.tolist(),
            'after_world_xyz': None if after is None else after.tolist(),
            'position_change_m': None if after is None else float(np.linalg.norm(after - before)),
            'before_class_support': baseline.get('class_support_mean'),
            'new_class_support': new_support,
            'class_support_change': None if new_support is None else
            new_support - baseline['class_support_mean'],
            'before_fixed_geometry': None if reference is None else reference.measure(before),
            'after_fixed_geometry': None if reference is None or after is None else
            reference.measure(after),
            'ending_event': ending['event'], 'ending_state': ending['state'],
            'confirmation_sim_sec': ending['confirmation_sim_sec'],
            'confirmation_wall_sec': ending['confirmation_wall_sec'],
            'sources_used': ending['sources_used'],
            'new_coverage_proxy': ending.get('new_coverage_proxy'),
            'fresh_confirmation_obtained': any(
                value.get('reason') == 'new_sources_support_candidate'
                for value in events[index:stop]),
        })
    return result
