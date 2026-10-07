"""Compare memory rules on identical recorded Gazebo fusion inputs, without navigation claims."""

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

import numpy as np

from semantic_mapping.gazebo.indoor_benchmark import _pose_matrix
from semantic_mapping.runtime.ga_bsvm_node import GABsvmNode
from semantic_mapping.runtime.query_target import (
    QueryTargetSnapshot, plan_query_target, select_approach_goal,
)
from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.voxel_map import VoxelMap


def make_query_map(parameters, retention, ttl):
    node = object.__new__(GABsvmNode)
    profile = open_profile(parameters['ontology_profile'])
    for name, value in parameters.items():
        if name.startswith('query_'):
            setattr(node, name, value)
    node.query_class_max_extent_m = profile.query_max_extents
    node.traversable_class_ids = profile.traversable_ids
    node.semantic_cost_dict = dict(enumerate(profile.semantic_costs))
    node.voxel_map = VoxelMap(
        voxel_size=parameters['voxel_size'], K=profile.K, class_colors=profile.colors,
        evidence_decay=retention, **{name: parameters[name] for name in (
            'evidence_prior', 'evidence_strength', 'evidence_decay_reference_sec',
            'max_frame_evidence', 'source_history_size', 'max_total_evidence',
            'uncertainty_entropy_weight',
        )})
    return node, profile, ttl


def replay(run_directory):
    recording = json.loads((run_directory / 'result.json').read_text())
    case = recording['case']
    trace = run_directory / 'fusion_trace'
    parameters = json.loads((trace / 'parameters.json').read_text())
    events = [json.loads(line) for line in (trace / 'events.jsonl').read_text().splitlines()]
    checkpoints = recording['recording'].get('motion_observation', {}).get('checkpoints', [])
    if not checkpoints:
        raise ValueError('Motion checkpoints are required; preserve failed runs separately')
    # No TTL below separates the decay-rate effect. Original preset is an extra control.
    variants = {
        'original_decay': (.995, 0.0),
        'slow_half_life_60s': (2 ** (-.1 / 60), 0.0),
        'no_decay': (1.0, 0.0),
        'original_preset': (.995, 300.0),
    }
    maps = {name: make_query_map(parameters, *settings) for name, settings in variants.items()}
    trajectory = recording['recording']['trajectory']['estimate']
    pose_stamps = [pose['stamp'] for pose in trajectory]
    robot_position = None
    for node, _, _ in maps.values():
        node.get_robot_position = lambda: robot_position
    alignment = recording['recording']['alignment']
    world_from_odom = _pose_matrix(alignment['truth'], 'world') @ np.linalg.inv(
        _pose_matrix(alignment['estimate'], 'odom'))
    truths = [entry for entry in case['targets'] if entry['class'] == case['target_class']]
    profile = maps['no_decay'][1]
    class_id = profile.classes.index(case['target_class'])
    reference = None
    results, timeline = [], []
    checkpoint_index, last_sample = 0, -math.inf
    observation_times = []
    reference_input_times = []

    def sample(stamp, label):
        nonlocal reference, robot_position
        index = bisect_right(pose_stamps, stamp) - 1
        robot_position = None if index < 0 else np.asarray(trajectory[index]['position'])
        selected = {name: node.select_class_query_target(class_id)[0]
                    for name, (node, _, _) in maps.items()}
        if label == 'before_motion' and selected['no_decay'] is not None:
            reference = np.asarray(selected['no_decay']['pos'])
        values = {}
        for name, target in selected.items():
            value = {'queryable': target is not None}
            if target is not None:
                position = np.asarray(target['pos'])
                world = (world_from_odom @ np.r_[position, 1])[:3]
                nearest = min(truths, key=lambda gt: math.dist(world[:2], gt['pose'][:2]))
                value.update(position_odom=position.tolist(), position_world=world.tolist(),
                             nearest_model_id=nearest['id'],
                             origin_error_xy_m=math.dist(world[:2], nearest['pose'][:2]),
                             class_support=float(target['class_probability']),
                             cluster_evidence=float(target['evidence']))
            node = maps[name][0]
            value['approachable'] = False
            if target is not None:
                limit = parameters['query_fallback_max_attempts']
                ranked = node._ranked_query_candidates(target)[:limit]
                snapshot = node._build_query_approach_snapshot(tuple(t['pos'] for t in ranked))
                policy = node._query_approach_policy()
                decision = plan_query_target(
                    QueryTargetSnapshot(tuple(ranked)), limit,
                    lambda candidate: select_approach_goal(snapshot, candidate['pos'], policy))
                value['approachable'] = bool(decision.succeeded)
                if decision.succeeded:
                    actual = (world_from_odom @ np.r_[decision.candidate['pos'], 1])[:3]
                    nearest = min(truths, key=lambda gt: math.dist(actual[:2], gt['pose'][:2]))
                    value.update(approach_target_world=actual.tolist(),
                                 approach_target_origin_error_xy_m=math.dist(
                                     actual[:2], nearest['pose'][:2]))
            if reference is not None:
                near = [voxel for voxel in node.voxel_map.voxels.values()
                        if np.linalg.norm(voxel['pos'] - reference) <= .45]
                value['reference_area_voxels'] = len(near)
                value['reference_area_evidence'] = sum(v['weight_sum'] for v in near)
                value['reference_area_last_observation_sec'] = max(
                    (v['last_observed_at_sec'] for v in near), default=None)
            values[name] = value
        return {'sim_sec': stamp, 'label': label, 'variants': values}

    for event in events:
        stamp = event['processed_sim_sec']
        while (checkpoint_index < len(checkpoints)
               and checkpoints[checkpoint_index]['sim_sec'] <= stamp):
            checkpoint = checkpoints[checkpoint_index]
            results.append(sample(checkpoint['sim_sec'], checkpoint['name']))
            checkpoint_index += 1
        if event['event'] == 'observation':
            with np.load(trace / event['file'], allow_pickle=False) as stored:
                inputs = {name: stored[name] for name in (
                    'points', 'reliability', 'logits', 'colors', 'features')
                    if name in stored.files}
            if reference is not None:
                near = np.linalg.norm(inputs['points'] - reference, axis=1) <= .45
                if np.any(near & (inputs['reliability'] > 0)):
                    reference_input_times.append(event['observation_ns'] / 1e9)
            observation_times.append(event['observation_ns'] / 1e9)
            for node, _, _ in maps.values():
                node.voxel_map.update(**inputs, timestamp_sec=event['observation_ns'] / 1e9,
                                      source_id=tuple(event['source_id']))
        else:
            for node, profile, ttl in maps.values():
                node.voxel_map.prune(
                    event['now_sec'], dynamic_class_ids=profile.dynamic_ids,
                    dynamic_ttl_sec=parameters['dynamic_voxel_ttl_sec'], stale_ttl_sec=ttl,
                    center=event['center'], max_distance_m=parameters['voxel_prune_radius_m'],
                    max_count=parameters['voxel_max_count'])
        if stamp - last_sample >= 5:
            timeline.append(sample(stamp, 'periodic'))
            last_sample = stamp
    for checkpoint in checkpoints[checkpoint_index:]:
        results.append(sample(checkpoint['sim_sec'], checkpoint['name']))
    summary = {}
    for name in variants:
        errors = [row['variants'][name]['origin_error_xy_m'] for row in results
                  if row['variants'][name]['queryable']]
        summary[name] = {'queryable_checkpoints': len(errors), 'checkpoints': len(results),
                         'approachable_checkpoints': sum(
                             row['variants'][name]['approachable'] for row in results),
                         'mean_origin_error_xy_m': None if not errors else float(np.mean(errors)),
                         'max_origin_error_xy_m': None if not errors else max(errors)}
    report = {
        'scope': 'same_real_projected_inputs_fusion_query_only',
        'navigation_outcome_is_only_for_online_no_decay': recording['passed'],
        'motion_completed': recording['recording']['motion_observation']['completed'],
        'reference_center_odom': None if reference is None else reference.tolist(),
        'reference_area_projected_observation_times': reference_input_times,
        'observations': len(observation_times), 'summary': summary,
        'checkpoints': results, 'timeline': timeline,
    }
    (run_directory / 'paired_replay.json').write_text(
        json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    result = replay(args.run_directory)
    print(json.dumps(result['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
