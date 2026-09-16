"""Run one isolated indoor Gazebo case and write a deletable raw run."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import traceback
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import numpy as np
import psutil
from scipy.spatial.transform import Rotation
import yaml


MAX_POSE_PAIR_SKEW_SEC = 0.15


def _pose_matrix(pose, frame):
    if pose['frame'] != frame:
        raise ValueError(f'Expected {frame} pose, received {pose["frame"]}')
    position = np.asarray(pose['position'], dtype=float)
    quaternion = np.asarray(pose['orientation'], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError('Invalid pose dimensions')
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError('Non-finite pose')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def evaluate_case(case, recording):
    """Score observed results, keeping model origins and unmeasured metrics explicit."""
    metrics = {name: 'not_evaluated' for name in (
        'safe_approach_surface_distance', 'valid_goal_rate', 'collision_rate',
        'navigation_success_rate', 'spl')}
    result = {
        'passed': False,
        'reason': recording.get('termination', 'incomplete'),
        'acceptance_scope': 'not_passed',
        'metrics': metrics,
    }
    if not recording.get('readiness', {}).get('passed'):
        return result
    if case['expected'] == 'start_validation':
        passed = (
            recording.get('termination') == 'start_validation_complete'
            and bool(recording.get('alignment'))
            and bool(recording.get('stationary'))
            and not recording.get('target_poses')
            and not recording.get('goal_poses')
            and not recording.get('bridge_events'))
        result.update(
            passed=passed,
            reason='start_validation_pass' if passed else 'start_validation_failed',
            acceptance_scope='start_validation' if passed else 'not_passed',
        )
        return result
    query = recording.get('query', {})
    targets, goals = recording.get('target_poses', []), recording.get('goal_poses', [])
    if not query.get('sent'):
        result['reason'] = 'query_not_sent'
        return result
    if case['expected'] == 'no_target_or_goal':
        metrics['false_goal_publication'] = bool(goals)
        passed = (
            recording['termination'] == 'observation_complete'
            and query.get('ack') == case['query_status']
            and query.get('observed_wall_sec', 0) >= case['negative_observation_wall_sec']
            and query.get('observed_sim_sec', 0) > 0
            and not targets and not goals and not recording.get('bridge_events'))
        result.update(
            passed=passed,
            reason='negative_pass' if passed else 'negative_failed',
            acceptance_scope='fail_closed_case' if passed else 'not_passed',
        )
        return result
    if not targets or query.get('ack') != 'accepted':
        return result
    try:
        alignment = recording['alignment']
        truth, estimate = alignment['truth'], alignment['estimate']
        if abs(truth['stamp'] - estimate['stamp']) > MAX_POSE_PAIR_SKEW_SEC:
            raise ValueError('Unsynchronized world/odom alignment')
        world_from_odom = _pose_matrix(truth, 'world') @ np.linalg.inv(
            _pose_matrix(estimate, 'odom'))
        target = targets[-1]
        if target['stamp'] < query['stamp']:
            raise ValueError('Pre-query target pose cannot satisfy a case')
        target_world = (world_from_odom @ _pose_matrix(target, 'odom'))[:3, 3]
        candidates = [t for t in case['targets'] if t['class'] == case['target_class']]
        if not candidates or case['instance_policy'] != 'nearest_same_class_gt':
            raise ValueError('A smoke case requires same-class GT model origins')
        nearest = min(candidates, key=lambda t: math.dist(target_world[:2], t['pose'][:2]))
        metrics.update(
            nearest_gt_id=nearest['id'], target_world_xyz=target_world.tolist(),
            target_to_gt_model_origin_xy_m=math.dist(target_world[:2], nearest['pose'][:2]),
            target_to_gt_model_origin_3d_m=math.dist(target_world, nearest['pose'][:3]),
            goal_to_estimated_target_xy_m='not_evaluated',
            final_robot_to_goal_gt_xy_m='not_evaluated')
        if not goals:
            result['reason'] = 'safe_approach_not_published'
            return result
        goal = goals[-1]
        if goal['stamp'] < query['stamp']:
            raise ValueError('Pre-query goal pose cannot satisfy a case')
        goal_world = (world_from_odom @ _pose_matrix(goal, 'odom'))[:3, 3]
        final_truth = _pose_matrix(recording['final_robot']['truth'], 'world')[:3, 3]
        metrics.update(
            goal_world_xyz=goal_world.tolist(),
            goal_to_estimated_target_xy_m=math.dist(goal_world[:2], target_world[:2]),
            final_robot_to_goal_gt_xy_m=math.dist(goal_world[:2], final_truth[:2]))
        accepted = {(event['sequence'], event.get('nav2_goal_id'))
                    for event in recording['bridge_events'] if event['event'] == 'ACCEPTED'}
        completed = any(
            event['event'] == 'SUCCEEDED' and event['sequence'] == len(goals)
            and event.get('nav2_goal_id')
            and (event['sequence'], event['nav2_goal_id']) in accepted
            for event in recording['bridge_events'])
        passed = (
            recording['termination'] == 'nav_terminal' and completed
            and metrics['target_to_gt_model_origin_xy_m'] <= case['target_origin_tolerance_xy_m']
            and metrics['final_robot_to_goal_gt_xy_m'] <= .5)
        result.update(
            passed=bool(passed),
            reason='navigation_chain_smoke_pass' if passed else 'navigation_failed',
            acceptance_scope='functional_chain_only' if passed else 'not_passed',
            formal_safe_approach_passed=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        result['reason'] = f'invalid_observation: {exc}'
    return result


def _load_case(manifest_path, case_id):
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark manifest schema_version')
    if not manifest.get('scenario_id'):
        raise ValueError('Benchmark manifest requires scenario_id')
    matches = [entry for entry in manifest['cases'] if entry['id'] == case_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown case or duplicate id: {case_id}')
    case = {**manifest, **matches[0]}
    case.pop('cases')
    starts = [entry for entry in case.pop('robot_starts') if entry['id'] == case['start']]
    if len(starts) != 1:
        raise ValueError(f'Unknown robot start: {case["start"]}')
    case['robot_start'] = starts[0]
    from ament_index_python.packages import get_package_share_directory

    world = (Path(get_package_share_directory(case['world_package']))
             / 'worlds' / case['world_file'])
    root = ET.parse(world).getroot().find('world')
    if case['gt_frame'] != 'world' or case['gt_reference'] != 'sdf_top_level_model_origin':
        raise ValueError('Only explicit world-frame SDF model-origin GT is supported')
    absent_assets = case.get('world_absent_assets', [])
    if (not isinstance(absent_assets, list)
            or any(not isinstance(asset, str) or not asset for asset in absent_assets)):
        raise ValueError('world_absent_assets must be a list of asset URIs')
    world_assets = {uri.text for uri in root.iter('uri') if uri.text}
    unexpected_assets = sorted(set(absent_assets) & world_assets)
    if unexpected_assets:
        raise ValueError(f'Expected world-absent asset is present: {unexpected_assets[0]}')
    for target in case['targets']:
        model = root.find(f'model[@name="{target["gazebo_model"]}"]')
        if model is None:
            raise ValueError(f'GT model missing: {target["gazebo_model"]}')
        pose = model.find('pose')
        if (pose is None or pose.get('relative_to', '') or pose.get('frame', '')
                or [float(v) for v in pose.text.split()] != target['pose']
                or model.findtext('include/uri') != target['asset']):
            raise ValueError(f'GT pose/asset does not match SDF: {target["id"]}')
    case['world_path'] = str(world.resolve())
    if case['expected'] not in (
            'navigation_success', 'no_target_or_goal', 'start_validation'):
        raise ValueError('Unsupported expected outcome')
    if case.get('query_status') == 'accepted' and case['expected'] == 'no_target_or_goal':
        if not absent_assets:
            raise ValueError('Accepted absent-target control requires world_absent_assets')
    case.setdefault('startup_timeout_wall_sec', 180.0)
    for name in ('startup_timeout_wall_sec', 'warmup_sim_sec', 'query_timeout_wall_sec',
                 'negative_observation_wall_sec'):
        if not math.isfinite(float(case[name])) or float(case[name]) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not isinstance(case['seed'], int) or not 0 <= case['seed'] < 2**32:
        raise ValueError('seed must be a Gazebo uint32')
    start_values = [*starts[0]['pose'], starts[0]['spawn_height_m']]
    if len(start_values) != 4 or not all(math.isfinite(float(v)) for v in start_values):
        raise ValueError('Invalid spawn pose')
    if case['expected'] == 'navigation_success':
        tolerance = float(case['target_origin_tolerance_xy_m'])
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Positive model-origin XY tolerance required')
    return case


def _stop_owned_process(process):
    """Stop this launch's session, including children reparented after its exit."""
    if process is None:
        return []
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    owned = {child.pid: child for child in children}
    # Popen(start_new_session=True) makes the launch PID the session ID.
    # Session membership survives reparenting; a late descendants() query does not.
    for candidate in psutil.process_iter():
        try:
            if candidate.pid != process.pid and os.getsid(candidate.pid) == process.pid:
                owned[candidate.pid] = candidate
        except (ProcessLookupError, PermissionError):
            continue
    children = list(owned.values())
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    else:
        for child in children:
            try:
                child.send_signal(signal.SIGINT)
            except psutil.NoSuchProcess:
                pass
    _, alive = psutil.wait_procs(children, timeout=1)
    for child in alive:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    remaining = []
    for child in alive:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                remaining.append(child.pid)
        except psutil.NoSuchProcess:
            pass
    return remaining


def main(argv=None):
    """Run one fresh local simulation and write a non-overwriting result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--fusion-input', choices=['full_posterior', 'hard_mask_confidence'],
        default='full_posterior',
        help='Compare existing fusion inputs while retaining the same navigation chain.')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'format': 'semantic_mapping_gazebo_run',
        'format_version': 1,
        'passed': False,
        'reason': 'preflight_incomplete',
        'case_id': args.case,
        'fusion_input': args.fusion_input,
    }
    observer = process = None
    exit_code = 2
    try:
        case = _load_case(args.manifest, args.case)
        report['case'] = case
        report['scenario'] = {
            'id': case['scenario_id'],
            'world_package': case['world_package'],
            'world_file': case['world_file'],
            'source_commit': case.get('source_commit'),
        }
        report['profile'] = case['ontology_profile']
        domain = int(os.environ.get('ROS_DOMAIN_ID', '216'))
        uri = os.environ.get('GAZEBO_MASTER_URI', 'http://127.0.0.1:11357')
        address = urlparse(uri)
        if not 1 <= domain <= 232:
            raise ValueError('Use an isolated nonzero ROS_DOMAIN_ID (1..232)')
        if (address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost')
                or address.port is None or not 1024 <= address.port <= 65535):
            raise ValueError('Gazebo master must be an explicit localhost port >=1024')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', address.port))
        os.environ.update(ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1',
                          GAZEBO_MASTER_URI=uri, ROS_LOG_DIR=str(output / 'ros_logs'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          GAZEBO_MODEL_DATABASE_URI='')
        from huggingface_hub import snapshot_download
        from semantic_mapping.runtime.semantic_profile import open_profile

        checkpoint = snapshot_download(
            repo_id=case['model_id'], revision=case['model_revision'], local_files_only=True)
        labels = json.loads((Path(checkpoint) / 'config.json').read_text())['id2label']
        if case['expected'] != 'start_validation':
            decision = open_profile(
                case['ontology_profile'], labels).resolve_query(case['query'])
            expected_status = ('accepted' if case['expected'] == 'navigation_success'
                               else case['query_status'])
            if decision.status.value != expected_status:
                raise ValueError(
                    f'Case expects {expected_status}, '
                    f'actual checkpoint gives {decision.status.value}')
        case['checkpoint_path'] = checkpoint
        report['isolation'] = {'ros_domain_id': domain, 'gazebo_master_uri': uri,
                               'ros_localhost_only': True}
        from semantic_mapping.gazebo.observation import BenchmarkObserver

        observer = BenchmarkObserver(case)
        observer.require_empty_domain()
        start = case['robot_start']
        command = [
            'ros2', 'launch', 'semantic_mapping', 'semantic_sim.launch.py',
            'gui:=false', 'rviz:=false',
            'segformer_use_full_posterior:='
            + ('true' if args.fusion_input == 'full_posterior' else 'false'),
            f'ontology_profile:={case["ontology_profile"]}', f'model_id:={checkpoint}',
            f'world:={case["world_path"]}', f'seed:={case["seed"]}',
            f'ros_domain_id:={domain}', f'gazebo_master_uri:={uri}',
            f'world_init_x:={start["pose"][0]}', f'world_init_y:={start["pose"][1]}',
            f'world_init_z:={start["spawn_height_m"]}',
            f'world_init_heading:={start["pose"][2]}',
        ]
        report['command'] = command
        exit_code = 1
        with (output / 'simulation.log').open('w') as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                start_new_session=True)
            observer.observe(process)
            report['recording'] = observer.recording
            report.update(evaluate_case(case, observer.recording))
            try:
                children = psutil.Process(process.pid).children(recursive=True)
                report['gazebo_commands'] = [
                    child.cmdline() for child in children
                    if child.name() == 'gzserver']
            except psutil.NoSuchProcess:
                report['gazebo_commands'] = []
            exit_code = 0 if report['passed'] else 1
    except (Exception, KeyboardInterrupt) as exc:
        report.update(passed=False, reason=f'{type(exc).__name__}: {exc}')
        report['error_traceback'] = traceback.format_exc()
        if observer is not None:
            report['recording'] = observer.recording
    finally:
        remaining = _stop_owned_process(process)
        report['remaining_owned_pids'] = remaining
        if remaining:
            report.update(passed=False, reason='owned_process_cleanup_incomplete')
            exit_code = 1
        if observer is not None:
            observer.close()
        (output / 'result.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'reason': report['reason'],
                      'result': str(output / 'result.json')}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
