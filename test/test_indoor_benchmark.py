"""Evaluate benchmark outcomes through the same recording interface as the runner."""

import copy
from collections import deque
import math
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from semantic_mapping.gazebo.indoor_benchmark import evaluate_case


def test_runner_rejects_unknown_case_before_launching_simulation(tmp_path):
    manifest = Path(__file__).resolve().parents[1] / 'benchmark/small_house_manifest.yaml'
    output = tmp_path / 'result'
    process = subprocess.run([
        sys.executable, '-m', 'semantic_mapping.gazebo.indoor_benchmark',
        '--manifest', str(manifest), '--case', 'does_not_exist', '--output', str(output),
    ], capture_output=True, text=True, timeout=10)
    assert process.returncode == 2, process.stdout + process.stderr
    report = json.loads((output / 'result.json').read_text())
    assert report['passed'] is False
    assert 'Unknown case' in report['reason']
    assert not (output / 'simulation.log').exists()


def test_manifest_contains_static_inputs_not_historical_outcomes():
    import yaml

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / 'benchmark/small_house_manifest.yaml').read_text())
    assert manifest['schema_version'] == 1
    assert manifest['scenario_id'] == 'aws_small_house'
    assert 'acceptance_status' not in manifest
    for start in manifest['robot_starts']:
        assert 'validated' not in start
        assert 'validation_evidence' not in start
    for case in manifest['cases']:
        assert 'acceptance_status' not in case
        assert 'validation_evidence' not in case


def test_runner_can_validate_a_candidate_start_without_querying(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / 'benchmark/small_house_manifest.yaml').read_text())
    manifest.update(warmup_sim_sec=1.0, startup_timeout_wall_sec=15.0)
    manifest['cases'].append({
        'id': 'validate_candidate',
        'start': manifest['robot_starts'][0]['id'],
        'expected': 'start_validation',
    })
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest))
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'id2label': {'0': 'floor'}}))
    monkeypatch.setenv('BENCHMARK_TEST_CHECKPOINT', str(checkpoint))
    monkeypatch.setenv('ROS_DOMAIN_ID', '230')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('GAZEBO_MASTER_URI', 'http://127.0.0.1:11359')
    output = tmp_path / 'run'
    process = subprocess.run([
        sys.executable, str(root / 'test/fixtures/indoor_benchmark_driver.py'),
        '--manifest', str(manifest_path), '--case', 'validate_candidate',
        '--output', str(output),
    ], capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / 'result.json').read_text())
    assert report['format'] == 'semantic_mapping_gazebo_run'
    assert report['format_version'] == 1
    assert report['scenario']['id'] == manifest['scenario_id']
    assert report['passed'] is True
    assert report['reason'] == 'start_validation_pass'
    assert report['acceptance_scope'] == 'start_validation'
    assert report['recording']['query']['sent'] is False
    assert report['recording']['target_poses'] == report['recording']['goal_poses'] == []


def test_runner_rejects_absent_control_when_asset_is_in_world(tmp_path):
    import yaml

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / 'benchmark/small_house_manifest.yaml').read_text())
    control = next(case for case in manifest['cases'] if case['id'] == 'bed_absent_control')
    control['world_file'] = manifest['world_file']
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest))
    output = tmp_path / 'result'
    process = subprocess.run([
        sys.executable, '-m', 'semantic_mapping.gazebo.indoor_benchmark',
        '--manifest', str(manifest_path), '--case', 'bed_absent_control',
        '--output', str(output),
    ], capture_output=True, text=True, timeout=10)
    assert process.returncode == 2, process.stdout + process.stderr
    report = json.loads((output / 'result.json').read_text())
    assert report['passed'] is False
    assert 'Expected world-absent asset is present' in report['reason']
    assert not (output / 'simulation.log').exists()


@pytest.mark.parametrize('parent_exits', [False, True])
def test_runner_executes_negative_case_and_stops_only_owned_simulator(
    tmp_path, monkeypatch, parent_exits,
):
    import yaml

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / 'benchmark/small_house_manifest.yaml').read_text())
    manifest.update(warmup_sim_sec=1.0, negative_observation_wall_sec=.3,
                    startup_timeout_wall_sec=15.0)
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest))
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'id2label': {'0': 'floor'}}))
    monkeypatch.setenv('BENCHMARK_TEST_CHECKPOINT', str(checkpoint))
    monkeypatch.setenv('ROS_DOMAIN_ID', '230')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('GAZEBO_MASTER_URI', 'http://127.0.0.1:11359')
    if parent_exits:
        monkeypatch.setenv('BENCHMARK_TEST_PARENT_EXIT', '1')
        monkeypatch.setenv('BENCHMARK_TEST_ORPHAN_PID_FILE', str(tmp_path / 'orphan_pid'))
    output = tmp_path / 'run'
    process = subprocess.run([
        sys.executable, str(root / 'test/fixtures/indoor_benchmark_driver.py'),
        '--manifest', str(manifest_path), '--case', 'floor_nonqueryable', '--output', str(output),
        '--fusion-input', 'hard_mask_confidence',
    ], capture_output=True, text=True, timeout=40)
    assert process.returncode == (1 if parent_exits else 0), process.stdout + process.stderr
    assert 'runner fixture: cleanup verified' in process.stdout, process.stdout + process.stderr
    report = json.loads((output / 'result.json').read_text())
    assert report['fusion_input'] == 'hard_mask_confidence'
    assert 'segformer_use_full_posterior:=false' in report['command']
    assert report['passed'] is (not parent_exits)
    if not parent_exits:
        assert report['recording']['query']['ack'] == 'nonqueryable'
    else:
        assert report['recording']['query']['sent'] is False
    assert report['recording']['target_poses'] == report['recording']['goal_poses'] == []
    assert report['remaining_owned_pids'] == []


def pose(position, yaw=0.0, frame='odom', stamp=12.0):
    return {'position': position, 'orientation': [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
            'frame': frame, 'stamp': stamp}


@pytest.fixture
def successful_recording():
    case = {
        'id': 'table', 'expected': 'navigation_success', 'target_class': 'table',
        'instance_policy': 'nearest_same_class_gt', 'target_origin_tolerance_xy_m': 1.5,
        'targets': [
            {'id': 'table_near', 'class': 'table', 'pose': [10, 22, .75, 0, 0, 0]},
            {'id': 'table_far', 'class': 'table', 'pose': [20, 22, 0, 0, 0, 0]},
        ],
    }
    recording = {
        'termination': 'nav_terminal', 'readiness': {'passed': True},
        'query': {'sent': True, 'ack': 'accepted', 'stamp': 11.0},
        'alignment': {'truth': pose([10, 20, 0], math.pi / 2, 'world', 10.0),
                      'estimate': pose([1, 2, 0], stamp=10.0)},
        'target_poses': [pose([3, 2, 0])],
        'goal_poses': [pose([2, 2, 0])],
        'bridge_events': [
            {'sequence': 1, 'event': 'ACCEPTED', 'nav2_goal_id': 'test-goal'},
            {'sequence': 1, 'event': 'SUCCEEDED', 'nav2_goal_id': 'test-goal'},
        ],
        'final_robot': {'truth': pose([10, 21, 0], math.pi / 2, 'world', 20.0),
                        'estimate': pose([2, 2, 0], stamp=20.0)},
    }
    return case, recording


def test_benchmark_aligns_frames_without_calling_model_origin_a_surface(successful_recording):
    case, recording = successful_recording
    result = evaluate_case(case, recording)
    assert result['passed'] is True
    assert result['reason'] == 'navigation_chain_smoke_pass'
    assert result['acceptance_scope'] == 'functional_chain_only'
    assert result['formal_safe_approach_passed'] is False
    metrics = result['metrics']
    assert metrics['nearest_gt_id'] == 'table_near'
    assert metrics['target_to_gt_model_origin_xy_m'] == pytest.approx(0)
    assert metrics['target_to_gt_model_origin_3d_m'] == pytest.approx(.75)
    assert metrics['goal_to_estimated_target_xy_m'] == pytest.approx(1)
    assert metrics['final_robot_to_goal_gt_xy_m'] == pytest.approx(0)
    for name in (
        'safe_approach_surface_distance', 'valid_goal_rate', 'collision_rate',
        'navigation_success_rate', 'spl',
    ):
        assert metrics[name] == 'not_evaluated'


def test_benchmark_reports_target_error_when_safe_goal_is_rejected(successful_recording):
    case, recording = successful_recording
    recording.update(termination='query_timeout', goal_poses=[], bridge_events=[])
    result = evaluate_case(case, recording)
    assert result['passed'] is False
    assert result['reason'] == 'safe_approach_not_published'
    assert result['metrics']['nearest_gt_id'] == 'table_near'
    assert result['metrics']['target_to_gt_model_origin_xy_m'] == pytest.approx(0)
    assert result['metrics']['target_to_gt_model_origin_3d_m'] == pytest.approx(.75)
    assert result['metrics']['goal_to_estimated_target_xy_m'] == 'not_evaluated'


def test_nav2_success_without_matching_accepted_goal_is_not_success(successful_recording):
    case, recording = successful_recording
    recording['bridge_events'][1]['nav2_goal_id'] = 'unrelated-goal'
    assert evaluate_case(case, recording)['passed'] is False


def test_negative_case_needs_full_live_observation_and_query_ack(successful_recording):
    _, recording = successful_recording
    case = {'id': 'unsupported', 'expected': 'no_target_or_goal',
            'query_status': 'unsupported', 'negative_observation_wall_sec': 15.0}
    recording.update(termination='observation_complete', target_poses=[], goal_poses=[],
                     bridge_events=[])
    recording['query'].update(ack='unsupported', observed_wall_sec=15.1, observed_sim_sec=3.0)
    assert evaluate_case(case, recording)['passed'] is True
    for changes in ({'ack': None}, {'observed_wall_sec': 14.0}, {'observed_sim_sec': 0.0}):
        broken = copy.deepcopy(recording)
        broken['query'].update(changes)
        assert evaluate_case(case, broken)['passed'] is False
    recording['goal_poses'] = [pose([2, 2, 0])]
    result = evaluate_case(case, recording)
    assert result['passed'] is False
    assert result['metrics']['false_goal_publication'] is True


def test_runtime_does_not_treat_delayed_active_service_poll_as_deactivation():
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    now = time.monotonic()
    observer = SimpleNamespace(
        recording={'sim_time': 10.0},
        clock_wall=now,
        last={name: (10.0, now) for name in (
            'rgb', 'imu', 'lidar', 'registered', 'posterior', 'semantic',
            'costmap', 'truth', 'estimate')},
        capability=object(),
        service_clients={'controllers': object()},
        service_status={'controllers': (True, now - 20.0)},
        navigation=SimpleNamespace(server_is_ready=lambda: True),
        query_publisher=SimpleNamespace(get_subscription_count=lambda: 1),
        _tf_authority_problem=lambda: None,
        get_publishers_info_by_topic=lambda _topic: [
            SimpleNamespace(node_name='active_perception_node')],
    )

    assert BenchmarkObserver._live_problem(observer) == 'controllers_not_active'
    assert BenchmarkObserver._live_problem(
        observer, require_fresh_service_status=False) is None


def test_empty_domain_check_ignores_ros2_cli_daemon(monkeypatch):
    import semantic_mapping.gazebo.observation as observation_module
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    clock = iter((0.0, 3.0))
    monkeypatch.setattr(observation_module.time, 'monotonic', lambda: next(clock))
    observer = SimpleNamespace(
        ros_executor=SimpleNamespace(spin_once=lambda timeout_sec: None),
        get_node_names_and_namespaces=lambda: [
            ('_ros2cli_daemon_228_deadbeef', '/'),
        ],
        get_name=lambda: 'indoor_benchmark_observer',
    )

    BenchmarkObserver.require_empty_domain(observer)


def test_observer_uses_newest_synchronized_history_pair():
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    observer = SimpleNamespace(
        poses={
            'truth': deque([pose([0, 0, 0], stamp=10.05),
                            pose([1, 0, 0], stamp=20.30)]),
            'estimate': deque([pose([0, 0, 0], stamp=10.00),
                               pose([1, 0, 0], stamp=20.00)]),
        },
    )

    pair = BenchmarkObserver._paired_pose(observer)

    assert pair['estimate']['stamp'] == pytest.approx(10.00)
    assert pair['truth']['stamp'] == pytest.approx(10.05)


def test_observer_returns_no_pair_until_history_is_synchronized():
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    observer = SimpleNamespace(
        poses={
            'truth': deque([pose([0, 0, 0], stamp=20.30)]),
            'estimate': deque([pose([1, 0, 0], stamp=20.00)]),
        },
    )

    assert BenchmarkObserver._paired_pose(observer) is None


def test_runtime_records_a_freshness_failure_as_history_not_current_state():
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    now = time.monotonic()
    streams = (
        'rgb', 'imu', 'lidar', 'registered', 'posterior', 'semantic',
        'costmap', 'truth', 'estimate',
    )
    observer = SimpleNamespace(
        recording={'sim_time': 10.0},
        clock_wall=now,
        last={name: (10.0, now) for name in streams},
        capability=object(), service_clients={}, service_status={},
        navigation=SimpleNamespace(server_is_ready=lambda: True),
        query_publisher=SimpleNamespace(get_subscription_count=lambda: 1),
        get_publishers_info_by_topic=lambda topic: [SimpleNamespace(
            node_name='active_perception_node')],
        _tf_authority_problem=lambda: None,
    )
    observer.last['rgb'] = (0.0, 0.0)

    assert BenchmarkObserver._live_problem(observer) == 'rgb_not_fresh'
    assert observer.recording['last_freshness_failure']['stream'] == 'rgb'
    assert 'freshness_failure' not in observer.recording
    observer.last['rgb'] = (10.0, now)
    assert BenchmarkObserver._live_problem(observer) is None


def test_observer_summarizes_navigation_command_chain_without_raw_samples():
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float32, String
    from semantic_mapping.gazebo.observation import BenchmarkObserver

    observer = SimpleNamespace(recording={'control': {}})
    moving = Twist()
    moving.linear.x = 0.2
    moving.angular.z = -0.4
    BenchmarkObserver._command(observer, 'nav_cmd', moving)
    BenchmarkObserver._command(observer, 'nav_cmd', Twist())
    BenchmarkObserver._speed_scale(observer, Float32(data=0.3))
    BenchmarkObserver._speed_scale(observer, Float32(data=0.8))
    BenchmarkObserver._perception_mode(observer, String(data='CAUTIOUS'))

    control = observer.recording['control']
    assert control['nav_cmd']['count'] == 2
    assert control['nav_cmd']['nonzero_count'] == 1
    assert control['nav_cmd']['max_linear_mps'] == pytest.approx(0.2)
    assert control['nav_cmd']['max_angular_rps'] == pytest.approx(0.4)
    assert control['speed_scale'] == {
        'count': 2, 'min': pytest.approx(0.3),
        'max': pytest.approx(0.8), 'last': pytest.approx(0.8),
    }
    assert control['perception_mode'] == {
        'count': 1, 'last': 'CAUTIOUS', 'counts': {'CAUTIOUS': 1},
    }
