"""Realized motion acceptance and faithful fusion-input trace contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from semantic_mapping.gazebo.observation import BenchmarkObserver
from semantic_mapping.runtime.fusion_trace import FusionTraceWriter


def test_trace_preserves_inputs_source_identity_and_pruning_order(tmp_path):
    writer = FusionTraceWriter(tmp_path / 'trace', {'ontology_profile': 'indoor7'})
    points = np.array([[1.1, 2.2, 3.3]], dtype=np.float64)
    frame = {'reliability': np.array([.2]), 'logits': np.array([[.1, .9]]),
             'colors': np.array([[1, 2, 3]]), 'features': None,
             'source_id': ('camera', 10_000_000_001)}
    writer.observation(frame, points, 10_100_000_000, (0, 1), 10.2)
    writer.prune(12.0, [0, 0, .225])
    lines = (writer.directory / 'events.jsonl').read_text().splitlines()
    events = [json.loads(value) for value in lines]
    assert [event['event'] for event in events] == ['observation', 'prune']
    assert events[0]['source_id'] == ['camera', 10_000_000_001]
    assert events[0]['observation_ns'] == 10_100_000_000
    with np.load(writer.directory / events[0]['file'], allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['points'], points)
        np.testing.assert_array_equal(stored['reliability'], frame['reliability'])
        np.testing.assert_array_equal(stored['logits'], frame['logits'])
        assert 'features' not in stored.files


def test_projection_diagnostics_keep_atomic_source_rgb_and_posterior(tmp_path):
    writer = FusionTraceWriter(tmp_path / 'trace', {})
    diagnostics = {name: np.eye(4) for name in ('lidar_to_camera', 'source_to_map')}
    diagnostics.update(pixel_u=np.array([1.]), pixel_v=np.array([2.]),
                       camera_depth=np.array([3.]), camera_matrix=np.eye(3),
                       rgb=np.full((3, 4, 3), 120, dtype=np.uint8),
                       posterior=np.full((3, 4, 2), .5))
    frame = {'points': np.array([[0., 0, 3.]]), 'reliability': np.array([.2]),
             'logits': np.zeros((1, 2)), 'colors': None, 'features': None,
             'source_id': ('camera', 1), 'diagnostics': diagnostics}
    writer.observation(frame, frame['points'], 1, (1, 0), .1)
    writer.observation(frame, frame['points'], 2, (0, 0), .2)
    events = [json.loads(value) for value in (writer.directory / 'events.jsonl').read_text()
              .splitlines()]
    assert events[0]['source_rgb_file'] == events[1]['source_rgb_file']
    assert len(list(writer.directory.glob('source_*.png'))) == 1
    with np.load(writer.directory / events[0]['file']) as stored:
        np.testing.assert_array_equal(stored['pixel_u'], diagnostics['pixel_u'])
        np.testing.assert_array_equal(stored['sensor_points'], frame['points'])


def make_motion_observer():
    observer = SimpleNamespace(
        recording={'sim_time': 1.0}, case={'motion_revisit': {
            'distance_m': .4, 'turn_rad': 2.0, 'hold_sim_sec': 45.0}},
        latest_rgb=None, output_directory=None,
        motion_publisher=SimpleNamespace(publish=lambda value: None))
    observer._motion_checkpoint = lambda name, pair: BenchmarkObserver._motion_checkpoint(
        observer, name, pair)
    return observer


def motion_pose(x, yaw):
    return {'position': [x, 0, .225],
            'orientation': Rotation.from_euler('z', yaw).as_quat().tolist()}


@pytest.mark.parametrize('real_motion', [True, False])
def test_motion_needs_truth_displacement_and_turn_not_just_commands(real_motion):
    observer = make_motion_observer()
    states = [(1, 0, 0), (2, .41, 0), (5.1, .41, 0), (6, .41, 2.05),
              (51.1, .41, 2.05), (52, .41, 0), (57.1, .41, 0)]
    for stamp, x, yaw in states:
        observer.recording['sim_time'] = stamp
        pair = {'estimate': motion_pose(x, yaw),
                'truth': motion_pose(x if real_motion else 0, yaw if real_motion else 0)}
        observer._paired_pose = lambda: pair
        finished = BenchmarkObserver._motion_step(observer)
    assert finished
    assert observer.recording['motion_observation']['completed'] is real_motion
    checkpoints = observer.recording['motion_observation']['checkpoints']
    hold_start = next(value['sim_sec'] for value in checkpoints if value['name'] == 'away_hold')
    hold_end = next(value['sim_sec'] for value in checkpoints if value['name'] == 'turn_back')
    assert hold_end - hold_start >= 45


def test_motion_stall_terminates_without_claiming_realized_maneuver():
    observer = make_motion_observer()
    observer._paired_pose = lambda: {'truth': motion_pose(0, 0), 'estimate': motion_pose(0, 0)}
    assert BenchmarkObserver._motion_step(observer) is False
    observer.recording['sim_time'] = 47
    assert BenchmarkObserver._motion_step(observer) is True
    assert observer.recording['motion_observation']['completed'] is False
    assert observer.recording['motion_observation']['failed_stage'] == 'drive'


def test_motion_command_cadence_does_not_follow_fast_sensor_callbacks(monkeypatch):
    import semantic_mapping.gazebo.observation as module
    wall = [0.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: wall[0]))
    observer = make_motion_observer()
    observer.case['motion_revisit']['timeout_wall_sec'] = 1
    observer.recording['motion_observation'] = {'completed': True}
    observer._live_problem = lambda **kwargs: None
    observer._paired_pose = lambda: None
    samples = []

    def spin_once(**kwargs):
        wall[0] += .001  # Continuously queued camera/TF/command callbacks.

    def step():
        samples.append(wall[0])
        return len(samples) == 3

    observer.ros_executor = SimpleNamespace(spin_once=spin_once)
    observer._motion_step = step
    process = SimpleNamespace(poll=lambda: None)
    assert BenchmarkObserver._observe_motion(observer, process, np.eye(4))
    assert min(np.diff(samples)) >= .05 - 1e-9


def test_paired_replay_applies_recorded_pruning_to_the_same_input(tmp_path):
    from semantic_mapping.offline.replay_fusion_trace import replay
    root = Path(__file__).resolve().parents[1]
    parameters = yaml.safe_load((root / 'config/semantic_mapping_sim_indoor.yaml').read_text())[
        'ga_bsvm_node']['ros__parameters']
    writer = FusionTraceWriter(tmp_path / 'fusion_trace', parameters)
    points = np.repeat([[2.01, .01, .41], [2.11, .01, .41], [2.21, .01, .41]], 20, axis=0)
    logits = np.zeros(8)
    logits[3] = 6
    writer.observation({'reliability': np.ones(len(points)), 'logits': logits,
                        'features': None, 'colors': None, 'source_id': ('camera', 10_000_000_000)},
                       points, 10_000_000_000, (3, 0), 10.1)
    for stamp in (11, 50, 100):
        writer.prune(stamp, [0, 0, .225])
    estimate = {**motion_pose(0, 0), 'frame': 'odom', 'stamp': 11}
    truth = {**estimate, 'frame': 'world'}
    target = {'id': 'chair', 'class': 'chair', 'pose': [2.11, .01, .01, 0, 0, 0]}
    recording = {
        'passed': False, 'case': {'target_class': 'chair', 'targets': [target]},
        'recording': {'alignment': {'truth': truth, 'estimate': estimate},
                      'trajectory': {'estimate': [estimate]},
                      'motion_observation': {'completed': True, 'checkpoints': [
                          {'name': 'before_motion', 'sim_sec': 12},
                          {'name': 'turn_back', 'sim_sec': 60}]}}}
    (tmp_path / 'result.json').write_text(json.dumps(recording))
    result = replay(tmp_path)
    assert result['observations'] == 1
    initial, later = result['checkpoints']
    assert initial['variants']['original_decay']['queryable']
    assert not later['variants']['original_decay']['queryable']
    assert later['variants']['no_decay']['queryable']
    assert result['navigation_outcome_is_only_for_online_no_decay'] is False
