import ast
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path):
    with (PROJECT_ROOT / relative_path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def test_nav2_uses_conservative_go2_footprint_and_inflation():
    params = load_yaml('config/nav2_params.yaml')
    local = params['local_costmap']['local_costmap']['ros__parameters']
    global_map = params['global_costmap']['global_costmap']['ros__parameters']

    local_footprint = ast.literal_eval(local['footprint'])
    global_footprint = ast.literal_eval(global_map['footprint'])

    assert local_footprint == global_footprint
    assert max(abs(point[0]) for point in local_footprint) >= 0.35
    assert max(abs(point[1]) for point in local_footprint) >= 0.22
    assert local['footprint_padding'] >= 0.05
    assert local['inflation_layer']['inflation_radius'] >= 0.85
    assert global_map['inflation_layer']['inflation_radius'] >= 0.85


def test_nav2_progress_checker_matches_deliberate_slowdown():
    params = load_yaml('config/nav2_params.yaml')
    controller = params['controller_server']['ros__parameters']
    progress = controller['progress_checker']

    assert progress['required_movement_radius'] <= 0.15
    assert progress['movement_time_allowance'] >= 20.0
    assert controller['FollowPath']['BaseObstacle.scale'] >= 0.2


def test_simulation_requires_verified_robot_side_approach():
    params = load_yaml('config/semantic_mapping_sim_livox.yaml')
    mapping = params['ga_bsvm_node']['ros__parameters']
    active = params['active_perception_node']['ros__parameters']

    assert mapping['query_require_safe_approach'] is True
    assert mapping['query_require_robot_side'] is False
    assert mapping['query_require_robot_pose_for_approach'] is True
    assert mapping['query_path_clearance_radius_m'] >= 0.65
    assert mapping['tf_retry_queue_size'] >= 5
    assert mapping['tf_retry_max_age_sec'] >= 2.5
    assert active['preserve_command_curvature'] is True
