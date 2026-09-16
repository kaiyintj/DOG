import ast
from pathlib import Path

import yaml

from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path):
    with (PROJECT_ROOT / relative_path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def test_semantic_profiles_share_electric_bicycle_schema():
    expected_vocab = list(DEFAULT_CLASSES)
    for relative_path in (
        'config/semantic_mapping_m2dgr.yaml',
        'config/semantic_mapping_sim_livox.yaml',
        'config/semantic_mapping_lite3_real.yaml',
    ):
        params = load_yaml(relative_path)
        clip = params['clip_node']['ros__parameters']
        segformer = params['segformer_node']['ros__parameters']
        mapping = params['ga_bsvm_node']['ros__parameters']

        assert clip['vocab'] == expected_vocab
        assert mapping['vocab'] == expected_vocab
        assert mapping['num_classes'] == len(expected_vocab)
        assert len(mapping['semantic_costs']) == len(expected_vocab)
        assert len(mapping['query_class_max_extent_m']) == len(expected_vocab)
        assert segformer[
            'require_electric_bicycle_training_acceptance'] is True
        assert segformer['electric_bicycle_min_target_iou'] >= 0.30
        assert segformer['electric_bicycle_min_iou'] >= 0.35
        assert segformer['electric_bicycle_min_road_iou'] >= 0.50


def test_semantic_profiles_use_headered_full_project_posterior():
    for relative_path in (
        'config/semantic_mapping_m2dgr.yaml',
        'config/semantic_mapping_sim_livox.yaml',
        'config/semantic_mapping_lite3_real.yaml',
    ):
        params = load_yaml(relative_path)
        segformer = params['segformer_node']['ros__parameters']
        mapping = params['ga_bsvm_node']['ros__parameters']

        assert segformer['posterior_temperature'] == 1.0
        assert segformer['project_posterior_topic'] == (
            '/segformer/project_posterior')
        assert mapping['segformer_use_full_posterior'] is True
        assert mapping['segformer_project_posterior_topic'] == (
            segformer['project_posterior_topic'])


def test_semantic_profiles_bound_costmap_height_and_prune_stale_voxels():
    for relative_path in (
        'config/semantic_mapping_m2dgr.yaml',
        'config/semantic_mapping_sim_livox.yaml',
        'config/semantic_mapping_lite3_real.yaml',
    ):
        params = load_yaml(relative_path)
        mapping = params['ga_bsvm_node']['ros__parameters']

        assert mapping['costmap_min_height_m'] < 0.0
        assert mapping['costmap_max_height_m'] > 0.0
        assert (
            mapping['costmap_min_height_m']
            < mapping['costmap_max_height_m']
        )
        assert 0.0 < mapping['dynamic_voxel_ttl_sec'] \
            < mapping['voxel_ttl_sec']
        assert mapping['voxel_prune_period_sec'] > 0.0
        assert mapping['voxel_prune_radius_m'] > 0.0
        assert mapping['voxel_max_count'] > 0
        assert mapping['evidence_decay_reference_sec'] > 0.0
        assert mapping['max_observation_weight'] >= 0.0
        assert mapping['legacy_map_publish_enabled'] is False
        assert mapping['map_topic'] != '/map'


def test_active_perception_profiles_derive_entropy_and_limit_scale_by_time():
    expected_imu_topics = {
        'config/semantic_mapping_m2dgr.yaml': '/handsfree/imu',
        'config/semantic_mapping_sim_livox.yaml': '/imu/data',
        'config/semantic_mapping_lite3_real.yaml': '/timefix/imu',
    }
    for relative_path in (
        'config/semantic_mapping_m2dgr.yaml',
        'config/semantic_mapping_sim_livox.yaml',
        'config/semantic_mapping_lite3_real.yaml',
    ):
        params = load_yaml(relative_path)
        active = params['active_perception_node']['ros__parameters']

        assert active['num_classes'] == len(DEFAULT_CLASSES)
        assert active['h_max'] == 0.0
        assert active['max_scale_rate_per_sec'] > 0.0
        assert active['legacy_cmd_rate_hz'] > 0.0
        assert 0.0 <= active['uncertainty_percentile'] <= 100.0
        assert 0.0 <= active['uncertainty_percentile_weight'] <= 1.0
        assert active['plan_max_age_sec'] > 0.0
        assert active['entropy_max_age_sec'] > 0.0
        assert active['imu_topic'] == expected_imu_topics[relative_path]
        assert active['imu_max_age_sec'] > 0.0
        assert active['require_fresh_imu'] is True
        assert active['cmd_vel_timeout_sec'] <= 0.3
        assert (
            active['watchdog_period_sec']
            < active['cmd_vel_timeout_sec']
        )


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


def test_simulation_goal_checker_ignores_drifting_yaw():
    params = load_yaml('config/nav2_sim_params.yaml')
    checker = params['controller_server']['ros__parameters']['general_goal_checker']
    assert checker['plugin'] == 'nav2_controller::PositionGoalChecker'
    assert checker['xy_goal_tolerance'] <= 0.30
    assert 'yaw_goal_tolerance' not in checker


def test_simulation_requires_verified_robot_side_approach():
    params = load_yaml('config/semantic_mapping_sim_livox.yaml')
    clip = params['clip_node']['ros__parameters']
    segformer = params['segformer_node']['ros__parameters']
    mapping = params['ga_bsvm_node']['ros__parameters']
    active = params['active_perception_node']['ros__parameters']

    # gazebo_ros_camera applies camera_name when producing the final runtime
    # topic. Match the live publisher instead of an inner xacro remapping tag.
    assert segformer['image_topic'] == '/d435i/image_raw'
    assert clip['image_topic'] == segformer['image_topic']
    assert mapping['image_topic'] == segformer['image_topic']
    assert mapping['pointcloud_topic'] == '/livox/lidar'

    assert mapping['projection_calibration_verified'] is True
    assert mapping['query_require_safe_approach'] is True
    assert mapping['query_require_robot_side'] is True
    assert mapping['query_require_robot_pose_for_approach'] is True
    assert 'query_path_clearance_radius_m' not in mapping
    assert mapping['tf_retry_queue_size'] >= 5
    assert mapping['tf_retry_max_age_sec'] >= 2.5
    assert active['preserve_command_curvature'] is True
    assert active['nav_cmd_vel_topic'] == '/cmd_vel'
    assert active['cmd_vel_topic'] == '/cmd_vel_champ'


def test_lite3_profile_fails_closed_until_calibration_and_bridge_are_ready():
    params = load_yaml('config/semantic_mapping_lite3_real.yaml')
    mapping = params['ga_bsvm_node']['ros__parameters']
    active = params['active_perception_node']['ros__parameters']

    assert mapping['semantic_backend'] == 'segformer'
    assert mapping['use_camera_info'] is True
    assert mapping['require_camera_info'] is True
    assert mapping['require_zero_distortion'] is True
    assert mapping['projection_calibration_verified'] is False
    assert mapping['lidar_to_camera_translation'] == [
        0.03541400632660336,
        0.41457697624860806,
        -0.1046040335091429,
    ]
    assert mapping['lidar_to_camera_quaternion'] == [
        0.585917353743501,
        -0.5934590870789174,
        0.388658516840907,
        0.39172914600868997,
    ]
    assert mapping['query_require_safe_approach'] is True
    assert mapping['query_require_robot_side'] is True
    assert mapping['query_require_robot_pose_for_approach'] is True
    assert mapping['query_approach_min_distance_m'] >= 1.2
    assert 'query_path_clearance_radius_m' not in mapping
    assert mapping['imu_acceleration_scale'] == 9.80665
    assert active['nav_cmd_vel_topic'] == '/cmd_vel'
    assert active['cmd_vel_topic'] == '/cmd_vel_lite3_safe'


def test_non_lite3_profiles_keep_si_imu_acceleration_scale():
    for relative_path in (
        'config/semantic_mapping_m2dgr.yaml',
        'config/semantic_mapping_sim_livox.yaml',
    ):
        params = load_yaml(relative_path)
        mapping = params['ga_bsvm_node']['ros__parameters']

        assert mapping['imu_acceleration_scale'] == 1.0


def test_lite3_moving_fast_lio_profile_publishes_nav2_body_cloud():
    params = load_yaml('config/fast_lio_lite3_real.yaml')['/**'][
        'ros__parameters']

    assert params['common']['lid_topic'] == '/timefix/lidar'
    assert params['common']['imu_topic'] == '/timefix/imu'
    assert params['common']['time_sync_en'] is False
    assert params['mapping']['extrinsic_est_en'] is True
    assert params['publish']['map_en'] is True
    assert params['publish']['scan_bodyframe_pub_en'] is True
    assert params['pcd_save']['pcd_save_en'] is False


def test_nav2_declares_rewriteable_odom_topic_for_lite3_fast_lio():
    params = load_yaml('config/nav2_params.yaml')

    assert params['bt_navigator']['ros__parameters']['odom_topic'] == '/odom'
    assert (
        params['controller_server']['ros__parameters']['odom_topic']
        == '/odom'
    )
    assert (
        params['velocity_smoother']['ros__parameters']['odom_topic']
        == 'odom'
    )

    launch_source = (
        PROJECT_ROOT / 'launch' / 'nav_with_remap.launch.py'
    ).read_text(encoding='utf-8')
    assert "'odom_topic'" in launch_source
    assert "param_rewrites={'odom_topic': odom_topic}" in launch_source


def test_nav_launch_starts_one_profile_selected_active_perception_gate():
    launch_source = (
        PROJECT_ROOT / 'launch' / 'nav_with_remap.launch.py'
    ).read_text(encoding='utf-8')

    assert "'active_perception_enabled'" in launch_source
    assert "default_value='true'" in launch_source
    assert "'active_perception_params_file'" in launch_source
    assert "'semantic_mapping_lite3_real.yaml'" in launch_source
    assert "executable='active_perception_node'" in launch_source
    assert 'condition=IfCondition(active_perception_enabled)' in launch_source
    assert 'active_perception_params_file,' in launch_source


def test_nav_launch_keeps_custom_bridge_as_only_goal_action_writer():
    launch_source = (
        PROJECT_ROOT / 'launch' / 'nav_with_remap.launch.py'
    ).read_text(encoding='utf-8')

    assert 'GroupAction([' in launch_source
    assert "SetRemap(src='/goal_pose', dst='/nav2_internal_goal_pose')" in launch_source
    assert "executable='nav_goal_bridge_node'" in launch_source


def test_nav_launch_defaults_match_lite3_wall_time_and_split_profiles():
    core_source = (
        PROJECT_ROOT / 'launch' / 'nav_with_remap.launch.py'
    ).read_text(encoding='utf-8')
    sim_source = (
        PROJECT_ROOT / 'launch' / 'nav_sim.launch.py'
    ).read_text(encoding='utf-8')
    real_source = (
        PROJECT_ROOT / 'launch' / 'nav_lite3_real.launch.py'
    ).read_text(encoding='utf-8')

    assert "'use_sim_time',\n            default_value='False'" in core_source
    assert "'use_sim_time',\n            default_value='true'" in sim_source
    assert 'semantic_mapping_sim_livox.yaml' in sim_source
    assert "'use_sim_time',\n            default_value='false'" in real_source
    assert 'semantic_mapping_lite3_real.yaml' in real_source
    assert "default_value='/Odometry'" in real_source
    assert "default_value='false'" in real_source


def test_simulation_nav_profile_keeps_local_obstacles_and_semantic_global_standoff():
    params = load_yaml('config/nav2_sim_params.yaml')
    controller = params['controller_server']['ros__parameters']
    local = params['local_costmap']['local_costmap']['ros__parameters']
    global_map = params['global_costmap']['global_costmap']['ros__parameters']
    sim_source = (
        PROJECT_ROOT / 'launch' / 'nav_sim.launch.py'
    ).read_text(encoding='utf-8')

    assert "'nav2_sim_params.yaml'" in sim_source
    assert controller['FollowPath']['use_collision_detection'] is False
    assert local['plugins'] == ['voxel_layer', 'inflation_layer']
    assert global_map['plugins'] == ['semantic_layer', 'inflation_layer']
