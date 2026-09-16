"""Indoor simulation configuration preserves the platform safety chain."""

from pathlib import Path
import runpy

import pytest
import yaml

from semantic_mapping.runtime.semantic_profile_ros import load_semantic_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_indoor_preset_selects_one_profile_without_repeating_class_metadata():
    with (PROJECT_ROOT / 'config/semantic_mapping_sim_indoor.yaml').open() as stream:
        preset = yaml.safe_load(stream)
    assert 'clip_node' not in preset
    for name in ('segformer_node', 'ga_bsvm_node', 'active_perception_node'):
        parameters = preset[name]['ros__parameters']
        assert parameters['ontology_profile'] == 'indoor7'
        assert load_semantic_contract(parameters).profile.K == 8
        assert not {'num_classes', 'vocab', 'semantic_costs'}.intersection(parameters)
    frontend = preset['segformer_node']['ros__parameters']
    mapper = preset['ga_bsvm_node']['ros__parameters']
    active = preset['active_perception_node']['ros__parameters']
    assert frontend['model_id'] == 'nvidia/segformer-b0-finetuned-ade-512-512'
    assert mapper['semantic_backend'] == 'segformer'
    assert mapper['segformer_use_full_posterior'] is True
    assert mapper['query_require_safe_approach'] is True
    # Keep same-side selection as a preference in Gazebo.  A strict side
    # requirement can reject every observed approach voxel when the object is
    # seen from behind, preventing the navigation chain from starting.
    assert mapper['query_prefer_robot_side'] is True
    assert mapper['query_require_robot_side'] is False
    assert mapper['query_face_target'] is False
    assert 'query_require_direct_path' not in mapper
    assert mapper['query_require_robot_pose_for_approach'] is True
    assert active['nav_cmd_vel_topic'] == '/cmd_vel'
    assert active['cmd_vel_topic'] == '/cmd_vel_champ'
    assert active['cmd_vel_timeout_sec'] == 0.25
    with (PROJECT_ROOT / 'config/semantic_mapping_sim_livox.yaml').open() as stream:
        outdoor = yaml.safe_load(stream)['ga_bsvm_node']['ros__parameters']
    for key in (
        'camera_k', 'lidar_to_camera_translation', 'lidar_to_camera_quaternion',
        'pointcloud_topic', 'pointcloud_frame', 'imu_topic',
    ):
        assert mapper[key] == outdoor[key]


def test_simulation_global_costmap_does_not_promote_transient_cloud_to_blocking_layer():
    with (PROJECT_ROOT / 'config/nav2_sim_params.yaml').open() as stream:
        params = yaml.safe_load(stream)
    global_map = params['global_costmap']['global_costmap']['ros__parameters']
    assert global_map['plugins'] == ['semantic_layer', 'inflation_layer']


def test_simulation_goal_checker_terminates_on_standoff_position():
    with (PROJECT_ROOT / 'config/nav2_sim_params.yaml').open() as stream:
        params = yaml.safe_load(stream)
    controller = params['controller_server']['ros__parameters']
    checker = controller['general_goal_checker']
    assert checker['plugin'] == 'nav2_controller::PositionGoalChecker'
    assert checker['xy_goal_tolerance'] == 0.25
    assert 'yaw_goal_tolerance' not in checker


@pytest.mark.parametrize('full_posterior', ['true', 'false'])
def test_simulation_launch_composes_existing_chain_and_rejects_unknown_profile(
    monkeypatch, tmp_path, full_posterior,
):
    import ament_index_python.packages
    from launch import LaunchContext
    from launch.actions import (
        DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction,
        RegisterEventHandler,
    )
    from launch.events import Shutdown
    from launch.events.process import ProcessExited
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'logs'))
    # Substitute only the external package index/filesystem, not launch logic.
    monkeypatch.setattr(
        ament_index_python.packages, 'get_package_share_directory',
        lambda name: str(PROJECT_ROOT if name == 'semantic_mapping' else tmp_path / name))
    world = tmp_path / 'go2_config/worlds/aws_robomaker/small_house/worlds/small_house.world'
    world.parent.mkdir(parents=True)
    world.write_text('<sdf version="1.6"><world name="test"/></sdf>')
    entry = runpy.run_path(str(PROJECT_ROOT / 'launch/semantic_sim.launch.py'))
    description = entry['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations['ontology_profile'] = 'indoor7'
    context.launch_configurations['seed'] = '7'
    context.launch_configurations['segformer_use_full_posterior'] = full_posterior
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    composition = next(
        action for action in description.entities if isinstance(action, OpaqueFunction))
    actions = composition.execute(context)
    nodes = [action for action in actions if isinstance(action, Node)]
    assert [node.node_executable for node in nodes] == [
        'sim_sensor_gate', 'segformer_node', 'ga_bsvm_node',
    ]
    fusion_parameters = evaluate_parameters(context, nodes[-1]._Node__parameters)
    assert fusion_parameters[-1]['segformer_use_full_posterior'] is (full_posterior == 'true')
    included = [a for a in actions if isinstance(a, IncludeLaunchDescription)]
    assert len(included) == 2  # FAST-LIO must not start before the sensor gate exits zero.
    arguments = [dict(action.launch_arguments) for action in included]
    assert arguments[0]['cmd_vel_topic'] == '/cmd_vel_champ'
    assert arguments[0]['publish_odom_tf'] == 'false'
    assert arguments[0]['seed'].perform(context) == '7'
    assert arguments[1]['odom_topic'] == '/Odometry'
    assert arguments[1]['active_perception_params_file'].endswith(
        'semantic_mapping_sim_indoor.yaml')
    assert actions[-1].condition.evaluate(context) is True
    assert context.launch_configurations['ros_domain_id'] != '0'

    handler = next(a.event_handler for a in actions if isinstance(a, RegisterEventHandler))
    for returncode in (0, 1):
        event = ProcessExited(
            action=nodes[0], name='sim_sensor_gate', cmd=[], cwd=None, env=None,
            pid=123, returncode=returncode)
        assert handler.matches(event)
        followup = list(handler.handle(event, context))
        if returncode == 0:
            assert len(followup) == 1
            assert isinstance(followup[0], IncludeLaunchDescription)
            fastlio_args = dict(followup[0].launch_arguments)
            assert fastlio_args['config_path'] == str(PROJECT_ROOT / 'config')
            assert fastlio_args['config_file'] == 'fast_lio_sim_pointcloud2.yaml'
            assert dict(followup[0].launch_arguments)['rviz'] == 'true'
        else:
            assert len(followup) == 1
            assert isinstance(followup[0], EmitEvent)
            assert isinstance(followup[0].event, Shutdown)

    context.launch_configurations['ontology_profile'] = 'mistyped_profile'
    with pytest.raises(ValueError, match='profile'):
        composition.execute(context)

    context.launch_configurations['ontology_profile'] = 'indoor7'
    context.launch_configurations['world'] = 'school_parking_lot'
    with pytest.raises(ValueError, match='requires ontology_profile=outdoor13'):
        composition.execute(context)


def test_simulation_launch_manual_mode_omits_nav2_and_command_writer(monkeypatch, tmp_path):
    import ament_index_python.packages
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
    from launch_ros.actions import Node

    root = Path(__file__).resolve().parents[1]
    go2_root = tmp_path / 'go2_config'
    world = go2_root / 'worlds/aws_robomaker/small_house/worlds/small_house.world'
    world.parent.mkdir(parents=True)
    world.write_text('<sdf version="1.6"><world name="test"/></sdf>')
    monkeypatch.setattr(
        ament_index_python.packages, 'get_package_share_directory',
        lambda name: str(root if name == 'semantic_mapping' else go2_root),
    )
    entry = runpy.run_path(str(root / 'launch/semantic_sim.launch.py'))
    description = entry['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations.update({
        'ontology_profile': 'indoor7', 'navigation_enabled': 'false', 'seed': '7',
    })
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    composition = next(
        action for action in description.entities if isinstance(action, OpaqueFunction))
    actions = composition.execute(context)
    nodes = [action for action in actions if isinstance(action, Node)]
    assert [node.node_executable for node in nodes] == [
        'sim_sensor_gate', 'segformer_node', 'ga_bsvm_node',
    ]
    included = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
    assert len(included) == 2
    nav_include = included[-1]
    assert nav_include.condition.evaluate(context) is False


def test_delayed_fastlio_rviz_selection_survives_go2_rviz_override(
    monkeypatch, tmp_path,
):
    """A fixed Go2 RViz setting must not shadow FAST-LIO's RViz choice."""
    import ament_index_python.packages
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
    from launch.events.process import ProcessExited
    from launch_ros.actions import Node

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'logs'))
    go2_root = tmp_path / 'go2_config'
    world = go2_root / 'worlds/aws_robomaker/small_house/worlds/small_house.world'
    world.parent.mkdir(parents=True)
    world.write_text('<sdf version="1.6"><world name="test"/></sdf>')
    monkeypatch.setattr(
        ament_index_python.packages, 'get_package_share_directory',
        lambda name: str(root if name == 'semantic_mapping' else go2_root),
    )
    entry = runpy.run_path(str(root / 'launch/semantic_sim.launch.py'))
    description = entry['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations.update({
        'ontology_profile': 'indoor7', 'navigation_enabled': 'false',
        'rviz': 'true', 'seed': '7',
    })
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    composition = next(
        action for action in description.entities if isinstance(action, OpaqueFunction))
    actions = composition.execute(context)
    included = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
    assert len(included) == 2
    nodes = [action for action in actions if isinstance(action, Node)]
    gate = next(node for node in nodes if node.node_executable == 'sim_sensor_gate')
    handler = next(action.event_handler for action in actions
                   if hasattr(action, 'event_handler'))

    # IncludeLaunchDescription for Go2 uses its own ``rviz:=false`` argument.
    # It shares the parent context, so emulate that write before the delayed
    # FAST-LIO include is materialised.
    context.launch_configurations['rviz'] = 'false'
    event = ProcessExited(
        action=gate, name='sim_sensor_gate', cmd=[], cwd=None, env=None,
        pid=123, returncode=0,
    )
    followup = list(handler.handle(event, context))
    fastlio_args = dict(followup[0].launch_arguments)
    assert fastlio_args['rviz'] == 'true'


def test_small_house_manifest_targets_are_exact_sdf_model_origins():
    import xml.etree.ElementTree as ET

    with (PROJECT_ROOT / 'benchmark/small_house_manifest.yaml').open() as stream:
        manifest = yaml.safe_load(stream)
    worlds = PROJECT_ROOT.parent / 'unitree-go2-ros2/robots/configs/go2_config/worlds'
    world_path = worlds / manifest['world_file']
    if not world_path.is_file():
        pytest.skip('Small House assets are an external go2_config dependency')
    world = ET.parse(world_path).getroot().find('world')
    assert manifest['gt_reference'] == 'sdf_top_level_model_origin'
    assert manifest['gt_frame'] == 'world'
    assert manifest['ontology_profile'] == 'indoor7'
    assert len(manifest['targets']) == 11
    for target in manifest['targets']:
        model = world.find(f"model[@name='{target['gazebo_model']}']")
        assert model is not None
        pose = model.find('pose')
        assert pose.get('frame', '') == ''
        assert [float(value) for value in pose.text.split()] == target['pose']
        assert model.findtext('include/uri') == target['asset']
    by_name = {target['gazebo_model']: target for target in manifest['targets']}
    assert by_name['KitchenTable_01_001']['pose'][:3] == [6.55269, 0.951173, -0.000006]
    assert by_name['ChairA_01_001']['pose'][:3] == [7.11516, 0.209028, 0.024685]


@pytest.mark.parametrize('seed', ['', '7'])
def test_seed_reaches_the_actual_champ_gazebo_command(seed, monkeypatch, tmp_path):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
    from launch.utilities import perform_substitutions

    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'logs'))
    source = PROJECT_ROOT.parent / 'unitree-go2-ros2/champ/champ_gazebo/launch/gazebo.launch.py'
    if not source.is_file():
        pytest.skip('CHAMP Gazebo is an external simulation dependency')
    description = runpy.run_path(str(source))['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations['seed'] = seed
    actions = list(description.entities)
    for action in list(actions):
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
        elif isinstance(action, OpaqueFunction):
            actions.extend(action.execute(context))
    commands = [[perform_substitutions(context, part) for part in action.cmd]
                for action in actions if isinstance(action, ExecuteProcess)]
    gazebo = next(command for command in commands if command[0] == 'gzserver')
    assert [part for part in gazebo if part.startswith('--seed')] == (
        [f'--seed={seed}'] if seed else [])
