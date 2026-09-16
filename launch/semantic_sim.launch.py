"""Compose one explicit ontology profile on the existing Go2 simulation chain."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from semantic_mapping.runtime.semantic_profile import open_profile


def _compose_simulation(context):
    profile = open_profile(LaunchConfiguration('ontology_profile').perform(context))
    semantic_share = Path(get_package_share_directory('semantic_mapping'))
    go2_share = Path(get_package_share_directory('go2_config'))
    fast_lio_share = Path(get_package_share_directory('fast_lio'))
    indoor = profile.id == 'indoor7'
    preset = semantic_share / 'config' / (
        'semantic_mapping_sim_indoor.yaml' if indoor
        else 'semantic_mapping_sim_livox.yaml')
    house = go2_share / 'worlds/aws_robomaker/small_house'
    world_aliases = {
        'aws_small_house': house / 'worlds/small_house.world',
        'school_parking_lot': go2_share / 'worlds/school_parking_lot.world',
    }
    choice = LaunchConfiguration('world').perform(context)
    if not choice:
        choice = 'aws_small_house' if indoor else 'school_parking_lot'
    expected_profile = {
        'aws_small_house': 'indoor7',
        'school_parking_lot': 'outdoor13',
    }.get(choice)
    if expected_profile and expected_profile != profile.id:
        raise ValueError(
            f'World {choice} requires ontology_profile={expected_profile}, '
            f'not {profile.id}')
    world = world_aliases.get(choice, Path(choice).expanduser())
    if not world.is_file():
        raise FileNotFoundError(f'Gazebo world not installed: {world}')
    if not preset.is_file():
        raise FileNotFoundError(f'Simulation preset not installed: {preset}')
    model_paths = [str(house / 'models'), context.environment.get('GAZEBO_MODEL_PATH', '')]
    model_override = LaunchConfiguration('model_id').perform(context)
    frontend_overrides = {'ontology_profile': profile.id, 'use_sim_time': True}
    if model_override:
        frontend_overrides['model_id'] = model_override
    starts = {}
    for axis, default in (('x', '3.5' if indoor else '0.0'),
                          ('y', '1.0' if indoor else '0.0'),
                          ('z', '0.35'), ('heading', '0.0')):
        starts['world_init_' + axis] = (
            LaunchConfiguration('world_init_' + axis).perform(context) or default)
    # The Go2 child launch has its own ``rviz`` argument, which is kept false
    # to avoid a duplicate URDF RViz window.  Resolve the parent's FAST-LIO
    # choice before including that child so the delayed include below cannot
    # observe the child's same-name override.
    fast_lio_rviz = LaunchConfiguration('rviz').perform(context)
    # The Gazebo Livox plugin publishes a normal PointCloud2 alongside its
    # legacy Livox CustomMsg.  Use the generic FAST-LIO parser by default: the
    # CustomMsg path depends on synthetic per-point timing and can make the
    # simulated odometry diverge after a turn.  Keep both launch arguments so
    # an explicit FAST-LIO comparison (or a hardware-specific config) remains
    # possible without changing this launch file.
    fast_lio_config_path = LaunchConfiguration('fast_lio_config_path').perform(context)
    if not fast_lio_config_path:
        fast_lio_config_path = str(semantic_share / 'config')
    fast_lio_config_file = LaunchConfiguration('fast_lio_config_file').perform(context)
    if not fast_lio_config_file:
        fast_lio_config_file = 'fast_lio_sim_pointcloud2.yaml'

    sensor_gate = Node(
        package='semantic_mapping', executable='sim_sensor_gate',
        name='sim_sensor_gate', output='screen')
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(fast_lio_share / 'launch/mapping.launch.py')),
        launch_arguments={
            'config_path': fast_lio_config_path,
            'config_file': fast_lio_config_file,
            'use_sim_time': 'true',
            'rviz': fast_lio_rviz,
        }.items())

    def start_localization(event, _context):
        if event.returncode == 0:
            return [localization]
        return [EmitEvent(event=Shutdown(reason='Simulation sensors did not settle for FAST-LIO'))]

    return [
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('ros_domain_id')),
        SetEnvironmentVariable('GAZEBO_MASTER_URI', LaunchConfiguration('gazebo_master_uri')),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', os.pathsep.join(filter(None, model_paths))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(go2_share / 'launch/gazebo.launch.py')),
            launch_arguments={
                'world': str(world), 'use_sim_time': 'true',
                'seed': LaunchConfiguration('seed'),
                'cmd_vel_topic': '/cmd_vel_champ', 'publish_odom_tf': 'false',
                'gui': LaunchConfiguration('gui'), 'rviz': 'false', **starts,
            }.items()),
        RegisterEventHandler(OnProcessExit(
            target_action=sensor_gate, on_exit=start_localization)),
        sensor_gate,
        Node(
            package='semantic_mapping', executable='segformer_node',
            name='segformer_node', output='screen',
            parameters=[str(preset), frontend_overrides]),
        Node(
            package='semantic_mapping', executable='ga_bsvm_node',
            name='ga_bsvm_node', output='screen',
            parameters=[str(preset), {
                'ontology_profile': profile.id, 'use_sim_time': True,
                'semantic_backend': 'segformer',
                'segformer_use_full_posterior': (
                    LaunchConfiguration('segformer_use_full_posterior').perform(context)
                    == 'true'),
            }]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(semantic_share / 'launch/nav_sim.launch.py')),
            condition=IfCondition(LaunchConfiguration('navigation_enabled')),
            launch_arguments={
                'use_sim_time': 'true', 'odom_topic': '/Odometry',
                'active_perception_params_file': str(preset),
                'goal_bridge_enabled': 'true',
            }.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'ontology_profile', default_value='outdoor13',
            choices=['outdoor13', 'indoor7'],
            description='Fixed profile for this complete run; restart to change.'),
        DeclareLaunchArgument(
            'world', default_value='',
            description='aws_small_house, school_parking_lot, or an existing world path.'),
        DeclareLaunchArgument(
            'model_id', default_value='',
            description='Optional checkpoint override; otherwise use the profile preset.'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument(
            'segformer_use_full_posterior', default_value='true',
            choices=['true', 'false'],
            description='False selects legacy mask/confidence fusion for comparison.'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Start FAST-LIO RViz for the live map and odometry.'),
        DeclareLaunchArgument(
            'fast_lio_config_path',
            default_value='',
            description=(
                'Directory containing the FAST-LIO config. Empty uses the '
                'installed semantic_mapping/config Gazebo preset.'),
        ),
        DeclareLaunchArgument(
            'fast_lio_config_file', default_value='',
            description='FAST-LIO config filename within fast_lio_config_path.'),
        DeclareLaunchArgument(
            'seed', default_value='', description='Optional Gazebo random seed for benchmarks.'),
        DeclareLaunchArgument(
            'ros_domain_id', default_value='216',
            description='Isolated local simulation domain; use it for query/monitor terminals.'),
        DeclareLaunchArgument(
            'gazebo_master_uri', default_value='http://127.0.0.1:11357'),
        DeclareLaunchArgument(
            'navigation_enabled', default_value='true',
            description=(
                'Start Nav2, active perception and goal bridge. Set false '
                'for keyboard-only mapping/observation.'),
        ),
        DeclareLaunchArgument('world_init_x', default_value=''),
        DeclareLaunchArgument('world_init_y', default_value=''),
        DeclareLaunchArgument('world_init_z', default_value=''),
        DeclareLaunchArgument('world_init_heading', default_value=''),
        OpaqueFunction(function=_compose_simulation),
    ])
