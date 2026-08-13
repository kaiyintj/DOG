import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = get_package_share_directory('semantic_mapping')
    core_launch = os.path.join(
        package_share, 'launch', 'nav_with_remap.launch.py')
    default_nav2_params = os.path.join(
        package_share, 'config', 'nav2_params.yaml')
    simulation_profile = os.path.join(
        package_share, 'config', 'semantic_mapping_sim_livox.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    odom_topic = LaunchConfiguration('odom_topic')
    active_perception_enabled = LaunchConfiguration(
        'active_perception_enabled')
    active_perception_params_file = LaunchConfiguration(
        'active_perception_params_file')
    goal_bridge_enabled = LaunchConfiguration('goal_bridge_enabled')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_nav2_params,
            description='Full path to the Nav2 parameters file.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo /clock by default.'),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Simulation odometry topic consumed by Nav2.'),
        DeclareLaunchArgument(
            'active_perception_enabled',
            default_value='true',
            description='Start the fail-closed velocity safety gate.'),
        DeclareLaunchArgument(
            'active_perception_params_file',
            default_value=simulation_profile,
            description='Simulation active-perception parameter profile.'),
        DeclareLaunchArgument(
            'goal_bridge_enabled',
            default_value='true',
            description='Forward semantic goals to Nav2 in simulation.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(core_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file,
                'odom_topic': odom_topic,
                'active_perception_enabled': active_perception_enabled,
                'active_perception_params_file': (
                    active_perception_params_file),
                'goal_bridge_enabled': goal_bridge_enabled,
            }.items()),
    ])
