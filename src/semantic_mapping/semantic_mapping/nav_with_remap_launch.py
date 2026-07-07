import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    semantic_mapping_dir = get_package_share_directory('semantic_mapping')

    nav2_launch_file = os.path.join(
        nav2_bringup_dir,
        'launch',
        'navigation_launch.py',
    )
    default_params_file = os.path.join(
        semantic_mapping_dir,
        'config',
        'nav2_params.yaml',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock if true.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Full path to the Nav2 parameters file.',
        ),
        SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_file),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file,
            }.items(),
        ),
    ])
