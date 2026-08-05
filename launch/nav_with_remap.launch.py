import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    goal_bridge_enabled = LaunchConfiguration('goal_bridge_enabled')
    goal_pose_topic = LaunchConfiguration('goal_pose_topic')
    navigate_to_pose_action = LaunchConfiguration('navigate_to_pose_action')

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
        DeclareLaunchArgument(
            'goal_bridge_enabled',
            default_value='true',
            description='Forward /goal_pose messages to the Nav2 action server.',
        ),
        DeclareLaunchArgument(
            'goal_pose_topic',
            default_value='/goal_pose',
            description='PoseStamped semantic goal topic.',
        ),
        DeclareLaunchArgument(
            'navigate_to_pose_action',
            default_value='/navigate_to_pose',
            description='Nav2 NavigateToPose action name.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_file),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file,
            }.items(),
        ),
        Node(
            package='semantic_mapping',
            executable='nav_goal_bridge_node',
            name='nav_goal_bridge_node',
            output='screen',
            condition=IfCondition(goal_bridge_enabled),
            parameters=[{
                'use_sim_time': use_sim_time,
                'goal_pose_topic': goal_pose_topic,
                'navigate_to_pose_action': navigate_to_pose_action,
            }],
        ),
    ])
