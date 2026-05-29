import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import SetRemap
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 找到系统里原本的 nav2_bringup 路径
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')

    # 定义我们要加载的参数文件路径
    my_params_file = '/home/yk/ws/src/semantic_mapping/config/nav2_params.yaml'

    return LaunchDescription([
        # 🌟 魔法指令：强制全局拦截！把这个 launch 文件里启动的所有 /cmd_vel 全部重命名为 /cmd_vel_nav
        SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),

        # 拉起原本的 Nav2
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_file),
            launch_arguments={
                'use_sim_time': 'True',
                'params_file': my_params_file
            }.items()
        )
    ])