import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'semantic_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'benchmark'), glob('benchmark/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        # Exact compatible versions live in requirements-runtime-common.txt.
        # Keep ROS/colcon metadata non-pinning so an offline build never tries
        # to replace the platform Python environment implicitly.
        'numpy',
        'scipy',
        'Pillow',
        'PyYAML',
        'psutil',
    ],
    zip_safe=True,
    maintainer='yk',
    maintainer_email='yk@todo.todo',
    description='Semantic mapping and active perception nodes for ROS 2 navigation.',
    license='Apache-2.0',
    extras_require={
        'clip': [
            'open_clip_torch==3.3.0',
            'torch>=2.0',
        ],
        'segformer': [
            'huggingface-hub==0.36.0',
            'Pillow>=9,<13',
            'tokenizers==0.20.3',
            'torch>=2.0',
            'transformers==4.46.3',
        ],
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'planar_odom_velocity = semantic_mapping.runtime.planar_odom_velocity:main',
            'clip_node = semantic_mapping.runtime.clip_node:main',
            'segformer_node = semantic_mapping.runtime.segformer_node:main',
            'segformer_image = semantic_mapping.runtime.segformer_image:main',
            'sim_sensor_gate = semantic_mapping.runtime.simulation_startup:main',
            'run_indoor_semantic_benchmark = semantic_mapping.gazebo.indoor_benchmark:main',
            'segformer_dataset = '
            'semantic_mapping.runtime.segformer_training:dataset_main',
            'segformer_finetune = '
            'semantic_mapping.runtime.segformer_training:finetune_main',
            'segformer_checkpoint = '
            'semantic_mapping.runtime.segformer_training:checkpoint_main',
            'ga_bsvm_node = semantic_mapping.runtime.ga_bsvm_node:main',
            'nav_goal_bridge_node = '
            'semantic_mapping.runtime.nav_goal_bridge_node:main',
            'active_perception_node = '
            'semantic_mapping.runtime.active_perception_node:main',
            'lite3_odom_bridge_node = '
            'semantic_mapping.runtime.lite3_odom_bridge_node:main',
            'clip_query = semantic_mapping.runtime.clip_query:main',
            'semantic_query = semantic_mapping.runtime.clip_query:main',
            'carla_capture_benchmark = '
            'semantic_mapping.carla.carla_capture_benchmark:main',
            'carla_evaluate_benchmark = '
            'semantic_mapping.carla.carla_evaluate_benchmark:main',
            'carla_capture_reliability = '
            'semantic_mapping.carla.carla_capture_reliability:main',
            'carla_evaluate_reliability = '
            'semantic_mapping.carla.carla_evaluate_reliability:main',
        ],
    },
)
