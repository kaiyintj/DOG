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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yk',
    maintainer_email='yk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'clip_node = semantic_mapping.clip_node:main',
        	'ga_bsvm_node = semantic_mapping.ga_bsvm_node:main',
        	'active_perception_node = semantic_mapping.active_perception_node:main',
        ],
    },
)
