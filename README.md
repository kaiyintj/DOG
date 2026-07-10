# DOG Semantic Mapping Workspace

ROS 2 workspace snapshot for Gazebo validation and later Lite3 laser-version robot deployment.

## Main Packages

- `src/semantic_mapping`: CLIP semantic mapping, GA-BSVM voxel map, language query, Nav2 remap, active perception.
- `src/fast_lio`: FAST-LIO front end with simulation and bag configuration changes.
- `src/unitree-go2-ros2`: Gazebo quadruped simulation and terrain world.
- `src/livox_laser_simulation_RO2`: Livox/Mid360 Gazebo plugin and scan patterns.
- `src/livox_ros_driver2`: Livox message definitions and driver package.

## Gazebo Test Flow

Open one terminal per command and source the workspace first:

```bash
cd ~/ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

```bash
ros2 launch go2_config gazebo.launch.py \
  world:=~/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/outdoor_car_benchmark.world \
  gui:=true rviz:=false use_sim_time:=true
```

```bash
ros2 launch fast_lio mapping.launch.py \
  config_file:=sim_mid360.yaml \
  use_sim_time:=true \
  rviz:=true
```

```bash
ros2 run semantic_mapping clip_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p use_sim_time:=true
```

```bash
ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p use_sim_time:=true
```

```bash
ros2 launch semantic_mapping nav_with_remap.launch.py use_sim_time:=True
```

```bash
ros2 run semantic_mapping active_perception_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p use_sim_time:=true
```

## Language Query

Listen for both the detected object and the safe approach pose before querying:

```bash
ros2 topic echo /query_target_pose
ros2 topic echo /goal_pose --once
```

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'car'}"
```

Check navigation output:

```bash
ros2 topic echo /plan --once
ros2 topic echo /cmd_vel_nav --once
ros2 topic echo /cmd_vel --once
ros2 topic echo /cmd_vel_champ --once
```

The Gazebo command chain has one publisher at every stage:

```text
Nav2 -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel
     -> active_perception -> /cmd_vel_champ -> CHAMP
```

The Go2 controller stops automatically when final velocity commands are stale
for 0.5 seconds. The Lite3 hardware preset keeps a separate robot command
interface; real camera calibration, LiDAR-to-camera extrinsics, footprint,
velocity limits and the Deep Robotics SDK bridge must be verified on hardware.
