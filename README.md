# semantic_mapping

ROS 2 nodes for CLIP/SegFormer perception experiments, GA-BSVM voxel fusion,
semantic costmap publishing, and active perception speed modulation.

Current project truth and reproducible commands are maintained in:

- [Current version status](docs/PROJECT_STATUS.md)
- [Bag, Gazebo and Lite3 runbook](docs/RUNBOOK.md)
- [CARLA CLIP/SegFormer image benchmark](docs/CARLA_IMAGE_BENCHMARK.md)

The status document distinguishes implemented code from validated behavior and
known deployment gaps. Its `新对话交接摘要` section is the handoff point for a
new conversation. Use it instead of commands copied from older chats.

## Mapping and navigation method

The online map uses reliability-weighted categorical Dirichlet evidence. Each
camera/LiDAR observation is weighted by synchronized IMU motion, local point
density, sensor range, image-edge projection and semantic entropy. Points
from one frame are aggregated per voxel and contribute bounded evidence, which
avoids confidence being determined only by LiDAR sampling density.

With the CLIP backend, each voxel stores both a closed-set semantic posterior
and a normalized CLIP feature. With the SegFormer backend, the timestamped
pixel class and confidence update the same posterior without a CLIP feature.
Both backends also fuse the RGB color observed at each projected LiDAR point.
Language targets are selected from spatial clusters rather than a single
maximum voxel. CLIP supports open-vocabulary similarity; SegFormer supports the
configured closed-set navigation classes. Object-like goals must have a
sufficiently clear road voxel nearby before `/goal_pose` is published. The
approach selector prefers a verified road voxel on the robot-facing side of
the object and then prefers a direct segment clear of semantic obstacles. The
Gazebo and Lite3 presets reject the goal when robot pose or a same-side road
approach is unavailable instead of silently falling back to the object center.
`nav_goal_bridge_node` converts that topic into a Nav2 `NavigateToPose` action
request and reports whether the server accepted, rejected or completed it.

`active_perception_node` converts the fused semantic and epistemic uncertainty
along the local path into a filtered velocity scale. Linear and angular
velocity are scaled together by default so the executed arc remains the one
that DWB collision-checked.

In Gazebo the command chain is deliberately separated to keep a single writer
at each stage: Nav2 publishes `/cmd_vel_nav`, its velocity smoother publishes
`/cmd_vel`, active perception publishes `/cmd_vel_champ`, and CHAMP consumes
only `/cmd_vel_champ`.

`/query_target_pose` is the estimated object position. `/goal_pose` is the
collision-aware navigation approach point, so the two poses are intentionally
separated for object-like queries.

Nav2 uses a conservative rectangular Go2 footprint including leg sweep,
footprint padding and a wider inflation band. A bounded TF retry queue holds a
synchronized semantic frame briefly when its historical transform is late;
the mapper still never substitutes the latest pose for a missing historical
transform.

## Python runtime dependencies

Install these Python packages in the ROS environment used to run the nodes:

```bash
pip install numpy scipy torch open_clip_torch pillow transformers tokenizers
```

The ROS package dependencies are declared in `package.xml`. The three runtime parameter presets are:

- `config/semantic_mapping_m2dgr.yaml`
- `config/semantic_mapping_sim_livox.yaml`
- `config/semantic_mapping_lite3_real.yaml`
- `config/fast_lio_lite3_offline.yaml`

The Lite3 preset disables feature-only query fallback. Its camera calibration,
LiDAR-to-camera transform and command bridge still need to be replaced with
measured values before real-robot trials.

## Lite3 sensor capture

The verified robot-side interfaces are `/timefix/lidar`, `/timefix/imu`,
`/camera/color/image_raw`, and `/camera/color/camera_info`. Copy the standalone
capture, validation and timestamp-audit scripts to the Foxy robot, keep the
LiDAR and RealSense drivers running in separate terminals, and start the third
terminal with:

```bash
bash ~/lite3_tools/record_lite3_sensors.sh
```

The capture script never publishes motion commands. It records IMU, LiDAR and
raw RGB with three independent rosbag2 processes, then validates database
integrity, topic types, rates, camera count agreement and the common capture
interval. `audit_lite3_timestamps.py` separately distinguishes rosbag receipt
jitter from gaps in sensor `header.stamp`.
See the [Lite3 runbook](docs/RUNBOOK.md#8-lite3-实机传感器采集与上机前清单)
for deployment, acceptance and USB restart rules.

After a split capture passes validation, run the complete static perception
chain on the Humble development computer without exposing DDS outside the
computer:

```bash
bash scripts/run_lite3_offline_smoke.sh \
  --input \
  ysc@192.168.1.103:/home/ysc/lite3_bags/lite3_concurrent_20260723_141216_KkOBA6
```

The runner copies or references the immutable source bags, merges the four
sensor topics into one replay timeline, and runs FAST-LIO, CPU CLIP and
GA-BSVM under an isolated ROS domain. It never starts Nav2, active perception
or a robot command bridge. A successful result is deliberately named
`ALGORITHM_STATIC_PASS_NON_GEOMETRIC`: it verifies the offline software chain,
not camera-LiDAR calibration, map accuracy or readiness to move the robot.
Before working away from the robot, follow the
[travel checklist](docs/RUNBOOK.md#89-离开机器狗前的电脑资料清单) so the
raw capture, model cache, robot-side configuration and hardware evidence are
available offline.

## SegFormer fusion backend

`segformer_node` preserves the source image header and publishes a 12-class
mask, per-pixel confidence, RGB preview and the source frame used for color
attributes:

- `/segformer/class_mask` (`mono8`, values `0..11`)
- `/segformer/confidence` (`32FC1`)
- `/segformer/color_mask` (`rgb8`)
- `/segformer/source_image` (`rgb8`)

GA-BSVM accepts `semantic_backend=clip` (the default) or
`semantic_backend=segformer`. In SegFormer mode it synchronizes the class mask,
confidence image and point cloud by their source timestamps. Run the M2DGR
SegFormer fusion path with two nodes:

```bash
ros2 run semantic_mapping segformer_node --ros-args \
  --params-file config/semantic_mapping_m2dgr.yaml

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=segformer
```

`clip_node` is not required in this mode. The configured classes are `road`,
`building`, `tree`, `person`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`,
`chair`, `bench`, and `unknown background`. Common aliases such as `bike` and
`pedestrian` are accepted. GA-BSVM queries the fused categorical posterior,
publishes `/query_target_pose`, and then publishes the safe approach
`/goal_pose`. Use the CLIP backend when arbitrary open-vocabulary text outside
the configured classes is required.

Basic color attributes are supported for `red`, `orange`, `yellow`, `green`,
`blue`, `purple`, `brown`, `black`, `white`, and `gray`. A query such as
`blue car` or `red bicycle` first forms spatial 3D instances from the semantic
class, then checks the mean color score and matching-color support ratio of each
complete instance. This prevents a white headlight on a red vehicle from being
treated as a white vehicle. It remains a lightweight attribute path and does not
replace instance segmentation for overlapping objects or complex descriptions.

The default Cityscapes model maps road/sidewalk to `road`, structural labels to
`building`, vegetation to `tree`, person/rider to `person`, and keeps car,
truck, bus, bicycle and motorcycle separate. Classes absent from Cityscapes,
including chair and bench, plus predictions below the confidence threshold,
map to `unknown background`. Use the CLIP backend for those open-vocabulary
queries. The Lite3 preset requests CUDA FP16; final deployment should use an
ONNX/TensorRT engine built for the robot's JetPack and TensorRT versions.

## Gazebo semantic benchmark

Use `outdoor_semantic_benchmark.world` for the first end-to-end `person` test.
The original outdoor world uses animated Gazebo actors, which are visible to the
camera but have no ray-sensor collision and therefore cannot produce fused
camera/LiDAR person voxels. The benchmark world contains one stationary actor at
`(6, 0)` and an invisible collision proxy at the same position. The simulation
preset uses a `4 x 6` CLIP grid and requires the closed-set `person` probability
to pass before a goal can be published.

The moving actors in `outdoor_terrain.world` are useful only after dynamic
targets receive matching LiDAR collision proxies or the target tracker is
changed to support camera-only detections.

For a deterministic `car` test, use `outdoor_car_benchmark.world`. It contains
one static red car at `(6, 0)` whose visible body and LiDAR collision belong to
the same SDF model. Publish `car` on `/text_query`; the estimated object pose
should be close to `(6, 0)` and the navigation approach pose should remain about
`2.0 m` from the observed car surface cluster.

## Algorithm checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test/test_voxel_map_algorithm.py test/test_query_selection.py \
  test/test_active_perception.py test/test_navigation_safety_config.py \
  test/test_nav_goal_bridge.py \
  test/test_segformer_mapping.py test/test_semantic_schema.py \
  test/test_lite3_capture_validation.py
python3 -m py_compile semantic_mapping/*.py launch/*.py
bash -n scripts/record_lite3_sensors.sh
colcon build --symlink-install --packages-select semantic_mapping
```
