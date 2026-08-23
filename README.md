# semantic_mapping

ROS 2 nodes for CLIP/SegFormer perception experiments, GA-BSVM voxel fusion,
semantic costmap publishing, and active perception speed modulation.

Current project truth and reproducible commands are maintained in:

- [Current version status](docs/PROJECT_STATUS.md)
- [Bag, Gazebo and Lite3 runbook](docs/RUNBOOK.md)
- [Lite3 real-machine handoff](docs/LITE3_REAL_HANDOFF.md)
- [CARLA CLIP/SegFormer image benchmark](docs/CARLA_IMAGE_BENCHMARK.md)
- [CARLA point-level reliability benchmark](docs/CARLA_RELIABILITY_BENCHMARK.md)

The status document distinguishes implemented code from validated behavior and
known deployment gaps. Its `新对话交接摘要` section is the handoff point for a
new conversation. Use it instead of commands copied from older chats.

Historical review reports, timestamped planning snapshots and ARIS traces are
kept for audit, but are excluded from the default repository search by
`.rgignore`. They are not current status sources. For current paper claims use
`docs/THESIS_PROPOSAL.md` and `idea-stage/docs/research_contract.md`. The
un-timestamped CARLA/ARIS plan and tracker under `refine-logs/` are historical on
the migrated B disk; use them only after the user explicitly resumes that branch
and its excluded installation and data have been restored.

## Code layout

Python sources are split inside the package:

- `semantic_mapping/runtime/`: modules the real robot system uses, plus the
  shared core they depend on (VoxelMap, semantic ontology, posterior,
  projection, reliability factors, SegFormer/CLIP nodes, active perception,
  navigation goal bridge).
- `semantic_mapping/carla/`: CARLA-simulation-only capture and evaluation
  tooling. Nothing under `runtime/` imports `carla/`.

## Mapping and navigation method

The online map uses reliability-weighted categorical Dirichlet evidence. Each
camera/LiDAR observation is weighted by synchronized IMU motion, local point
density, sensor range, image-edge projection and semantic entropy. Points
from one frame are aggregated per voxel and contribute bounded evidence, which
avoids confidence being determined only by LiDAR sampling density.

With the CLIP backend, each voxel stores both a closed-set semantic posterior
and a normalized CLIP feature. With the SegFormer backend, all checkpoint
probabilities are summed into the 13 project classes before ``argmax``. A
Header-bearing native-resolution FP16 posterior grid updates the same voxel
posterior without inventing a uniform distribution for the non-maximum
classes. Both backends also fuse the RGB color observed at each projected
LiDAR point.
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
that DWB collision-checked. The categorical bound is derived from the configured
class count, incoming paths and uncertainty clouds are transformed into `odom`,
the risk score blends its mean with an upper percentile, and scale changes are
limited per second rather than per callback.

The semantic map continuously decays evidence against elapsed time rather than
callback count. Periodic pruning also ages voxels that are not hit again, while
class/color/feature weights use bounded EMA-style accumulation. Dynamic classes
still have a short TTL, and the remaining map is bounded by age, distance and
voxel count. Ray-based free-space clearing is not implemented yet. Its Nav2
projection only uses a height band relative to `base_link`; the legacy
OccupancyGrid output is disabled by default, leaving `/semantic_cost_map` as
the only semantic Nav2 map topic.

In Gazebo the command chain is deliberately separated to keep a single writer
at each stage: Nav2 publishes `/cmd_vel_nav`, its velocity smoother publishes
`/cmd_vel`, active perception publishes `/cmd_vel_champ`, and CHAMP consumes
only `/cmd_vel_champ`.

`/query_target_pose` is the estimated object position. `/goal_pose` is the
collision-aware navigation approach point, so the two poses are intentionally
separated for object-like queries.

The active-perception gate rejects stale path, uncertainty-cloud and IMU data,
uses sensor-data QoS for IMU, and has a steady-clock 250 ms command watchdog.
`nav_goal_bridge_node` keeps at most one accepted Nav2 goal and one latest
pending goal; newer goals explicitly cancel older ones, with retries, timeout
events and result de-duplication published on `/nav_goal_bridge/status`.

Nav2 uses a conservative rectangular quadruped footprint including leg sweep,
footprint padding and a wider inflation band. Its current padded envelope also
covers the Lite3 standing body dimensions from the product manual, but the
actual stance and leg sweep still require a low-speed physical check. A bounded TF retry queue holds a
synchronized semantic frame briefly when its historical transform is late;
the mapper still never substitutes the latest pose for a missing historical
transform.

## Python runtime dependencies

Install the compatible numerical/model set instead of upgrading packages one
by one. Before ROS tests or launches, inspect the selected backend, Python
imports and ROS overlay through the single read-only interface:

```bash
source /opt/ros/humble/setup.bash
python3 scripts/check_b_disk_runtime.py --backend clip
```

If it reports `B_DISK_RUNTIME_NOT_READY`, follow RUNBOOK section 1. After user
approval, install the platform-specific Torch wheel (desktop CUDA and JetPack
use different builds), then install one backend:

```bash
python3 -m pip install --user -r requirements-segformer.txt
# Only for the CLIP comparison backend:
python3 -m pip install --user -r requirements-clip.txt
```

`requirements-runtime-common.txt` deliberately pairs NumPy 1.26.4 with SciPy
1.11.4 and constrains setuptools to the shared Torch 2.13/colcon-core range
`>=77,<80`. Only `OVERALL=B_DISK_RUNTIME_READY` authorizes B-disk ROS tests or
launches.
`setup.py` declares runtime package names but leaves exact pins in the
requirements files so an offline `colcon build` never replaces the host Python
environment implicitly.

The ROS package dependencies are declared in `package.xml`. The three runtime parameter presets are:

- `config/semantic_mapping_m2dgr.yaml`
- `config/semantic_mapping_sim_livox.yaml`
- `config/semantic_mapping_lite3_real.yaml`
- `config/fast_lio_lite3_offline.yaml`
- `config/fast_lio_lite3_real.yaml`

The Lite3 semantic preset selects SegFormer, requires the matching runtime
`CameraInfo`, disables feature-only query fallback, and deliberately keeps
`projection_calibration_verified: false`. In that state it may build a debug
semantic map but will not publish object or navigation poses. The real FAST-LIO
preset is a candidate for controlled moving-data validation; the offline preset
remains the only one used by the static smoke runner. Measured LiDAR-to-camera
extrinsics, a single authoritative odometry/TF source, and a vendor SDK safety
bridge are still required before motion trials.

Use `nav_sim.launch.py` for Gazebo (`use_sim_time=true`, `/odom`, simulation
profile) and `nav_lite3_real.launch.py` for Lite3 (`use_sim_time=false`,
`/Odometry`, fail-closed real profile). The generic
`nav_with_remap.launch.py` remains an advanced entry point and now defaults to
wall time because its default profile is Lite3. The Lite3 preset writes only
`/cmd_vel_lite3_safe`; a watchdog-protected vendor bridge must be its sole
consumer and sole SDK command writer. Goal forwarding is disabled by default
in the Lite3 wrapper until calibration and that bridge pass acceptance.

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
  /home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7
```

The runner copies or references the immutable source bags, merges the four
sensor topics into one replay timeline, and runs FAST-LIO, CPU CLIP and
GA-BSVM under an isolated ROS domain. It never starts Nav2, active perception
or a robot command bridge. A successful result is deliberately named
`ALGORITHM_STATIC_PASS_NON_GEOMETRIC`: it verifies the offline software chain,
not camera-LiDAR calibration, map accuracy or readiness to move the robot.
The canonical 2026-08-21 run is
`/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN`.
It passed on clean semantic-mapping commit `d27c103`, while calibration remained
unverified and `motion_ready` remained false. The B-disk copy is a compact
evidence archive: reports, logs, generated configuration, manifests and hashes
are retained, while the reproducible `merged/` and `output_bag/` payloads are
intentionally omitted. The immutable source bags remain available separately.
Validate that compact archive with
`scripts/verify_lite3_migrated_archive.py`, then follow the
[offline reproduction and migration checklist](docs/RUNBOOK.md#89-离线复现与迁移备份清单)
so the raw capture, canonical smoke result, model cache, robot-side
configuration and hardware evidence remain available.

## SegFormer fusion backend

### Offline image inspection

The same SegFormer checkpoint and project-class mapping can be run on one
local image without starting ROS or publishing navigation commands:

```bash
ros2 run semantic_mapping segformer_image \
  --image /path/to/image.jpg \
  --device cuda \
  --overlay /tmp/image_segformer_overlay.png \
  --json /tmp/image_segformer.json
```

The command prints the dominant project classes and their pixel fractions and
optionally writes a color overlay and a JSON report. The stock Cityscapes
checkpoint has no native `electric_bicycle` class; an e-bike will therefore
not be reliably separated from bicycle/motorcycle until a trained checkpoint
is selected.

`segformer_node` preserves the source image header and publishes the complete
13-class project posterior at native decoder resolution. It also retains the
historical raw-argmax-mapped class mask, raw maximum confidence and RGB
products for visualization and an exact legacy regression baseline:

- `/segformer/project_posterior` (`16FC13`, native-resolution FP16 probabilities)
- `/segformer/class_mask` (`mono8`, values `0..12`)
- `/segformer/confidence` (`32FC1`)
- `/segformer/color_mask` (`rgb8`)
- `/segformer/source_image` (`rgb8`)

GA-BSVM accepts `semantic_backend=clip` (the default) or
`semantic_backend=segformer`. In the default SegFormer mode it synchronizes the
atomic posterior grid, source RGB and point cloud by source timestamps, samples
the low-resolution posterior at projected LiDAR pixels, and fuses the complete
distribution. Set `segformer_use_full_posterior:=false` only to reproduce the
legacy hard-mask plus maximum-confidence baseline. Run the M2DGR SegFormer
fusion path with two nodes:

```bash
ros2 run semantic_mapping segformer_node --ros-args \
  --params-file config/semantic_mapping_m2dgr.yaml

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=segformer
```

`clip_node` is not required in this mode. The configured classes are `road`,
`building`, `tree`, `person`, `car`, `truck`, `bus`, `bicycle`,
`electric_bicycle`, `motorcycle`, `chair`, `bench`, and `unknown background`.
Common aliases such as `bike`, `electric bike`, `e-bike`, `电动车` and
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

The default Cityscapes model sums road/sidewalk probability into `road`,
structural-label probability into `building`, vegetation into `tree`, and
person/rider into `person`, while keeping car, truck, bus, bicycle and
motorcycle separate. This aggregation occurs before hard classification and
therefore preserves probability mass and class competition. The stock model
has no electric-bicycle output,
so `electric_bicycle` queries fail closed with the stock checkpoint. At startup
the node prints the navigation classes supported by the selected checkpoint. A
fine-tuned SegFormer whose `id2label` contains `electric_bicycle`,
`electric bicycle`, `electric bike`, `e-bike` or `ebike` plugs into the same 13-class
navigation chain without another schema change. Other classes absent from the
checkpoint, plus predictions below the confidence threshold, map to `unknown
background`. Use the CLIP backend for arbitrary open-vocabulary
queries. The Lite3 preset requests CUDA FP16; final deployment should use an
ONNX/TensorRT engine built for the robot's JetPack and TensorRT versions.

The paired-mask dataset format, 13-class classifier initialization, training,
acceptance check and runtime commands for actual electric-bicycle recognition
are documented in
[docs/SEGFORMER_EBIKE_FINETUNE.md](docs/SEGFORMER_EBIKE_FINETUNE.md).

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
python3 -m py_compile semantic_mapping/runtime/*.py \
  semantic_mapping/carla/*.py launch/*.py
bash -n scripts/record_lite3_sensors.sh
colcon build --symlink-install --packages-select semantic_mapping
```
