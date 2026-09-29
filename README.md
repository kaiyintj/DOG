# semantic_mapping

ROS 2 Humble semantic mapping and language navigation for quadruped robots.
The maintained path is:

```text
RGB + LiDAR + IMU
  -> FAST-LIO
  -> SegFormer
  -> camera-LiDAR projection
  -> reliability-weighted Dirichlet voxel fusion (GA-BSVM)
  -> semantic map and filtered approach goal
  -> Nav2
  -> robot
```

SegFormer is the primary closed-set backend. CLIP remains available as an
open-vocabulary comparison backend and is not mixed into SegFormer results.
The primary query interface uses category/colour aliases, not unrestricted language
understanding. “Query published” confirms delivery, not acceptance or arrival.

## Semantic profiles

One profile is selected for an entire run. Switching profile requires restarting
the semantic map because the posterior dimensions differ.

| Profile | Project classes | Checkpoint | Gazebo world |
| --- | --- | --- | --- |
| `outdoor13` | Existing outdoor 13-class ontology | Cityscapes B0 | `school_parking_lot` |
| `indoor7` | floor, wall, door, chair, table, shelf, bed, unknown | ADE20K B0 | `aws_small_house` |

Navigation uses semantic roles rather than fixed class IDs. `road` is traversable
outdoors and `floor` is traversable indoors. The semantic map keeps them as distinct
classes.

## Quick start: Gazebo

Build once after changing this package:

```bash
cd ~/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select semantic_mapping
source install/setup.bash
```

Start Small House with mapping, SegFormer, GA-BSVM and FAST-LIO RViz:

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
export HF_HOME=~/ws/.cache/huggingface

ros2 launch semantic_mapping semantic_sim.launch.py \
  ontology_profile:=indoor7 \
  world:=aws_small_house \
  navigation_enabled:=false \
  gui:=true \
  rviz:=true
```

`navigation_enabled:=false` is the keyboard-mapping mode. SegFormer and GA-BSVM
still run, while Nav2, the goal bridge and active speed modulation stay off. Stop
the keyboard before enabling navigation so `/cmd_vel_champ` has one publisher.

For the corresponding outdoor run, use:

```text
ontology_profile:=outdoor13 world:=school_parking_lot
```

The complete keyboard-to-navigation transition, RViz displays, queries and
troubleshooting are in [RUNBOOK.md](docs/RUNBOOK.md).

With the semantic stack running, send a query directly:

```bash
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
ros2 run semantic_mapping semantic_query chair
```

如果地图里有多个同类物体，GA-BSVM 会先按语义、证据和距离评分排序；排名靠前的物体
没有满足 traversability、同侧和距离约束的接近点时，最多继续尝试下一个候选物体。
候选确认可行前不会发布 `/query_target_pose`；两个候选都失败时会保留查询，等地图
revision 变化且至少间隔 1 秒后再重试。`query_max_candidates` 仍只控制候选构造，
每轮候选尝试数由 `query_fallback_max_attempts`（默认 2）控制。
接近点快照先按本轮候选的搜索邻域筛选，再计算语义置信度；候选共享快照。
规划器返回目标和接近点决策后，由 ROS 节点统一发布。

No topic listeners are required for navigation; they are only diagnostics.

## Main runtime interfaces

| Topic | Meaning |
| --- | --- |
| `/segformer/project_posterior` | Full project-class posterior |
| `/semantic_cloud` | Class-coloured 3D semantic map |
| `/uncertainty_cloud` | Fused uncertainty cloud |
| `/voxel_entropy_data` | Path uncertainty input for speed modulation |
| `/semantic_cost_map` | Nav2 semantic cost layer |
| `/text_query` | Language query input |
| `/query_target_pose` | Estimated target surface-cluster pose |
| `/goal_pose` | Checked approach pose sent toward Nav2 |
| `/semantic_speed_scale` | Active-perception velocity multiplier |
| `/perception_mode` | Speed-gate state and stale-input reason |

`/query_target_pose` is an observation estimate. `/goal_pose` is the approach point
after the traversability, robot-side and distance filters. Nav2 plans and controls
the route using costmaps. The Gazebo preset disables the RPP predictive collision
veto; these filters do not establish collision-free arrival.

## Repository layout

```text
config/                 Runtime presets
  benchmark/            Versioned static experiment manifests
launch/                 ROS 2 launch composition
semantic_mapping/
  runtime/              Online nodes, shared semantic math and checkpoint checks
  offline/              SegFormer training and single-image inspection
  gazebo/               Simulation startup gate and benchmark runner support
  carla/                CARLA-only capture and evaluation
benchmark_runs/         Legacy local output; ignored by Git
docs/                   Current runbooks and focused experiment documents
test/                   Unit and integration tests
```

For reading on Windows, start with [current status](docs/PROJECT_STATUS.md),
[indoor Gazebo results](docs/results/indoor_gazebo_20260915_20260918/README.md),
[Lite3 observations](docs/results/lite3_20260924_20260926/README.md), and
[CARLA Motion V2 results](docs/results/carla_motion_v2_20260813/README.md).
These are dated summaries; large ROS bags, raw images, model weights and local
workspace inventories are stored separately and are not needed to browse this repository.

New indoor recordings belong to `/home/yk/ws/indoor_benchmark_runs/<world>/<run>/`.

`segformer_image`, `segformer_dataset`, `segformer_finetune`, `segformer_checkpoint`
and `sim_sensor_gate` keep their existing command names. Rebuild this package after
updating so installed commands point to the reorganized modules. Online inference
imports shared checkpoint checks directly and does not import the training module.

Gazebo world assets belong to `go2_config`; benchmark manifests belong to this
package because they describe semantic targets, starts and ground truth.

## Documentation

- [Current capability and open work](docs/PROJECT_STATUS.md)
- [Commands for Gazebo, Bag and diagnostics](docs/RUNBOOK.md)
- [Indoor profile and Small House benchmark](docs/INDOOR_GAZEBO_BENCHMARK.md)
- [Lite3 real-robot handoff](docs/LITE3_REAL_HANDOFF.md)
- [CARLA image benchmark](docs/CARLA_IMAGE_BENCHMARK.md)
- [CARLA reliability benchmark](docs/CARLA_RELIABILITY_BENCHMARK.md)
- [SegFormer electric-bicycle fine-tuning](docs/SEGFORMER_EBIKE_FINETUNE.md)

Files under `docs/research/` are optional background material. Current commands and
capability claims live only in the documents listed above.

## Current boundary

Outdoor Gazebo has an established working path. Indoor Small House has working
sensors, FAST-LIO startup, SegFormer/GA-BSVM fusion and fail-closed negative cases.
Historical chair runs on September 16 and 18 reached SUCCEEDED; the latest code
changes still need a new Gazebo run. Table positive navigation, repeatability,
collision-free arrival and speed-modulation benefit remain unverified.
See [dated evidence and limitations](docs/PROJECT_STATUS.md).

Lite3 remains fail closed for motion until its calibration, TF authority and SDK
safety bridge are accepted. See the dedicated handoff before any real-robot work.

For a focused check after an edit:

```bash
cd ~/ws/src/semantic_mapping
python3 -m py_compile launch/semantic_sim.launch.py
git diff --check
```
