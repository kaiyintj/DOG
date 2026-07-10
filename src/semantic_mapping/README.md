# semantic_mapping

ROS 2 nodes for CLIP-based semantic mapping, GA-BSVM voxel fusion, semantic costmap publishing, and active perception speed modulation.

## Mapping and navigation method

The online map uses reliability-weighted categorical Dirichlet evidence. Each
camera/LiDAR observation is weighted by synchronized IMU motion, local point
density, sensor range, image-edge projection and CLIP semantic entropy. Points
from one frame are aggregated per voxel and contribute bounded evidence, which
avoids confidence being determined only by LiDAR sampling density.

Each voxel stores both a closed-set semantic posterior and a normalized CLIP
feature. Language targets are selected from spatial clusters rather than a
single maximum-similarity voxel. Class probability, CLIP similarity, cluster
support and robot distance contribute to the target utility. Object-like goals
must have a sufficiently clear road voxel nearby before a Nav2 goal is sent.

`active_perception_node` converts the fused semantic and epistemic uncertainty
along the local path into a filtered linear-velocity scale. Angular velocity is
preserved by default so that uncertainty does not prevent path tracking.

In Gazebo the command chain is deliberately separated to keep a single writer
at each stage: Nav2 publishes `/cmd_vel_nav`, its velocity smoother publishes
`/cmd_vel`, active perception publishes `/cmd_vel_champ`, and CHAMP consumes
only `/cmd_vel_champ`.

`/query_target_pose` is the estimated object position. `/goal_pose` is the
collision-aware navigation approach point, so the two poses are intentionally
separated for object-like queries.

## Python runtime dependencies

Install these Python packages in the ROS environment used to run the nodes:

```bash
pip install numpy scipy torch open_clip_torch pillow
```

The ROS package dependencies are declared in `package.xml`. The three runtime parameter presets are:

- `config/semantic_mapping_m2dgr.yaml`
- `config/semantic_mapping_sim_livox.yaml`
- `config/semantic_mapping_lite3_real.yaml`

The Lite3 preset disables feature-only query fallback. Its camera calibration,
LiDAR-to-camera transform and command bridge still need to be replaced with
measured values before real-robot trials.

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
  test/test_voxel_map_algorithm.py test/test_query_selection.py
python3 -m py_compile semantic_mapping/*.py launch/*.py
colcon build --symlink-install --packages-select semantic_mapping
```
