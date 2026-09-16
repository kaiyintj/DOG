# 运行手册

更新日期：2026-09-09

本手册只保存当前可执行流程。能力边界见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，
室内实验定义和结果见 [INDOOR_GAZEBO_BENCHMARK.md](INDOOR_GAZEBO_BENCHMARK.md)。

## 1. 通用准备

以下示例假定工作空间为 `/home/yk/ws`：

```bash
export SEMANTIC_WS=/home/yk/ws
source /opt/ros/humble/setup.bash
source "$SEMANTIC_WS/install/setup.bash"
export HF_HOME="$SEMANTIC_WS/.cache/huggingface"
```

只修改 `semantic_mapping` 后，定向构建该包：

```bash
cd "$SEMANTIC_WS"
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select semantic_mapping
source "$SEMANTIC_WS/install/setup.bash"
```

运行前确认模型与 ROS 入口：

```bash
cd "$SEMANTIC_WS/src/semantic_mapping"
python3 scripts/check_b_disk_runtime.py --backend segformer
ros2 pkg executables semantic_mapping
```

如果预检失败，按它报告的缺项修复，不复制其他磁盘上的旧 `build/`、`install/` 或
`log/`。CLIP 对比实验把 `--backend segformer` 改为 `--backend clip`。

## 2. Gazebo 统一入口

室内和室外共用 `semantic_sim.launch.py`，不是两套运行代码：

| 场景 | 参数 |
| --- | --- |
| Small House | `ontology_profile:=indoor7 world:=aws_small_house` |
| School Parking Lot | `ontology_profile:=outdoor13 world:=school_parking_lot` |

每个仿真终端都先设置相同的隔离域：

```bash
export SEMANTIC_WS=/home/yk/ws
source /opt/ros/humble/setup.bash
source "$SEMANTIC_WS/install/setup.bash"
export HF_HOME="$SEMANTIC_WS/.cache/huggingface"
export ROS_DOMAIN_ID=216
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI=http://127.0.0.1:11357
```

启动新实验前，先在旧 launch 终端按 `Ctrl+C`。不要同时启动两个 Gazebo master、两个
FAST-LIO 或两个 GA-BSVM。

### 2.1 先键盘移动和建图

终端 1：启动 Small House、传感器、FAST-LIO、SegFormer、GA-BSVM 和 RViz：

```bash
ros2 launch semantic_mapping semantic_sim.launch.py \
  ontology_profile:=indoor7 \
  world:=aws_small_house \
  navigation_enabled:=false \
  gui:=true \
  rviz:=true \
  ros_domain_id:=216 \
  gazebo_master_uri:=http://127.0.0.1:11357
```

传感器门会等待控制器、连续稳定 IMU 和非空 LiDAR，再启动 FAST-LIO。RViz 因此不会
与 Gazebo 同时立刻出现；等待门通过后，它会由 FAST-LIO 启动。

终端 2：键盘控制 Go2：

```bash
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/cmd_vel_champ
```

此模式仍包含 SegFormer 与 GA-BSVM，只关闭 Nav2、Goal Bridge 和主动减速。键盘直接成为
`/cmd_vel_champ` 的发布者，所以不要同时启动自动导航速度链。

户外使用完全相同的两条命令，只替换终端 1 的场景参数：

```text
ontology_profile:=outdoor13 world:=school_parking_lot
```

### 2.2 RViz 显示

Fixed Frame 使用 `odom`。根据需要添加以下 display：

| 类型 | Topic | 用途 |
| --- | --- | --- |
| Odometry | `/Odometry` | FAST-LIO 位姿 |
| PointCloud2 | `/cloud_registered` | 当前配准点云 |
| PointCloud2 | `/Laser_map` | FAST-LIO 地图 |
| PointCloud2 | `/semantic_cloud` | GA-BSVM 类别地图，Color Transformer 选 RGB8 |
| PointCloud2 | `/uncertainty_cloud` | 融合不确定度 |
| OccupancyGrid | `/semantic_cost_map` | 语义代价地图 |
| Image | `/segformer/color_mask` | 分割可视化 |
| PoseStamped | `/query_target_pose` | 目标估计 |
| PoseStamped | `/goal_pose` | 安全接近点 |

如果 RViz 没打开，先确认传感器门是否已成功退出，再确认进程：

```bash
ros2 node list
ros2 topic hz /Odometry
ros2 topic hz /cloud_registered
```

### 2.3 在同一张地图上转入语义导航

1. 在键盘终端按 `Ctrl+C`，停止它发布 `/cmd_vel_champ`。
2. 保持终端 1 的 Gazebo、FAST-LIO、SegFormer 和 GA-BSVM 继续运行。
3. 新开终端启动 Nav2、主动感知减速和 Goal Bridge。

室内：

```bash
export SEMANTIC_WS=/home/yk/ws
source /opt/ros/humble/setup.bash
source "$SEMANTIC_WS/install/setup.bash"
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1

ros2 launch semantic_mapping nav_sim.launch.py \
  use_sim_time:=true \
  odom_topic:=/Odometry \
  active_perception_params_file:="$SEMANTIC_WS/src/semantic_mapping/config/semantic_mapping_sim_indoor.yaml" \
  active_perception_enabled:=true \
  goal_bridge_enabled:=true
```

Gazebo 入口默认使用 `config/nav2_sim_params.yaml`。它仍保留点云和语义代价层供规划使用，
但关闭 RPP 的预测碰撞 veto；当前 Small House 的语义接近点需要这个仿真专用设置才能完成
最后一段到达。真实机仍使用 `nav2_params.yaml`，不要把这个仿真参数文件用于实机。

户外只把参数文件改为：

```text
/home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml
```

此时速度链是：

```text
Nav2 /cmd_vel -> active_perception_node -> /cmd_vel_champ -> CHAMP
```

确认 action 和单一最终速度发布者：

```bash
ros2 action info /navigate_to_pose
ros2 topic info /cmd_vel_champ --verbose
```

`/cmd_vel_champ` 应只有主动感知节点这一条自动控制发布链。发现键盘仍是发布者时，不要发送
导航目标，先停止键盘。

也可以从一开始就令终端 1 使用 `navigation_enabled:=true`，一次启动完整链；但首次观察
场景时，建议先用键盘模式确认传感器、定位和地图。

### 2.4 查询目标

正常导航不需要预先监听任何 topic。只要终端 1 的语义主链和终端 3 的 Nav2 仍在运行，
新开一个已设置相同 ROS domain 的终端直接查询：

```bash
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 run semantic_mapping semantic_query chair
```

该命令会等待 GA-BSVM 的 `/text_query` 订阅者出现、发布一次查询后退出。GA-BSVM 找到
target 和 approach goal 后，Goal Bridge 会自动交给 Nav2。旧命令
`ros2 run semantic_mapping clip_query chair` 仍可用，但只是兼容别名，不表示必须使用
CLIP backend。

户外直接查询：

```bash
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 run semantic_mapping semantic_query car
```

只有需要调试时，才在查询前另开终端监听瞬时结果。显式写消息类型可以避免 ROS 2 CLI
自动发现缓存导致的类型判断失败：

```bash
export ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 topic echo /query_target_pose geometry_msgs/msg/PoseStamped --once
```

```bash
ros2 topic echo /goal_pose geometry_msgs/msg/PoseStamped --once
```

当前 Small House 也可直接查询 `table`。`floor` 不可查询，场景不存在或证据不足的目标应
保持不发布 `/goal_pose`，这是失败关闭，不是节点崩溃。

### 2.5 观察 GA-BSVM、导航和减速

```bash
ros2 topic hz /segformer/project_posterior
ros2 topic hz /semantic_cloud
ros2 topic echo /nav_goal_bridge/status
ros2 topic echo /semantic_speed_scale
ros2 topic echo /perception_mode
```

`/semantic_speed_scale` 是主动感知速度倍率。室内配置中，正常倍率范围为 0.3–1.0；路径
缺少地图支持时目标倍率为 0.65；必要输入过期时降到 0.3；速度命令超过 0.25 秒未更新时
watchdog 发布零速。`STALE[...]` 会在 `/perception_mode` 中说明缺少或过期的输入。

这些话题证明减速链在运行，不单独证明减速提高了 Navigation Success。正式收益需要同一
world、start、query、seed 和障碍条件下做 OFF/ON 对比。

## 3. Small House benchmark runner

`benchmark/small_house_manifest.yaml` 是可追踪的实验说明：world、目标真值、起点、case、
seed 和 checkpoint。它不是 Gazebo world，也不是一次运行产生的日志。

运行一个失败关闭 smoke：

```bash
export SEMANTIC_WS=/home/yk/ws
source /opt/ros/humble/setup.bash
source "$SEMANTIC_WS/install/setup.bash"
export HF_HOME="$SEMANTIC_WS/.cache/huggingface"
cd "$SEMANTIC_WS/src/semantic_mapping"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="$SEMANTIC_WS/indoor_benchmark_runs"
ros2 run semantic_mapping run_indoor_semantic_benchmark \
  --manifest benchmark/small_house_manifest.yaml \
  --case floor_nonqueryable \
  --output "$RESULT_ROOT/aws_small_house/$RUN_ID/floor_nonqueryable"
```

可用 case 以 manifest 为准。先运行 `validate_kitchen_close` 验证起点，再运行
`chair_kitchen` 或 `table_kitchen`。runner 会创建目标输出目录，因此不要复用已存在路径。

`/home/yk/ws/indoor_benchmark_runs/` 是统一的本机实验结果根目录，和源码仓库分开保存。
正式公开结果应从 `result.json` 提炼为小型表格或报告，不把 ROS logs、`simulation.log` 或诊断
压缩包提交到 `docs/`。

## 4. M2DGR Bag：SegFormer 主线

Bag 只验证感知、定位、语义融合和目标生成，不验证机器人执行。

终端 1：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 launch fast_lio mapping.launch.py \
  config_file:=m2dgr.yaml use_sim_time:=true rviz:=true
```

终端 2：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ros2 run semantic_mapping segformer_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p use_sim_time:=true -p device:=cpu
```

终端 3：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=segformer -p use_sim_time:=true
```

终端 4，CPU 推理先用低速播放：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
ros2 bag play /home/yk/Downloads/gate_03_ros2 \
  --topics /velodyne_points /handsfree/imu /camera/color/image_raw/compressed \
  --rate 0.1 --clock 100 --read-ahead-queue-size 10000
```

先查询 `car`。查询时地图证据不足会在地图 revision 变化且距离上次尝试至少 1 秒后重试；
同一次查询最多评估两个候选，只有接近点确认可行后才发布 target/goal。收到新查询或被
能力门拒绝后结束当前重试。

## 5. CLIP 对比路线

CLIP 只作为开放词汇对比。M2DGR 的 FAST-LIO 和 Bag 播放不变，将语义节点替换为：

```bash
ros2 run semantic_mapping clip_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p use_sim_time:=true
```

GA-BSVM 使用：

```bash
ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=clip -p use_sim_time:=true
```

不要同时启动 `clip_node` 与 `segformer_node`，也不要启动两个 GA-BSVM。`/query_feature`
有消息只表示文本已编码，最终结果仍以 `/query_target_pose` 和 `/goal_pose` 为准。

## 6. 常见问题

### 没有语义地图

按数据流顺序检查：

```bash
ros2 node list
ros2 topic hz /segformer/project_posterior
ros2 topic hz /cloud_registered
ros2 run tf2_ros tf2_echo odom base_link
ros2 topic hz /semantic_cloud
```

posterior 正常而融合点长期为 0 时，检查 CameraInfo、LiDAR–camera 外参、时间同步和 QoS。
Indoor-7 正常后验编码应为 `16FC8`。

### 查询没有 target 或 goal

先读 GA-BSVM 日志中的能力回执和拒绝原因：

- `unsupported`：checkpoint 没有该类；
- `unresolved`：当前 profile 没有该查询或别名；
- `nonqueryable`：类别用于结构/通行性，不是目标；
- `not_ready`：profile、posterior 或标定能力尚未就绪；
- accepted 但无 target：检查类别主导体素、证据阈值和聚类数量；
- 有 target 但无 goal：检查 traversable 支持、机器人同侧和接近距离约束；有 goal 但不移动时查看 Nav2 costmap 与规划器日志。

不要通过把 unknown 强映射为目标类、关闭类别主导门或取消安全接近约束来伪造正例。

### 有 goal 但机器人不动

```bash
ros2 node list | rg nav_goal_bridge_node
ros2 action info /navigate_to_pose
ros2 topic hz /cmd_vel
ros2 topic hz /cmd_vel_champ
ros2 topic echo /nav_goal_bridge/status
```

action 不可用时检查 Nav2 生命周期；有 `/cmd_vel` 但没有 `/cmd_vel_champ` 时检查主动感知
节点及其 stale 原因；两者都有而不移动时检查 CHAMP 控制器和最终速度话题的发布者数量。

### Gazebo 或 ROS 图混入旧进程

回到启动旧实验的终端逐一 `Ctrl+C`。确认 11357 端口和 domain 216 不再由旧实验占用后，
再启动新 world。不要用全局进程清理命令误伤其他 ROS 工作。

## 7. 结果与专题入口

- 室内 Gazebo：[INDOOR_GAZEBO_BENCHMARK.md](INDOOR_GAZEBO_BENCHMARK.md)
- Lite3 实机：[LITE3_REAL_HANDOFF.md](LITE3_REAL_HANDOFF.md)
- CARLA 图像：[CARLA_IMAGE_BENCHMARK.md](CARLA_IMAGE_BENCHMARK.md)
- CARLA 三维可靠性：[CARLA_RELIABILITY_BENCHMARK.md](CARLA_RELIABILITY_BENCHMARK.md)
- 电动自行车训练：[SEGFORMER_EBIKE_FINETUNE.md](SEGFORMER_EBIKE_FINETUNE.md)

Lite3 的采集、离线复验和运动前门禁只维护在实机交接文档中，本手册不复制。当前室内任务
也不授权真实机器人运动。
