# 运行手册

核对日期：2026-09-29

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

训练/单图工具已迁到 `semantic_mapping/offline/`，仿真启动门已迁到 `gazebo/`；
`ros2 run` 和 launch 的名称、参数不变。更新源码后需要重新构建，让安装入口指向新位置。

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
python3 scripts/check_b_disk_runtime.py --backend segformer --ontology-profile indoor7
ros2 pkg executables semantic_mapping
```

如果预检失败，按它报告的缺项修复，不复制其他磁盘上的旧 `build/`、`install/` 或
`log/`。户外 SegFormer 改用 `--ontology-profile outdoor13`（默认）；CLIP 对比改用
`--backend clip`，不按 profile 选择缓存。预检检查默认模型文件和依赖，不证明运行推理或导航成功。
本地模型齐全且不需下载时，可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，
避免启动时访问 Hugging Face；自定义 checkpoint 需单独确认文件和加载结果。

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
| PoseStamped | `/goal_pose` | 语义约束接近点，路径/碰撞由 Nav2 检查 |

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
但关闭 RPP 的预测碰撞 veto；历史 Small House 到达案例使用了此设置，当前代码仍需复测。真实机仍使用 `nav2_params.yaml`，不要把这个仿真参数文件用于实机。

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
watchdog 发布零速。不确定性点云按最新融合观测的源时间判断过期，包含 TF 等待时间。
`STALE[...]` 会在 `/perception_mode` 中说明缺少或过期的输入。

这些话题证明减速链在运行，不单独证明减速提高了 Navigation Success。正式收益需要同一
world、start、query、seed 和障碍条件下做 OFF/ON 对比。

## 3. Small House benchmark runner

`config/benchmark/small_house_manifest.yaml` 是可追踪的实验说明：world、目标真值、起点、case、
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
  --manifest config/benchmark/small_house_manifest.yaml \
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
每轮查询尝试最多评估两个候选，只有接近点确认可行后才发布 target/goal。收到新查询或被
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
- accepted 但无 target/goal：检查类别主导体素、证据阈值、聚类、可通行支持及接近距离；
- 当前代码在接近点有效后才发布 target/goal；只收到其中之一时，先查订阅时机和通信；
- 有 goal 但不移动：查看 Goal Bridge、Nav2 costmap 与规划器日志。

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

Lite3 命令集中在下方第 8 节，已有证据和运动门禁集中在实机交接文档。

## 8. Lite3 实机传感器采集与上机前清单

Lite3 机载 Jetson 使用 Ubuntu 20.04、ROS 2 Foxy 和厂商工作区。第 8.2–8.4 节在机器狗
SSH 终端执行，不使用第 1 节的 Humble 环境；第 8.8–8.9 节在开发电脑执行。传感器数据和 Bag 都保存在机器狗
本地，因此机器狗不需要访问互联网，也不依赖电脑与机器狗之间的 DDS 组播。

### 8.1 已确认接口

| 数据 | 话题 | 类型 | 实测频率 |
| --- | --- | --- | --- |
| Mid360 点云 | `/timefix/lidar` | `livox_ros_driver2/msg/CustomMsg` | 约 10 Hz |
| Mid360 IMU | `/timefix/imu` | `sensor_msgs/msg/Imu` | 约 200 Hz |
| D435I 彩色图像 | `/camera/color/image_raw` | `sensor_msgs/msg/Image` | `424x240` Bag 中约 15 Hz |
| D435I 内参 | `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Bag 中约 15 Hz |

`/timefix/lidar` 的 `header.stamp`、`timebase` 和点偏移已恢复为同一时间基准。这里的频率
只证明消息与录包链路可用，不等于 LiDAR、IMU 和相机外参或硬件同步已经完成标定。

### 8.2 终端 1：Mid360 与 IMU

机器狗保持趴下，在第一个 SSH 终端执行：

```bash
source /opt/ros/foxy/setup.bash
source ~/lite_cog_ros2/driver/mid360_ws/install/setup.bash

LIVOX_CONFIG=~/lite_cog_ros2/driver/mid360_ws/src/livox_ros_driver2-master/config/MID360_config.json

ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=1 \
  -p multi_topic:=0 \
  -p data_src:=0 \
  -p publish_freq:=10.0 \
  -p output_data_type:=0 \
  -p frame_id:=rslidar \
  -p user_config_path:="$LIVOX_CONFIG" \
  -p cmdline_input_bd_code:=livox0000000001 \
  -r /livox/lidar:=/timefix/lidar \
  -r /livox/imu:=/timefix/imu
```

该终端持续运行，不要再启动第二个 Livox 驱动。

### 8.3 终端 2：D435I

第二个 SSH 终端执行：

```bash
source /opt/ros/foxy/setup.bash
source ~/lite_cog_ros2/driver/realsense_ws/install/setup.bash

ros2 launch realsense2_camera dr_camera_launch.py \
  enable_color:=true \
  color_width:=424 \
  color_height:=240 \
  color_fps:=15.0 \
  enable_depth:=true \
  depth_width:=424 \
  depth_height:=240 \
  depth_fps:=15.0 \
  enable_pointcloud:=false \
  enable_sync:=true
```

2026-07-26 使用 `rs-enumerate-devices` 确认该 D435I 原生支持 Color RGB8 和 Depth
Z16 的 `424x240@15Hz` 模式。相对原来的 `640x480`，彩色图每帧数据量约降低 67%，
更适合 Jetson 同时写入 LiDAR、IMU 和原始 RGB。当前厂商 wrapper 的纯彩色模式曾出现
“话题存在但没有图像”的情况，因此保留同分辨率深度流作为稳定启动基准；采集脚本
不会记录深度图。启动时短暂出现
`control_transfer ... error 11` 可以记录为警告。若它持续刷屏并伴随图像停流，应停止本次采集并检查 USB 供电、带宽和驱动，再重启传感器。

### 8.4 静止录制与验收

机器狗终端 3：前两个传感器终端保持运行，使用已部署到 `~/lite3_tools` 的当前脚本。
脚本依赖同目录的 `verify_lite3_capture.py` 和 `audit_lite3_timestamps.py`。

```bash
source /opt/ros/foxy/setup.bash
source ~/lite_cog_ros2/driver/mid360_ws/install/setup.bash
bash ~/lite3_tools/record_lite3_sensors.sh
```

输出位于机器狗 `~/lite3_bags/`；以脚本打印的本次目录为准。录制中保持静止，
需要提前结束时在录制终端按 Ctrl-C，等待 recorder 完成关闭。通过要求为四路输入
频率、共同时间窗和 header 连续性满足脚本门限；报告中的接收调度警告不等同于
传感器 header 断流。已有推荐静止 Bag，无配置或安装变化时不重复录制。

TF 权威、受控运动与 SDK 桥接的门限见 [Lite3 交接第 6 节](LITE3_REAL_HANDOFF.md#6-运动前不可跳过的阻塞项)。
这里的传感器、录制和离线命令均不授权底盘运动。

### 8.8 开发电脑离线结构烟测

电脑终端：需本地 CLIP 权重、至少 8 GiB 空间和当前工作区。先检查运行环境，
只有 `OVERALL=B_DISK_RUNTIME_READY` 后才运行；环境检查失败先处理依赖，不回放 Bag。

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
cd /home/yk/ws/src/semantic_mapping
export HF_HOME=/home/yk/ws/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 scripts/check_b_disk_runtime.py --backend clip
```

环境检查通过后：

```bash
bash scripts/run_lite3_offline_smoke.sh \
  --input /home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7 \
  --work-root /home/yk/ws/lite3_offline_runs
```

默认 CPU CLIP、隔离 domain 42、仅本机 DDS、0.1 倍速；脚本不启动 Nav2 或运动桥。
每次运行在指定目录下创建独立批次，保留参数、Git 状态、日志、merged/output Bag
和验收报告。Ctrl-C 会清理本次进程组并记录 `ABORTED`；不要删除源 Bag。

通过状态 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC` 仅证明静止输入的软件链通过。
当前 CLIP 帧携带源图 Header，投影按观测时间匹配 TF；真实运动同步、几何精度和
实机执行仍需独立验收。实机标定和运动开关继续关闭。

### 8.9 离线复现与迁移备份清单

只读验证历史精简归档：

```bash
cd /home/yk/ws/src/semantic_mapping
python3 scripts/verify_lite3_migrated_archive.py \
  --source-dir /home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7 \
  --run-dir /home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN
```

期望 `SOURCE_HASHES`、`RETAINED_ARTIFACTS`、`EXPECTED_OMISSIONS`、
`IMPLEMENTATION_HASHES` 均为 `PASS`，且 `OVERALL=B_DISK_COMPACT_ARCHIVE_PASS`。
验证器按 manifest 的历史 commit 检查实现；不要改写历史来源路径。

该验证器仅适用历史精简目录，不能用于包含全部 payload 的完整烟测。完整证据位于
`/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260823T030733Z_wR2TtN`，其
`merged/`、`output_bag/`、manifest 和报告必须保留。归档完整性不代表当前运行环境可用，
更不代表 `MOTION_READY=YES`。各状态含义见 [Lite3 交接](LITE3_REAL_HANDOFF.md)。
