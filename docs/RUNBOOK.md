# 运行手册

更新日期：2026-08-13

本手册只描述 `~/ws` 当前代码可以实际执行的流程。系统能力边界见
[PROJECT_STATUS.md](PROJECT_STATUS.md)。

## 1. 通用准备

每个新终端都先执行：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
```

第一条加载 ROS 2 Humble 的命令、消息和系统包；第二条把当前工作区编译出的
`semantic_mapping`、FAST-LIO、Go2 等包叠加到环境中。环境变量不会自动继承到另一个
新终端，所以每个终端都要执行。

只在运行 SegFormer 的终端设置：

```bash
export HF_HOME=~/ws/.cache/huggingface
```

当前 SegFormer 模型缓存在该目录。设置 `HF_HOME` 可以避免 Transformers 转而访问默认的
`~/.cache/huggingface` 并重复下载模型。CLIP、FAST-LIO、Nav2 和 Bag 播放终端不依赖它。

修改源码或 YAML 后重新构建：

```bash
cd ~/ws
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths ~/ws/src/semantic_mapping \
  --symlink-install \
  --packages-select semantic_mapping
source install/setup.bash
```

确认节点已安装：

```bash
ros2 pkg executables semantic_mapping
```

应至少看到：

```text
semantic_mapping active_perception_node
semantic_mapping carla_capture_benchmark
semantic_mapping carla_evaluate_benchmark
semantic_mapping carla_capture_reliability
semantic_mapping carla_evaluate_reliability
semantic_mapping clip_node
semantic_mapping clip_query
semantic_mapping ga_bsvm_node
semantic_mapping nav_goal_bridge_node
semantic_mapping segformer_checkpoint
semantic_mapping segformer_dataset
semantic_mapping segformer_finetune
semantic_mapping segformer_image
semantic_mapping segformer_node
```

代码已按用途分层：`semantic_mapping/runtime/` 放实际运行代码及其依赖的共用核心，
`semantic_mapping/carla/` 只放 CARLA 仿真采集与评测代码；`runtime/` 不依赖
`carla/`。上面这些 ROS 入口的名字没有变，直接按本手册命令运行即可。

CARLA 的二维图像识别基准和三维点级 reliability 基准是两套独立数据格式。
后者的完整采集、评测、时间偏移、Semantic LiDAR GT 对齐和输出解释见
[CARLA_RELIABILITY_BENCHMARK.md](CARLA_RELIABILITY_BENCHMARK.md)。

电动自行车数据校验、微调和权重验收流程见
[SEGFORMER_EBIKE_FINETUNE.md](SEGFORMER_EBIKE_FINETUNE.md)。

检查 Python 模型依赖和实际版本：

```bash
python3 -c "import numpy, scipy, torch, PIL, transformers; print('numpy=', numpy.__version__, 'scipy=', scipy.__version__, 'torch=', torch.__version__, 'Pillow=', PIL.__version__, 'transformers=', transformers.__version__)"
```

当前系统的 SciPy 1.8 与 NumPy 1.26 会产生版本不兼容警告。正式实验使用隔离环境，先安装
与桌面 CUDA 或 JetPack 对应的 Torch，再安装项目锁定的数值与模型依赖：

```bash
python3 -m pip install --user -r ~/ws/src/semantic_mapping/requirements-segformer.txt
```

CLIP 对比路线另用：

```bash
python3 -m pip install --user -r ~/ws/src/semantic_mapping/requirements-clip.txt
```

不要在正在运行的 ROS 系统里临时升级 NumPy、SciPy 或 Torch，也不要让 `colcon build`
隐式替换系统环境。锁定组合见 `requirements-runtime-common.txt`；安装后重新运行上面的
版本检查并保存 `python3 -m pip freeze`。

### 四种运行组合

CLIP 和 SegFormer 是二选一的语义前端，FAST-LIO 与 GA-BSVM 是两条路线共同使用的
模块。按环境选择下面一行：

| 环境 | 启动的语义节点 | GA-BSVM 后端 | Nav2/主动感知 |
| --- | --- | --- | --- |
| M2DGR Bag + CLIP | `clip_node` | `clip` | 不启动 |
| M2DGR Bag + SegFormer | `segformer_node` | `segformer` | 不启动 |
| Gazebo + CLIP | `clip_node` | `clip` | 启动 |
| Gazebo + SegFormer | `segformer_node` | `segformer` | 启动 |

同一次运行不要同时启动 `clip_node` 和 `segformer_node`，也不要同时启动两个
`ga_bsvm_node`。切换路线时停止旧节点，再按对应命令启动。

## 2. 运行前清理与检查

优先在原终端用 `Ctrl+C` 正常停止节点。确认没有旧进程：

```bash
ps -ef | rg "gzserver|gzclient|fastlio_mapping|rviz2|clip_node|segformer_node|ga_bsvm_node|nav_goal_bridge_node|active_perception_node"
```

如果 Gazebo 已经关闭但仍有残留进程，必须一次只给 `pkill` 一个进程名。先温和停止，
等待后再清理仍未退出的进程：

```bash
pkill -TERM -x gzclient 2>/dev/null || true
pkill -TERM -x gzserver 2>/dev/null || true
sleep 3
pkill -KILL -x gzclient 2>/dev/null || true
pkill -KILL -x gzserver 2>/dev/null || true
```

不要写成 `pkill gzclient gzserver`；`pkill` 只接受一个匹配模式，这种写法会直接报错，
两个进程都不会停止。清理后必须确认 Gazebo 默认端口已经释放：

```bash
pgrep -af 'gzserver|gzclient'
ss -ltnp | rg ':11345'
```

两条命令都应无输出。若原来的 `ros2 launch` 终端仍在运行，先在该终端按 `Ctrl+C`，
否则只结束 GUI 不能保证服务器退出。

刷新 ROS 2 节点发现缓存：

```bash
ros2 daemon stop
ros2 daemon start
```

不要同时启动两个 `ga_bsvm_node`、两个 `active_perception_node`，也不要让多个节点
同时向机器人最终速度话题发布命令。

## 3. M2DGR Bag：SegFormer 路线

该流程用于验证 FAST-LIO、逐像素语义、三维融合和目标点生成。Bag 没有可控制的
机器人执行器，因此不能验证真实移动导航。

Bag 路径：

```text
/home/yk/Downloads/gate_03_ros2
```

### 终端 1：FAST-LIO

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch fast_lio mapping.launch.py \
  config_file:=velodyne.yaml \
  use_sim_time:=true \
  rviz:=true
```

### 终端 2：SegFormer

CPU 推理时建议降低 Bag 速度：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
export HF_HOME=~/ws/.cache/huggingface

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

ros2 run semantic_mapping segformer_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p use_sim_time:=true \
  -p device:=cpu
```

### 终端 3：GA-BSVM

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=segformer \
  -p use_sim_time:=true
```

SegFormer 模式不需要启动 `clip_node`。

### 终端 4：播放 Bag

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 bag play /home/yk/Downloads/gate_03_ros2 \
  --topics /velodyne_points /handsfree/imu /camera/color/image_raw/compressed \
  --rate 0.1 \
  --clock 100 \
  --read-ahead-queue-size 10000
```

### 监控输出

```bash
ros2 topic hz /velodyne_points
ros2 topic hz /Odometry
ros2 topic hz /segformer/color_mask
ros2 topic hz /semantic_cloud
ros2 topic hz /semantic_cost_map
```

在两个终端分别提前监听：

```bash
ros2 topic echo /query_target_pose --once
```

```bash
ros2 topic echo /goal_pose --once
```

先用 `car` 做正样本测试。监听命令要在查询前启动，因为目标话题不保存历史消息：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'car'}"
```

当前查询只在 `/text_query` 到达时检查一次已有地图，不会在新语义证据到达后自动重试。
如果日志显示“未找到合适目标”，让 Bag 继续播放、等待 SegFormer 和 GA-BSVM 再融合
一些帧，然后重新发布同一个查询。

本版本在该 Bag 中已经得到过如下有效结果：`car` 目标簇 2 个体素、证据 6.0，观测表面
簇质心约为 `(13.20, -2.75)`，安全接近点约为 `(14.15, -3.15)`。坐标随播放起点、
处理帧和参数可能变化，判断成功应以 GA-BSVM 的“导航触发”日志以及两个 Pose 话题为准。

以下命令用于能力边界测试，不应作为第一次冒烟测试：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'bicycle'}"
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'electric bicycle'}"
ros2 topic pub --once /text_query std_msgs/msg/String "{data: '电动车'}"
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'motorcycle'}"
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'blue car'}"
```

当前 Bag 的 SegFormer 帧统计中 `bicycle` 可能始终为 0；`blue car` 还需要颜色门控通过。
默认 Cityscapes 权重不能输出 `electric_bicycle`，因此两个电动车查询应安全失败；
换用带电动自行车标签的微调 SegFormer 后才能用于能力验收。

如果场景中不存在该类别，或者类别、颜色、证据和聚类阈值未通过，不发布目标是正常的
拒绝行为，不代表节点崩溃。

## 4. M2DGR Bag：CLIP 路线

CLIP 路线用于开放文本检索。FAST-LIO 和 Bag 播放命令与上一节相同，只替换语义节点。

### 终端 2：CLIP

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping clip_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p use_sim_time:=true
```

### 终端 3：GA-BSVM

YAML 默认后端就是 `clip`：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_m2dgr.yaml \
  -p semantic_backend:=clip \
  -p use_sim_time:=true
```

分别在独立终端提前监听文本特征和目标：

```bash
ros2 topic echo /query_feature --once
```

```bash
ros2 topic echo /query_target_pose --once
```

```bash
ros2 topic echo /goal_pose --once
```

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'car'}"
```

`/query_feature` 有输出只说明文本已被 CLIP 编码，并不表示已经找到物体。最终是否找到
以 GA-BSVM 的“导航触发”日志、`/query_target_pose` 和 `/goal_pose` 为准。监听命令应在
发布查询前启动，因为这些话题不是持久化历史记录。CLIP 查询同样是一次性检查，证据
不足时应等待地图继续增长后重新发布。

## 5. RViz 显示

FAST-LIO launch 已经启动 RViz。建议设置：

- Fixed Frame：`odom`；
- PointCloud2 `/cloud_registered`：FAST-LIO 全局点云；
- PointCloud2 `/semantic_cloud`：语义类别点云，Color Transformer 设为 `RGB8`；
- PointCloud2 `/uncertainty_cloud`：不确定性点云，Color Transformer 设为 `RGB8`；
- Map `/semantic_cost_map`：二维语义代价地图；
- Image `/segformer/project_posterior`：原生解码分辨率 `16FC13` 完整后验；
- Pose `/query_target_pose`：当前可见语义表面簇质心，不等于物体几何中心；
- Pose `/goal_pose`：接近点；
- Image `/segformer/color_mask`：SegFormer 预览。

`/semantic_cloud` 的 RGB8 颜色是语义类别调色板，不是物体真实漆面颜色。`white truck`
等颜色查询使用 GA-BSVM 内部从 `/segformer/source_image` 或相机图像融合的观测 RGB，
不能根据 RViz 点云显示成红色就判断它选中了红车。

SegFormer 使用 sensor-data QoS。如果 RViz Image 显示 `No Image` 或出现 incompatible QoS，
将该 Display 的 Reliability Policy 设置为 `Best Effort`。

`/semantic_cloud` 只包含成功投影到当前相机视野并完成体素融合的 LiDAR 点，不会与
`/cloud_registered` 一样覆盖全部 360 度点云。这是当前相机-LiDAR融合方式的预期结果。

## 6. Gazebo：Go2 car benchmark

推荐先使用静态红色汽车世界，因为目标几何体、视觉模型和 LiDAR 碰撞位于同一位置。
终端 1、2、5 是两种后端共用的；终端 3、4 只选择 SegFormer 或 CLIP 中的一组。

### 终端 1：Gazebo 和 Go2

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch go2_config gazebo.launch.py   world:=/home/yk/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/school_parking_lot.world  world_init_x:=0.0   world_init_y:=0.0   world_init_z:=0.35   cmd_vel_topic:=/cmd_vel_champ   gui:=true   rviz:=false   use_sim_time:=true

```
查看图像窗口
ros2 run rqt_image_view rqt_image_view
等待以下话题出现：

```bash
ros2 topic list | rg "clock|livox/lidar|imu/data|d435i/image_raw|cmd_vel"
```

### 终端 2：FAST-LIO

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch fast_lio mapping.launch.py \
  config_file:=sim_mid360.yaml \
  use_sim_time:=true \
  rviz:=true
```

当前 Gazebo 启动链和 FAST-LIO 都可能发布 `odom -> base_link`。该组合可以用于当前
功能调试，但如果 RViz 地图跳动、目标漂移或机器人位置不一致，应停止实验，先解决单一
TF 权威源，不能使用这次数据做论文定量结果。

### 方案 A，终端 3：SegFormer

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
export HF_HOME=~/ws/.cache/huggingface

ros2 run semantic_mapping segformer_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p use_sim_time:=true \
  -p device:=cuda \
  -p use_fp16:=true
```

当前工作站有可用 NVIDIA CUDA 时使用上述设置；若 CUDA 不可用，节点会记录警告并回退
CPU。不要在 GPU 可用时强制 `device:=cpu`，否则 SegFormer 推理更容易压低 Gazebo 的
Real Time Factor，使墙钟时间下的机器人运动忽快忽慢。

### 方案 A，终端 4：GA-BSVM（SegFormer）

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p semantic_backend:=segformer \
  -p use_sim_time:=true
```

SegFormer 路线不要启动 `clip_node`。

### 方案 B，终端 3：CLIP

若要测试 CLIP 路线，用下面两个终端替换方案 A：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping clip_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p use_sim_time:=true
```

### 方案 B，终端 4：GA-BSVM（CLIP）

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file ~/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p semantic_backend:=clip \
  -p use_sim_time:=true
```

CLIP 路线不要启动 `segformer_node`。还可提前运行
`ros2 topic echo /query_feature --once`，用来确认查询文本已经完成编码。

### 终端 5：Nav2

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch semantic_mapping nav_sim.launch.py
```

该 launch 会同时加载 Nav2、`nav2_params.yaml`、`nav_goal_bridge_node` 和
`active_perception_node`。桥接节点把 `/goal_pose` 转换为
`/navigate_to_pose` action；仿真入口固定采用 `/clock`、`/odom` 和
`semantic_mapping_sim_livox.yaml`，将 Nav2 velocity smoother 的 `/cmd_vel`
转发为 `/cmd_vel_champ`。不要再从旧聊天复制通用 launch 加参数的命令。

不要再另开终端重复启动主动感知节点。若只做不带速度门控的 Nav2 结构诊断，可显式传
`active_perception_enabled:=false`；这种模式不属于闭环导航配置。

### 键盘建图与闭环导航必须分开

键盘建图/补充观察时不要启动终端 5 的 Nav2。此时直接让键盘成为
`/cmd_vel_champ` 的唯一发布者：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/cmd_vel_champ
```

检查最终速度话题：

```bash
ros2 topic info /cmd_vel_champ --verbose
```

`Publisher count` 应为 1，发布者应是键盘节点。进入闭环语义导航前停止键盘节点，再启动
终端 5；此时 `/cmd_vel_champ` 的唯一发布者应变为 `active_perception_node`。不要让键盘
和主动感知同时发布 `/cmd_vel_champ`，也不要在 Nav2 action 仍执行时用键盘抢占控制。

注意：`nav_sim.launch.py` 默认在内部启动 `active_perception_node`，即使用户只手动
打开了“Nav”这一个终端，主动感知依然存在。可用以下命令核对实际速度链：

```bash
ros2 node list | rg 'active_perception|nav_goal_bridge'
ros2 param get /active_perception_node nav_cmd_vel_topic
ros2 param get /active_perception_node cmd_vel_topic
ros2 param get /active_perception_node imu_topic
ros2 topic info /cmd_vel_champ --verbose
ros2 topic echo /semantic_speed_scale
ros2 topic echo /perception_mode
```

仿真闭环的正确结果应为 `/cmd_vel -> active_perception_node -> /cmd_vel_champ`。若
`/perception_mode` 显示 `STALE[...]`，表示 `/plan`、`/voxel_entropy_data` 或
`/imu/data` 已缺失/过期，节点会立即降到最低倍率；若 250 ms 未收到新的 `/cmd_vel`，
独立稳态时钟 watchdog 会发布一次零速。`/semantic_speed_scale < 1.0` 且没有 STALE 时，
速度降低来自当前路径不确定度；若最终话题只有一个发布者且倍率接近 1.0，但 Gazebo
底部 `Real Time Factor` 低于 1.0，则墙钟时间下变慢来自仿真负载。
例如 `Real Time Factor=0.70` 表示仿真只以约 70% 实时速度运行，不应把它归因于导航
算法。正式记录导航耗时时必须同时记录 Real Time Factor。

### 先验证手动导航

在 RViz 使用 `2D Goal Pose` 选择一个无障碍位置。检查：

```bash
ros2 action info /navigate_to_pose
ros2 topic hz /cmd_vel
ros2 topic hz /cmd_vel_champ
ros2 topic echo /nav_goal_bridge/status
```

如果 `/cmd_vel` 有数据而 `/cmd_vel_champ` 没有，检查 `active_perception_node`；如果两者
都有而 Go2 不动，检查 Gazebo launch 的 `cmd_vel_topic` 和 CHAMP 控制器。

### 再验证语义查询

先监听：

```bash
ros2 topic echo /query_target_pose --once
```

```bash
ros2 topic echo /goal_pose --once
```

再查询静态红车。第一次可先查询类别 `car`，确认类别链路后再测试颜色属性：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'car'}"
```

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'red car'}"
```

汽车模型中心在 Gazebo 世界约 `(6, 0)`。`/query_target_pose` 是当前可见语义表面点组成
的空间簇质心，不是 Gazebo 模型的几何中心；它应落在汽车可见车身附近。
`/goal_pose` 才是机器人实际执行的车外 road 接近点。

查询成功后应看到 `nav_goal_bridge_node` 依次输出“收到语义导航目标”和“Nav2 已接受
目标”。如果 action 尚不可用，目标会被缓存，Nav2 启动后自动发送，无需人工复制坐标。

如果首次查询没有目标，继续移动机器人补充目标视角和语义证据，然后重新发布查询。
当前节点不会自动对失败查询重试。

### 6.1 school parking lot 颜色与导航安全回归

本节用于验收 2026-08-04 完成的簇级颜色约束和 Nav2 action 桥接。Gazebo 世界为：

```text
/home/yk/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/school_parking_lot.world
```

场景中的车辆中心真值约为：红车 `(7.5, 4.5)`、蓝车 `(3.2, 4.5)`、白车
`(7.5, -4.5)`、黄车 `(2.8, -4.5)`。旧算法查询 `white truck` 时曾输出
`(7.746, 3.25)`，这是红车一侧的错误结果。

必须先停止旧 `ga_bsvm_node`、Nav2 和 Gazebo，再按本节重新启动。GA-BSVM 体素图只
存在于进程内，重新启动能避免旧错误颜色证据继续参与本轮验收。终端 1 使用下面的世界，
其余终端继续使用本章终端 2、方案 A 终端 3/4 和终端 5 的命令。

停车场四辆车的 Gazebo 模型默认位于 `~/.gazebo/models`。启动前先确认资源完整：

```bash
for color in red blue white yellow; do
  test -f ~/.gazebo/models/semantic_sedan_${color}/model.sdf || \
    echo "missing semantic_sedan_${color}"
done
```

没有输出才表示四个模型都存在。

然后按第 2 节逐个关闭旧 `gzclient`/`gzserver` 并确认 `11345` 端口为空，再启动：

```bash
ros2 launch go2_config gazebo.launch.py \
  world:=/home/yk/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/school_parking_lot.world \
  world_init_x:=0.0 \
  world_init_y:=0.0 \
  world_init_z:=0.35 \
  cmd_vel_topic:=/cmd_vel_champ \
  gui:=true \
  rviz:=false \
  use_sim_time:=true
```

如果 GUI 仍显示旧的绿色简易世界，不要继续测试，也不要只反复打开 GUI。先停止上面的
完整 launch，直接验证世界文件：

```bash
gzserver --verbose \
  /home/yk/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/school_parking_lot.world
```

成功标志是：

```text
Loading world file [.../school_parking_lot.world]
```

需要查看画面时，在第二个终端运行 `gzclient`。验证结束后先关闭 `gzclient`，再在
`gzserver` 终端按 `Ctrl+C`，确认 `11345` 已释放后才能运行 ROS launch。手动
`gzserver` 与 `ros2 launch go2_config gazebo.launch.py` 不能同时运行，否则后者会因
`Address already in use` 启动失败，并可能让旧 GUI 看起来像是又加载了旧地图。

发布查询前检查感知、Nav2 和桥接节点：

```bash
ros2 node list | grep -E "nav_goal_bridge|bt_navigator|planner_server|controller_server"
ros2 action info /navigate_to_pose
ros2 topic hz /segformer/class_mask
ros2 topic hz /segformer/project_posterior
ros2 topic hz /semantic_cloud
```

分别在两个终端提前监听观测目标和接近点；这两个话题只在查询成功时发布一次，查询后
才启动 `--once` 可能收不到本次消息：

```bash
ros2 topic echo /query_target_pose --once
```

```bash
ros2 topic echo /goal_pose --once
```

然后发布：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'white truck'}"
```

本轮验收标准：

1. GA-BSVM 日志解析为 `class=truck, color=white`；
2. 被接受簇的 `color_support` 不低于配置值 30%；
3. `/query_target_pose` 位于白车的可见车身范围附近，而不是旧结果所对应的红车附近；
4. `nav_goal_bridge_node` 输出 action 已接受，随后 `/cmd_vel` 和
   `/cmd_vel_champ` 均有数据；
5. 若白车证据不足，系统应拒绝发布目标，不能退化为选择红车。

簇级颜色修复目前已通过自动测试，但在上述流程完成前仍记为“代码已实现、Gazebo
未验收”。

随后完全停止当前导航目标，将机器狗置于黄车右上侧空地（历史复现起点约为
`(5.03, -2.40)`），重新监听两个 Pose 话题并发布：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'yellow truck'}"
```

历史故障记录为：目标表面簇约 `(1.86, -4.29)`，错误接近点
`(-0.15, -4.85)`，位于黄车另一侧，迫使 Nav2 沿车身绕行。修复后的附加验收标准：

1. GA-BSVM 输出 `机器人同侧=True`；优先应同时输出 `直线路径净空=True`；
2. 若同侧没有满足距离和净空要求的 road 体素，系统拒绝发布 `/goal_pose`，不能再次
   使用 `(-0.15, -4.85)` 一类远侧点，也不能退回车辆中心；
3. RViz 中 `/plan` 不穿过车辆膨胀区，`local_costmap/published_footprint` 覆盖机身和
   腿部扫掠范围；
4. `/semantic_speed_scale` 降低时，线速度和角速度同比缩放，机器人不再以改变后的
   曲率切向贴车；
5. 控制日志不能再出现连续 `Failed to make progress` 或
   `No valid trajectories out of 419`；action 必须明确成功，否则本轮记为失败。

这里不能把 `/query_target_pose` 当作黄车几何中心 `(2.8, -4.5)` 直接比较。黄车模型
尺寸约为 `4.40 x 1.82 m`，在 world 中旋转 90 度后，车身水平范围约为
`x=[1.89, 3.71]`、`y=[-6.70, -2.30]`。2026-08-10 的一次实际查询得到：

```text
/query_target_pose = (3.354, -2.390)
/goal_pose         = (5.350, -2.550)
```

前者位于黄车可见的前右车角；后者与前者相距约 `2.00 m`，符合
`query_approach_distance_m: 2.0`，目标朝向也指回黄车。这组结果说明“识别到可见车身
表面，并在其外侧选择接近点”，不是导航到错误车辆。当前接近点选择是在连续 road 体素
中完成的，不会强制输出离散的“车上方/下方/左侧/右侧”；斜前方安全点属于允许结果。

历史 TF 晚到现在由有界队列短暂等待。日志出现“进入有界重试队列”本身不是失败；若
持续出现“等待历史TF超时，已安全丢弃”，仍需检查 FAST-LIO 延迟、单一 TF 权威源和
系统负载，不能用最新 TF 替代点云时刻 TF。

## 7. Gazebo person benchmark

静态 person 测试使用：

```text
/home/yk/ws/src/unitree-go2-ros2/unitree_go2_description/worlds/outdoor_semantic_benchmark.world
```

将终端 1 的 `world` 替换为该文件，然后查询：

```bash
ros2 topic pub --once /text_query std_msgs/msg/String "{data: 'person'}"
```

不要用 `outdoor_terrain.world` 中移动 actor 的历史点云做定位精度评价。普通 Gazebo
actor 对 ray sensor 没有与视觉同步的碰撞代理。当前体素图已有连续时间衰减和动态 TTL，
但仍没有射线自由空间清除、动态实例关联与跟踪，不能据此评价移动目标定位精度。

## 8. Lite3 实机传感器采集与上机前清单

Lite3 机载 Jetson 使用 Ubuntu 20.04、ROS 2 Foxy 和厂商工作区。本节命令全部在机器狗
SSH 终端执行，不能套用本手册第 1 节的 Humble 环境。传感器数据和 Bag 都保存在机器狗
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
`control_transfer ... error 11` 可以记录为警告。若它持续刷屏并伴随图像停流，才按
8.6 节处理。

### 8.4 终端 3：三路并发录制和自动验收

先从开发电脑把两个独立脚本复制到机器狗：

```bash
ssh ysc@192.168.1.103 'mkdir -p ~/lite3_tools'

scp \
  ~/ws/src/semantic_mapping/scripts/record_lite3_sensors.sh \
  ~/ws/src/semantic_mapping/scripts/verify_lite3_capture.py \
  ~/ws/src/semantic_mapping/scripts/audit_lite3_timestamps.py \
  ysc@192.168.1.103:~/lite3_tools/
```

然后在第三个机器狗 SSH 终端执行：

```bash
bash ~/lite3_tools/record_lite3_sensors.sh
```

脚本会：

1. 确认 `/cmd_vel` 没有发布者，且 D435I 仍在 USB 总线；
2. 检查四个主话题恰好各有一个发布者；
3. 在 15 秒内实际接收并验证一条 IMU、点云、图像和 CameraInfo，避免 endpoint
   存在但数据为零；
4. 用三个独立的 Foxy `ros2 bag record` 同时录制 70 秒，避免原始 RGB 阻塞
   LiDAR/IMU recorder；
5. 使用可靠、深度 1000 的 IMU 订阅 QoS；
6. Image/CameraInfo 使用相机发布者提供的默认 QoS，并保持
   `--max-cache-size 0` 直接写盘。本机 Foxy rosbag2 0.3.11 使用非零缓存后曾出现
   metadata 有计数、SQLite 实际零消息；可靠深度 100 又会积压数秒的旧图像；
7. 检查 SQLite 完整性、消息类型、频率、相机计数差和四话题共同时间窗；
8. 从精确录制开始时间检查内核 USB 错误。

脚本结束时会提示输入 `sudo` 密码读取内核日志。权限失败必须记为
`USB_KERNEL_CHECK=UNKNOWN`，不能当作通过。结果目录写入：

```text
~/lite3_bags/lite3_concurrent_时间_随机后缀/
```

其中包括三个 Bag、recorder 日志、`capture.env`、`manifest.txt`、
`validation.txt/json`、录制后 USB 枚举和录制期间内核日志。最近一次目录也保存为：

```bash
source ~/lite3_current_run.env
echo "$RUN_DIR"
```

已有 Bag 可以在任何带 Python 3 的电脑离线复验，不需要 ROS：

```bash
python3 scripts/verify_lite3_capture.py \
  /路径/lite3_concurrent_时间_后缀
```

如果接收间隔出现 `WARN`，或准备进入运动测试，必须在已安装 Foxy 和 Livox 消息定义的
机器狗上进一步审计消息自身时间戳：

```bash
source /opt/ros/foxy/setup.bash
source ~/lite_cog_ros2/driver/mid360_ws/install/setup.bash
source ~/lite3_current_run.env

python3 -u \
  ~/lite3_tools/audit_lite3_timestamps.py \
  "$RUN_DIR" |
  tee "$RUN_DIR/timestamp_audit.txt"
```

该工具只读 Bag，分别报告 rosbag 接收时间与 `header.stamp` 的最大间隔、LiDAR
`timebase - header.stamp`，以及 LiDAR 到最近相机/IMU 消息的时间差。

### 8.5 通过标准

- 三个 recorder 自然超时退出码均为 `124`；
- IMU 为 `190–210 Hz`，LiDAR 为 `9–11 Hz`；
- 图像和 CameraInfo 均为 `8–16 Hz`；
- rosbag 最大接收间隔：IMU 不超过 0.1 秒、其他三路不超过 0.5 秒为正常；
  IMU 在 0.1–1.0 秒、其他三路在 0.5–1.0 秒之间只记为接收调度 `WARN`；
  任一话题超过 1.0 秒才在基础验收中判失败，防止长时间停流或严重积压被平均频率
  掩盖；
- 图像与 CameraInfo 消息数差不超过较大值的 2%；
- 四个主话题的共同 Bag 接收时间窗不少于 60 秒；
- D435I 录制前后都能由 `lsusb -d 8086:0b3a` 枚举；
- SQLite `PRAGMA quick_check` 通过；
- 录制期间没有 `HC died`、xHCI 不响应、`error -110`、USB 断开或设备初始化失败。

自动验收统计的是 rosbag 接收时间。短时接收间隔可能来自系统调度、DDS 排队或磁盘
写入竞争，不等于传感器 `header.stamp` 真正断流。出现接收间隔 `WARN` 时，基础
`PASS` 只能记为 `PASS_WITH_RECEIPT_WARN`，必须继续运行
`audit_lite3_timestamps.py`。严格审计直接要求 IMU、LiDAR、Image 和 CameraInfo 的
header 最大间隔分别不超过 0.05、0.20、0.25 和 0.25 秒，并继续检查跨传感器同步。
正式融合实验还需检查内外参和 TF。

### 8.6 USB 警告、停止和重启条件

- 偶发 `control_transfer ... error 11`，且图像至少 8 Hz、内核无错误：只记
  `WARN`，不重启；
- 相机 Bag 为零或低于 8 Hz，但 `lsusb` 仍能看到 D435I、内核无 xHCI 致命错误：
  先只停止并重启终端 2 的 RealSense 节点，再复测一次；
- D435I 从 USB 消失，或出现 `xHCI host controller not responding`、`HC died`：
  停止采集并重启机器狗；
- 整机重启后仍复现：优先检查 D435I 线缆、内部 Hub、接口松动和供电，不要反复重启。

`/tf` 为 0 条、`timeout` 返回 124、Python `ros2 topic hz` 对大点云低估，都不是重启
条件。

### 8.7 进入运动测试前仍需完成

仿真配置到实机的主要对应关系是：

| 仿真 | Lite3 实机候选 | 说明 |
| --- | --- | --- |
| `semantic_mapping_sim_livox.yaml` | `semantic_mapping_lite3_real.yaml` | 语义融合、查询和主动感知参数 |
| `fast_lio/config/sim_mid360.yaml` | `semantic_mapping/config/fast_lio_lite3_real.yaml` | 受控运动数据的定位/建图候选 |

`fast_lio_lite3_offline.yaml` 不能替代第二行。它只服务静止烟测，关闭了运动导航所需的
几何输出和机体系点云。

当前实机配置有四层失败关闭保护：

1. GA-BSVM 必须收到与 `/camera/color/image_raw` 同模式的
   `/camera/color/camera_info` 才投影；内参会按实际图像尺寸缩放；
2. 当前投影代码不校正畸变。若 CameraInfo 含非零畸变而输入不是校正图，节点会拒绝
   投影；
3. `projection_calibration_verified: false` 时可以观察调试语义图，但不会发布
   `/query_target_pose` 或 `/goal_pose`；
4. 主动感知只输出 `/cmd_vel_lite3_safe`。在云深处 SDK 安全桥未实现时该话题没有
   执行端，机器人不会因误启动算法节点而运动。

当前地图和速度门控还固定了以下安全边界：证据按实际经过时间连续衰减，未再次击中的
体素也会由周期剪枝老化；颜色、特征和总观测权重均有上限。动态类别 10 秒未重新观测
后清理，其他体素受 300 秒 TTL、30 米半径和 250000 数量上限约束；语义 costmap 仅投影相对
`base_link` 高度 `[-0.6, 1.0] m` 的体素；遗留 `/map` 发布关闭；主动感知从 13 类自动
计算 `h_max`，并在使用 `/plan`、不确定度点云和 IMU 前检查消息时间。当前尚无 LiDAR
射线自由空间清除，动态遮挡环境仍需额外验收。参数可根据实测调整，但不得关闭高度
过滤、地图上限、输入新鲜度或 frame 失败关闭来绕过故障。

不得通过关闭上述门控或降低 GA-BSVM 阈值跳过标定。解锁运动前必须按顺序完成：

1. 保存完整 TF 树，核对点云 `header.frame_id`，并确认只有一个定位源发布
   `odom -> base_link`；
2. 核对 424x240 图像与 CameraInfo 的时间戳、frame、K、D。若 D 非零，改用对应的
   rectified 图像/CameraInfo，不能把 D 手工改成零；
3. 完成 `lidar frame -> camera color optical frame` 外参标定，替换
   `lidar_to_camera_translation/quaternion`，再用多距离、多方位标志物检查投影叠加；
4. 使用受控运动 Bag 验证 `fast_lio_lite3_real.yaml` 的轨迹、漂移、
   `/cloud_registered_body`、历史 TF 命中率和 MID360 内部 IMU 外参收敛；
5. 在 Humble 开发电脑/服务器运行 Nav2 时用 `odom_topic:=/Odometry` 对接 FAST-LIO，
   并再次确认 TF 只有一个权威源；
6. 按云深处官方 SDK 实现唯一运动桥：订阅 `/cmd_vel_lite3_safe`，完成限速、姿态/模式
   检查、急停、命令超时自动零速和网络断开本地停车；
7. 完成以上验收后才把 `projection_calibration_verified` 改为 `true`，然后依次进行
   架空、低速空场和障碍环境测试。

在 Humble 开发电脑上做“不下发底盘命令”的结构检查时，可使用：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch fast_lio mapping.launch.py \
  config_path:=~/ws/src/semantic_mapping/config \
  config_file:=fast_lio_lite3_real.yaml \
  use_sim_time:=false \
  rviz:=true
```

另一个终端只启动 Nav2 结构检查，并关闭语义目标 action 桥：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch semantic_mapping nav_lite3_real.launch.py
```

该入口默认 `use_sim_time=false`、`odom_topic=/Odometry`，并保持
`goal_bridge_enabled=false`。上述命令不等于实机运动授权。机器人机载环境是 Foxy；若 Nav2/算法运行在外部 Humble
电脑，必须把 DDS 网络、消息类型和云深处 SDK 本地安全桥作为独立接口验收。

若把算法放在外部服务器运行，机器人和服务器之间才需要稳定可路由网络、一致的
`ROS_DOMAIN_ID` 和兼容 DDS。此时必须测量相机/点云带宽、往返延迟、丢包率和控制命令
超时；网络中断时机器人必须在本地自动停车，不能依赖服务器继续发送零速。

### 8.8 开发电脑离线结构烟测

三路并发 Bag 和消息时间戳通过传感器验收后，先在开发电脑运行静止离线烟测，不要
直接启动实机导航。该流程只通过 SSH 复制 Bag；所有 ROS 2 节点都限制在电脑本机的
独立 DDS domain，不启动 Nav2、主动感知节点、运动桥，也不发布 `/cmd_vel`。

开发电脑需要 ROS 2 Humble、当前 `~/ws` overlay、`fast_lio`、
`livox_ros_driver2`、本地 CLIP 权重和至少 8 GiB 可用空间。机器狗不需要访问互联网。
在项目根目录执行：

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
cd ~/ws/src/semantic_mapping

bash scripts/run_lite3_offline_smoke.sh \
  --input \
  /home/yk/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7
```

上述目录是 2026-08-21 当前推荐的电脑原始归档。若只在机器狗上保留了数据，也可改用
`ysc@192.168.1.103:/home/ysc/lite3_bags/绝对目录`，脚本会通过 SSH 复制。

默认使用 CPU CLIP、`ROS_DOMAIN_ID=42`、`ROS_LOCALHOST_ONLY=1`、`0.10x`
回放和查询词 `road`。约 67 秒的 Bag 需要约 11 分钟完成回放。只准备数据、不启动
任何 ROS 节点时使用：

```bash
bash scripts/run_lite3_offline_smoke.sh \
  --input ysc@192.168.1.103:/绝对路径/lite3_concurrent_目录 \
  --prepare-only
```

脚本会保留原始 Bag，检查 SHA256 和 SQLite，使用一个 Humble
`ros2 bag convert` 进程把 IMU、LiDAR、Image 和 CameraInfo 合并成单一 Bag。
原 Bag 中的 `/tf`、`/tf_static` 不参与回放，避免与 FAST-LIO 的动态
`odom -> base_link` 冲突；`/tf_static` 只做结构审计。随后脚本按以下顺序运行：

1. 加载 `fast_lio_lite3_offline.yaml`，启动唯一 FAST-LIO；
2. 启动 CPU CLIP 和 GA-BSVM；
3. 在算法节点就绪后启动结果 recorder；
4. 用唯一 player 发布四路传感器和 `/clock`；
5. 语义图出现后发布一次文本查询；
6. 验收输出并只清理本次脚本创建的进程组。

结果保存在：

```text
~/lite3_offline_runs/lite3_clip_smoke_UTC时间_随机后缀/
```

其中包含原始数据副本或本地源路径、合并 Bag、输出 Bag、运行时参数、文件哈希、
Git/ROS 环境、每个进程的独立日志和 JSON/TXT 验收报告。状态含义为：

- `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`：FAST-LIO、CLIP 和 GA-BSVM 软件链结构通过；
- `FAIL`：输入、进程、输出或静止漂移门槛失败；
- `ABORTED`：用户中断，退出码为 130；
- `PREPARED`：`--prepare-only` 成功。

即使得到 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`，也只证明离线软件链能处理这份静止
数据。当前 LiDAR-相机外参仍未标定验收，CLIP 网格消息没有源图 Header；GA-BSVM 已按
点云时刻查询历史 TF 并使用有界重试队列，但该机制仍需受控运动 Bag 验证真实 frame、
命中率和掉帧统计。因此该结果不能评价语义地图几何精度，不能授权机器狗行走。

### 8.9 离线复现与迁移备份清单

2026-08-18 推荐 Bag 已复制到电脑，2026-08-21 完整烟测已经在干净 `d27c103` 上通过。
迁移到另一系统盘、重建 overlay 或升级依赖后，不必先重复采集数据；应先检查当前推荐
运行目录及其 manifest、源路径、输出状态和哈希账本：

```bash
RUN_DIR=/home/yk/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN

test "$(cat "$RUN_DIR/OVERALL")" = ALGORITHM_STATIC_PASS_NON_GEOMETRIC
test -f "$RUN_DIR/manifest.json"
test -f "$RUN_DIR/source_sha256.txt"
test -f "$RUN_DIR/implementation_sha256.txt"

du -sh "$RUN_DIR"
ros2 bag info "$RUN_DIR/merged"
```

只有在代码、依赖或工作区发生变化而需要验证重建环境时，才重新运行完整烟测。使用本地
输入时脚本只引用原始目录，不会创建 `$RUN_DIR/raw`；继续使用同一个不可变原始路径：

```bash
bash scripts/run_lite3_offline_smoke.sh \
  --input \
  /home/yk/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7

source ~/lite3_offline_runs/lite3_offline_current_run.env
cat "$RUN_DIR/OVERALL"
```

预期结果仍为 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`。如果重建环境后的完整算法烟测
失败，必须检查与 2026-08-21 推荐运行的 manifest、实现文件和依赖差异，并至少确认
`logs/model_preflight.log` 中出现
`CLIP_OFFLINE_PREFLIGHT=PASS`，并检查 `model_preflight.json` 已记录模型权重实际
路径、大小和 SHA256；在出发前解决缺少的依赖或权重。

必须留在电脑并再备份一份的内容：

1. 必须保留不可变原始 Bag 和至少一个绑定其源路径/哈希的完整成功烟测目录。完整烟测
   目录包含 `merged/`、`output_bag/`、`logs/`、`generated/`、`OVERALL`、manifest、图
   状态、哈希和全部验收报告。只有当远程 `--prepare-only` 目录中的 `raw/` 是电脑上唯一
   原始副本时，该准备目录才必须额外保留；当前推荐原始 Bag 位于 `~/lite3_bags`；
2. `~/ws/src/semantic_mapping`、`~/ws/src/fast_lio` 和
   `~/ws/src/livox_ros_driver2` 的完整源码；工作区未提交文件不能只靠 Git HEAD；
3. 当前电脑的 `~/ws/install`，不要在出差前执行清理；
4. CLIP 权重目录
   `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai`；
5. 机器狗独有的 `lite_cog_ros2` 源码与参数快照，尤其是 timefix 适配、实际
   MID360 JSON、RealSense launch、静态 TF、启动脚本，以及
   `transfer_ros2`/`realsense_ros2` systemd unit；
6. `~/Desktop/version_log.txt`、ROS/Foxy 包版本、Lite3 产品手册、厂商接口资料和
   这次运行日志；
7. 硬件证据：D435I 序列号/固件、`rs-enumerate-devices`、`lsusb -t`、Mid360
   序列号/IP/固件，以及传感器安装方向、相对位置的照片和尺量记录。Bag 无法还原
   设备身份或机械安装。

建议把版本快照也写入运行目录：

```bash
python3 -m pip freeze > "$RUN_DIR/pip-freeze.txt"
dpkg-query -W 'ros-humble-*' > "$RUN_DIR/ros-humble-packages.txt"
uname -a > "$RUN_DIR/uname.txt"
git -C ~/ws/src/semantic_mapping status --short \
  > "$RUN_DIR/semantic-mapping-git-status.txt"
git -C ~/ws/src/semantic_mapping diff \
  > "$RUN_DIR/semantic-mapping-working-tree.patch"
```

对 `fast_lio` 和 `livox_ros_driver2` 也保存 HEAD、status 和 diff。`git diff`
不包含未跟踪文件，因此还要把三个源码目录做文件级备份，或者在确认改动范围后提交到
自己的仓库。给完整 `$RUN_DIR`、源码归档和 CLIP 模型归档生成 SHA256；复制到移动
硬盘或另一块磁盘后执行 `sha256sum -c` 验证。不能只保留电脑内的一份。

没有机器狗时可以继续做：离线 FAST-LIO/CLIP/GA-BSVM 回放、自动验收、查询逻辑、
合成故障 Bag、性能统计和文档整理。不能完成：D435I/Mid360 USB 稳定性复测、真实静态
TF/外参测量、底盘控制桥、急停、实机 Nav2 和任何运动安全验收。

## 9. 常见故障定位

### 没有 `/semantic_cloud`

按顺序检查：

```bash
ros2 node list
ros2 topic hz /相机话题
ros2 topic hz /点云话题
ros2 topic hz /segformer/class_mask
ros2 topic hz /segformer/confidence
ros2 topic hz /segformer/project_posterior
ros2 run tf2_ros tf2_echo odom base_link
```

还要检查 GA-BSVM 日志中的“投影并融合 N/M 个3D点”。如果 N 长期为 0，优先检查
相机内参、LiDAR-相机外参、图像尺寸、时间戳和同步 QoS。

SegFormer 配置默认 `segformer_use_full_posterior: true`。正常启动时 GA-BSVM 应记录
`full project posterior ... (FP16 grid)`。只有复现实验中的旧硬标签基线时才设置
`segformer_use_full_posterior:=false`；该回退会恢复 mask+最大置信度人工重构分布，不能
作为概率保真方法的正式结果。

### `/uncertainty_cloud` 大面积红色

红色表示体素证据少或类别分布熵高。启动初期、相机视野外、投影稀疏、运动较快、标定
不准或模型置信度低都会造成高不确定性，不能只根据红色多少判断 SegFormer 好坏。

### 发布查询但没有目标

检查 GA-BSVM 日志中的：

- 候选体素数量；
- 类别阈值过滤数量；
- 低颜色分数体素、颜色拒绝簇、簇颜色均值和 `color_support`；
- `最高观测证据概率` 与 `最高原始后验`；
- `查询类为主类别体素`、`通过类别门控` 和 `主类别Top3`；
- 聚类体素数和证据；
- 是否找到满足距离、净空和 road 约束的接近点。
- 障碍类目标是否输出 `机器人同侧=True`，以及 `直线路径净空` 状态。

`最高观测证据概率` 高于阈值但 `通过类别门控=0`，通常表示查询类别不是这些体素的
主类别；`通过类别门控>0` 仍无目标，继续检查聚类数量、目标尺寸和颜色约束。若查询类
为主类别体素始终为 0，则先查看 SegFormer 帧统计，再检查相机-LiDAR 外参与同步。

M2DGR Bag 的相机视野内有效 LiDAR 投影较稀疏，因此该配置允许 2 个相邻体素组成目标
簇；Gazebo 与 Lite3 配置仍要求至少 3 个体素。这个差异只用于数据集感知验证，不能直接
当作实机参数。

Gazebo 与 Lite3 配置要求机器人位姿可用并存在同侧安全 road 体素。找不到同侧点时
不发布目标是新的安全拒绝行为，不应重新打开 `query_require_safe_approach=false` 规避。

SegFormer 只能查询当前 13 类及其别名，且实际可识别类别取决于启动时打印的
`checkpoint supported navigation classes`。默认 Cityscapes 权重不支持
`electric_bicycle`；仅在词表里加别名不会创造该识别能力。任意开放词汇应改用
CLIP，或者换用含目标标签的 SegFormer 微调权重。

### 有 `/goal_pose` 但机器人不动

检查 `nav_goal_bridge_node` 是否存在、Nav2 action 是否就绪，以及底盘速度链路：

```bash
ros2 node list | grep nav_goal_bridge_node
ros2 action info /navigate_to_pose
ros2 topic hz /cmd_vel
ros2 topic hz /cmd_vel_champ
```

若桥接节点提示 action 不可用，说明 Nav2 没有启动或生命周期节点未激活；若 action
已接受但没有 `/cmd_vel`，检查规划器和代价地图；若有 `/cmd_vel` 但没有
`/cmd_vel_champ`，检查主动感知节点和 CHAMP 接口。

### Go2 腿不动但机身平移

通常表示速度命令没有经过 CHAMP 步态控制器，或者 Gazebo 模型被其他插件/状态发布器
直接改变位姿。检查 `/cmd_vel_champ` 的唯一发布者、Gazebo launch 的
`cmd_vel_topic`、控制器状态以及是否重复启动机器人。

## 10. 实验记录建议

每次正式实验至少记录：

```bash
ros2 bag record \
  /Odometry /odom /tf /tf_static \
  /cloud_registered /cloud_registered_body \
  /semantic_cloud /uncertainty_cloud /voxel_entropy_data \
  /semantic_cost_map /query_target_pose /goal_pose \
  /plan /cmd_vel /cmd_vel_champ /path_entropy /semantic_speed_scale
```

同时记录：Git commit、YAML 文件、模型版本、世界文件、查询文本、目标真值、是否成功、
耗时、路径长度、最终距离和碰撞情况。没有这些元数据的截图不能作为可重复的论文结果。

## 11. CARLA 基准测试入口

CARLA 相关代码位于 `semantic_mapping/carla/`，四个 ROS 入口如下：

| 入口 | 用途 | 详细文档 |
| --- | --- | --- |
| `carla_capture_benchmark` | 采集二维图像识别基准（RGB/语义/实例相机） | `CARLA_IMAGE_BENCHMARK.md` |
| `carla_evaluate_benchmark` | CLIP/SegFormer 图像分类与颜色评测 | `CARLA_IMAGE_BENCHMARK.md` |
| `carla_capture_reliability` | 采集三维点级可靠性数据（普通/语义 LiDAR + IMU） | `CARLA_RELIABILITY_BENCHMARK.md` |
| `carla_evaluate_reliability` | 五因素可靠性、时间偏移、消融与 VoxelMap 评测 | `CARLA_RELIABILITY_BENCHMARK.md` |

截至 2026-08-13，Motion V2 的 CARLA 实验记录、报告和体素验收结论见
[CARLA Motion V2 实验记录](results/carla_motion_v2_20260813/README.md)。
该实验只验证时间偏移风险的响应方向；体素 uncertainty 主指标未通过，
Motion V2 参数未冻结，也没有接入 `ga_bsvm_node.py` 或实机运行时权重。

每组代码都有不需要 CARLA 服务器和模型权重的离线冒烟测试，先跑一遍确认安装和分层
改动没有破坏入口：

```bash
cd ~/ws/src/semantic_mapping
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test/test_carla_benchmark.py \
  test/test_carla_reliability.py \
  test/test_carla_reliability_capture.py \
  test/test_carla_evaluate_reliability.py \
  test/test_reliability_factors.py \
  test/test_reliability_voxel_ablation.py \
  -q
```

确认参数解析正常：

```bash
ros2 run semantic_mapping carla_capture_benchmark --help
ros2 run semantic_mapping carla_evaluate_benchmark --help
ros2 run semantic_mapping carla_capture_reliability --help
ros2 run semantic_mapping carla_evaluate_reliability --help
```

再启动 CARLA server（另一终端），并先验证连接：

```bash
cd /home/yk/ws/third_party/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000

python3 -c "
import carla
c = carla.Client('127.0.0.1', 2000)
c.set_timeout(10)
print('CARLA:', c.get_server_version())
"
```

### 11.1 图像识别基准

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash

RUN_DIR="/home/yk/ws/carla_benchmark_data/town05_v2_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN_DIR" > /home/yk/ws/carla_benchmark_data/LATEST_DATASET.txt

ros2 run semantic_mapping carla_capture_benchmark \
  --host 127.0.0.1 \
  --port 2000 \
  --map Town05 \
  --output "$RUN_DIR" \
  --cache-dir /home/yk/ws/.cache/carla \
  --frames 30 \
  --save-every 1 \
  --vehicles 12 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --width 640 \
  --height 480 \
  --grid-rows 4 \
  --grid-cols 6 \
  --seed 42 \
  --stationary-ego

export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

ros2 run semantic_mapping carla_evaluate_benchmark \
  --dataset "$RUN_DIR" \
  --backend segformer \
  --device cuda \
  --fp16 \
  --save-overlay-every 1
```

### 11.2 三维可靠性基准

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash

RUN_DIR="/home/yk/ws/carla_benchmark_data/reliability_town05_seed42_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN_DIR" > /home/yk/ws/carla_benchmark_data/LATEST_RELIABILITY_DATASET.txt

ros2 run semantic_mapping carla_capture_reliability \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-manager-port 8000 \
  --town Town05 \
  --weather ClearNoon \
  --seed 42 \
  --num-frames 30 \
  --warmup-frames 10 \
  --output-dir "$RUN_DIR" \
  --cache-dir /home/yk/ws/.cache/carla \
  --fixed-delta 0.01 \
  --camera-tick 0.01 \
  --lidar-tick 0.05 \
  --width 640 \
  --height 480 \
  --fov 90 \
  --lidar-channels 64 \
  --lidar-range 50 \
  --vehicles 12 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --stationary-ego \
  --no-moving-targets

export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

ros2 run semantic_mapping carla_evaluate_reliability \
  --dataset "$RUN_DIR" \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  --time-offset-ms 0 20 50 100 150 \
  --device cuda \
  --fp16 \
  --voxel-eval
```

以上先用 30 帧冒烟；正式实验按两个 CARLA 文档里的完整参数（帧数、seed 矩阵、
LiDAR 密度/运动剖面、输出目录归档）执行。两组 benchmark 的数据格式互不兼容，
采集命令和数据目录不要混用。
