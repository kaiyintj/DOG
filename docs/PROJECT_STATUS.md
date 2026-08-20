# 当前版本状态

更新日期：2026-08-13

本文描述 `~/ws` 当前工作区中的实际代码。后续讨论、实验和新对话应以本文、
`RUNBOOK.md` 及当前源码为准，不再以旧聊天记录中的命令为准。

## 0. 新对话交接摘要

新对话开始时，先阅读本文、[RUNBOOK.md](RUNBOOK.md) 和项目根目录的
[README.md](../README.md)。当前最重要的交接信息如下：

若新对话的目标是继续 Lite3 实机迁移，还必须先阅读专用的
[LITE3_REAL_HANDOFF.md](LITE3_REAL_HANDOFF.md)。其中冻结了当前最佳 Bag、电脑归档路径、
下一条离线烟测命令和实机运动前门禁，优先级高于旧聊天中的临时命令。

- 代码已按用途分层：`semantic_mapping/runtime/` 放实际运行代码及其依赖的共用核心，
  `semantic_mapping/carla/` 只放 CARLA 仿真采集/评测代码；运行时目录不依赖 carla 目录，
  `setup.py` 的 ROS 入口已同步指向新路径；
- 主路线是 Cityscapes `SegFormer + GA-BSVM`；`CLIP + GA-BSVM` 是开放词汇对比
  路线，同一次实验只启动其中一个语义前端；
- 当前统一 13 类：road、building、tree、person、car、truck、bus、bicycle、
  electric_bicycle、motorcycle、chair、bench、unknown background；
- 闭集 SegFormer 导航架构已预留 `electric_bicycle`，查询别名可区分自行车、
  电动自行车、摩托车和汽车；当前 Cityscapes 权重不含电动自行车输出，
  必须换用带对应 `id2label` 的微调 SegFormer 才能真正识别；
- 已新增成对 RGB/单通道掩码数据校验、13 类分类头迁移初始化、纯 PyTorch
  微调、权重标签验收和运行时 IoU 失败关闭门禁；本地暂无电动自行车
  像素标注，因此尚未生成可用于真实导航的电动车权重；
- SegFormer 主链已完成 P0 概率保真改造：原始 checkpoint 概率先按项目类别求和，再
  进行硬判决；节点发布带源图 Header 的原生解码分辨率 `16FC13` FP16 后验，GA-BSVM
  默认直接采样完整后验。旧 mask+最大置信度链只保留为显式回归基线；
- 图像入口已统一按 RGB 处理，OpenCLIP `openai` 权重会自动选用 QuickGELU 模型；
  M2DGR、Gazebo 与 CARLA 基准的 CLIP 网格已按 4x6 对齐；Lite3 配置仍为 3x4，
  正式做跨平台对比前需要统一；
- 带颜色查询已改为“先按类别形成完整空间簇，再计算整个簇的颜色支持率”，参数
  `query_min_color_support_ratio` 默认是 `0.3`。这用于避免白色车灯、车牌或轮毂把
  红车误判为 `white truck`；
- `nav_goal_bridge_node` 已升级为“一个活动目标 + 一个最新待发送目标”的事务状态机；
  新目标会显式取消旧目标，发送异常/拒绝会保留并重试，结果去重，状态发布到
  `/nav_goal_bridge/status`；
- Nav2 已拆成两个明确入口：Gazebo 使用 `nav_sim.launch.py`（仿真时钟、`/odom`、
  仿真参数），Lite3 使用 `nav_lite3_real.launch.py`（墙钟、`/Odometry`、实机参数，
  标定/SDK 桥验收前默认不启动 Goal Bridge）。通用入口现在默认墙钟；
- 黄车回归暴露的“接近点落在车辆另一侧、沿车身擦行、主动降速改变 DWB 轨迹曲率”已完成
  代码修复：障碍类目标要求机器人同侧 road 接近点，优先直线路径净空，仿真不再使用
  未验证回退点；Nav2 改用包含腿部扫掠的多边形 footprint，主动感知同比缩放线角速度；
  自动测试已覆盖原复现坐标，仍需全新 Gazebo 进程做闭环验收；
- 2026-08-08 完成仿真到 Lite3 的配置审查：实机 GA-BSVM 现在强制读取匹配图像模式的
  `CameraInfo`，并保持 `projection_calibration_verified: false`。在标定验收前只允许生成
  调试语义图，不发布 `/query_target_pose` 或 `/goal_pose`；
- 新增 `fast_lio_lite3_real.yaml` 作为受控运动数据候选配置；原
  `fast_lio_lite3_offline.yaml` 仍只用于静止非几何烟测。Nav2 launch 新增
  `odom_topic` 参数，可用 `odom_topic:=/Odometry` 对接 FAST-LIO；
- Lite3 速度链已改为 `/cmd_vel -> active_perception_node -> /cmd_vel_lite3_safe`，避免
  主动感知与 Nav2 velocity smoother 同时发布 `/cmd_vel`。云深处 SDK 安全桥必须成为
  `/cmd_vel_lite3_safe` 的唯一消费者和底盘命令唯一写入者；
- 主动感知已按 13 类自动计算 `h_max=ln(13)`；路径、不确定度点云和 IMU 均有源时间戳
  新鲜度门控，感知和控制使用独立 callback group/多线程执行器，250 ms 稳态时钟
  watchdog 可在 ROS 仿真时钟暂停时发布零速；
- GA-BSVM 的 IMU 改用 SensorDataQoS；缺失或无对齐样本时可靠度为保守的 `0.2`，不再
  错误返回完全可信。语义证据按连续时间衰减，周期剪枝会老化未命中体素，颜色/特征/
  总观测权重采用有界累计；仍未实现射线自由空间清除；
- 语义 costmap 已增加相对 `base_link` 的 `[-0.6, 1.0] m` 高度带；动态类别使用
  10 秒 TTL，其他体素受 300 秒 TTL、30 米半径和 250000 数量上限约束；road 类按
  名称解析，遗留 `/map` 发布默认关闭；
- SegFormer 训练现在支持独立 `calibration/test`：`val` 只选模，最终阈值优先使用
  `test`，报告绑定 checkpoint SHA-256；缺少独立 test 时明确标为非正式结果；
- 本轮等价完整回归为 `184 passed, 1 skipped`，flake8/pep257 均通过；单包
  `colcon build`、两个 Nav2 入口解析和实际图片离线推理均已通过；
- Git 当前位于本地 `main`，基线 HEAD 为 `969affa`。本文新增的 Lite3 迁移修正仍在
  未提交工作区，开始新实验前不要清理或覆盖这些修改；
- 下一项明确任务是在全新进程和空 GA-BSVM 体素图中，使用
  `school_parking_lot.world` 依次复测 `white truck` 和 `yellow truck`。旧算法曾错误输出
  `(7.746, 3.25)`，落在红车附近；世界真值中红车约为 `(7.5, 4.5)`，白车约为
  `(7.5, -4.5)`。本轮簇级颜色修复已通过自动测试，但尚未完成干净 Gazebo 实测验收；
- 复测时还要确认 `nav_goal_bridge_node`、Nav2 action、`/cmd_vel` 和
  `/cmd_vel_champ` 均有闭环输出，并确认黄车目标不再落到车辆另一侧。完整步骤见
  [RUNBOOK 6.1](RUNBOOK.md#61-school-parking-lot-颜色与导航安全回归)。

## 1. 项目目标

目标是在四足机器人移动过程中完成：

1. LiDAR/IMU 在线定位与建图；
2. 相机语义与 LiDAR 点云融合，构建带不确定性的三维语义体素图；
3. 接收自然语言目标，例如 `car`、`bicycle`、`blue car`；
4. 估计目标物体位置和安全接近点；
5. 使用 Nav2 规划路径，并通过四足机器人运动接口执行；
6. 最终从 Gazebo 的 Unitree Go2 仿真迁移到云深处绝影 Lite3 激光版。

## 2. 当前结论

当前版本已经形成“传感器输入 -> 语义分割/特征 -> 三维概率融合 -> 文本查询 ->
目标位置与接近点”的可运行链路，并能发布 Nav2 使用的语义代价地图。

CLIP 与 SegFormer 是两套可替换的语义前端，不应在同一次实验中同时启动。当前建议把
`SegFormer + GA-BSVM` 作为固定类别、可通行性和导航安全的主基线，把
`CLIP + GA-BSVM` 作为开放文本检索和对比实验路线。

当前版本还不能称为完整的实机语义导航系统，主要缺口是：

- `/goal_pose` 已通过 `nav_goal_bridge_node` 转换为 Nav2 的
  `NavigateToPose` action，但仍需在每套底盘接口上做闭环验收；
- 文本查询目前只在消息到达时检查一次地图；若当时证据不足，地图后续增长不会自动
  重试，必须再次发布查询；
- Gazebo 中 CHAMP 与 FAST-LIO 都可能发布 `odom -> base_link`，正式实验前必须只保留
  一个 TF 权威源；
- Lite3 的传感器话题和录包链路已确认，运行时内参读取与失败关闭保护已实现；但
  LiDAR-相机外参、完整 TF 权威关系和底盘 SDK 安全桥尚未验收；
- 尚无实例分割、跨帧物体 ID、动态目标跟踪和三维实例图。

因此，当前成熟度应按场景区分：

| 场景 | 当前状态 |
| --- | --- |
| M2DGR Bag 感知、建图、语义融合 | 已运行验证 |
| M2DGR Bag 文本查询和目标点生成 | SegFormer `car` 已运行验证；Bag 本身不能驱动机器人 |
| Gazebo 手动 2D Goal 导航 | 已运行验证 |
| Gazebo 语义查询生成 `/goal_pose` | 已运行验证 |
| Gazebo 语言查询自动触发 Nav2 | action 桥接已实现，等待本轮 Gazebo 闭环复测 |
| Gazebo 主动感知速度调节 | 代码已接入，需定量实验验证收益 |
| Lite3 实机传感器采集 | 70 秒三路并发录制、消息时间戳和 USB 检查已通过，接收调度存在 WARN |
| Lite3 电脑离线结构烟测 | 自动准备/合并/回放/验收工具已实现，真实 Bag 端到端结果待运行 |
| Lite3 实机语义导航 | 配置已失败关闭；标定、定位 TF、运动安全桥和 Nav2 闭环未完成 |

## 3. 当前系统组成

### 3.1 定位与点云前端

- 算法：FAST-LIO；
- Bag 配置：`fast_lio/config/velodyne.yaml`；
- Gazebo 配置：`fast_lio/config/sim_mid360.yaml`；
- Lite3 静止烟测：`semantic_mapping/config/fast_lio_lite3_offline.yaml`；
- Lite3 受控运动数据候选：`semantic_mapping/config/fast_lio_lite3_real.yaml`；
- 主要输出：`/Odometry`、`/cloud_registered`、`/cloud_registered_body`、TF
  `odom -> base_link`。

注意：FAST-LIO 发布的是大写 `/Odometry`，Gazebo 中 `/odom` 通常由
CHAMP/robot_localization 提供。Nav2 配置默认仍读取 `/odom`，但
`nav_lite3_real.launch.py` 默认把 Nav2 改写到 `/Odometry`。该重写只解决消息话题，
不会自动解决重复的 `odom -> base_link` TF；实机必须先确定唯一 TF 权威源。

### 3.2 两种语义前端

每次运行只选择下列一个后端。选择 CLIP 时启动 `clip_node` 并设置
`semantic_backend:=clip`；选择 SegFormer 时启动 `segformer_node` 并设置
`semantic_backend:=segformer`。

#### CLIP 后端

- 模型：OpenCLIP `ViT-B-32/openai`；
- 将图像划分为网格，输出每个网格的类别 logits 和 512 维特征；
- 优点：支持配置词汇之外的开放文本相似度；
- 限制：空间分辨率较粗，容易把背景和目标混入同一网格。

#### SegFormer 后端

- 模型：`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`；
- 在原生 decoder 分辨率对原始类别做 softmax，并在 `argmax` 前将多对一标签概率求和到
  13 类项目本体；
- 输出带源图 Header 的 `16FC13` FP16 完整后验，同时保留历史“原始 argmax 后映射”
  硬类别、原始最大置信度、彩色预览和源图作为可视化/回归接口；
- 优点：边界和可通行区域比 CLIP 网格细；
- 限制：当前是 Cityscapes 闭集映射，适合道路、人员和交通参与者，但不能直接理解
  `chair` 等模型类别之外的任意词汇；CPU 推理较慢。

当前统一语义类别为：

1. `road`
2. `building`
3. `tree`
4. `person`
5. `car`
6. `truck`
7. `bus`
8. `bicycle`
9. `electric_bicycle`
10. `motorcycle`
11. `chair`
12. `bench`
13. `unknown background`

同时支持中英文常用别名，以及 red、orange、yellow、green、blue、purple、brown、
black、white、gray 十种基础颜色。例如：`bike`、`自行车`、`blue car`、
`红色自行车`。

### 3.3 GA-BSVM 三维融合

当前 `ga_bsvm_node` 的核心不是二值占据累计，而是可靠性加权的 categorical
Dirichlet 证据融合：

- 将相机语义投影到 LiDAR 点；
- 按 0.1 m 体素累计类别证据、CLIP 特征和观测颜色；
- 可靠性同时考虑 IMU 运动、局部点密度、距离、图像边缘位置和语义熵；
- 限制单帧证据和体素总证据，避免点数多就被错误视为高置信度；
- 发布语义点云、不确定性点云、路径熵数据和二维语义代价地图。

文本查询不是选择单个最大体素，而是进行类别/特征/颜色筛选、空间聚类、目标尺寸
约束和证据支持度评分。对障碍物类目标，系统尝试在目标附近寻找满足距离与净空要求的
`road` 体素作为接近点。

颜色属性不是先过滤单个体素再聚类，而是在类别候选形成完整空间簇后计算簇内颜色
均值和颜色支持率。簇必须达到 `query_min_color_support_ratio` 才能通过颜色约束，
因此局部白色车灯、车牌和轮毂不能单独代表整辆白车。

类别查询使用从 Dirichlet 参数中去除对称先验后的观测证据分布，并继续用完整后验计算
地图不确定性。这样 `query_min_class_prob` 不会因为类别从 6 类扩展到 13 类而系统性压低
稀有目标分数；`query_require_class_argmax` 默认还要求查询类别是该体素的观测主类别。

当前查询实现是一次性快照：`/text_query` 到达时立即筛选当时已有体素并决定是否发布
目标。若日志显示“未找到合适目标”，需要继续积累语义证据后重新发布同一查询。系统尚无
查询状态机、自动重试、超时或取消机制。

### 3.4 Nav2 与主动感知

Nav2 当前使用：

- 全局规划：NavFn；
- 局部规划：DWB；
- 局部障碍：`/cloud_registered_body` VoxelLayer；
- 全局障碍：`/cloud_registered_body` ObstacleLayer；
- 语义层：订阅 `/semantic_cost_map` 的 StaticLayer；
- Go2 footprint：`0.70 m x 0.44 m` 矩形加 `0.05 m` padding；
- 障碍膨胀半径：`0.85 m`，DWB `BaseObstacle.scale=0.2`；
- 全局坐标系：`odom`，机器人坐标系：`base_link`。

`active_perception_node` 根据 `/plan` 附近体素的融合不确定度调节速度。路径与
`/voxel_entropy_data` 会先转换到 `odom`；无法验证 frame 或 TF 时清除旧缓存并失败
关闭。路径、熵点云和 IMU 分别有最大年龄，任一必需输入缺失、过期、异常未来时间戳
或仿真时钟回跳时立即使用最低倍率。速度输入 250 ms 未更新时，稳态时钟 watchdog
发布零速。风险值组合均值和 90 分位数，速度倍率按 `dt` 和 `1.6/s` 限制。为避免
后处理改变 DWB 已经碰撞检查的转弯半径，当前默认按同一比例缩放线速度和角速度。
Gazebo 配置中的命令链为：

```text
Nav2 /cmd_vel -> active_perception_node -> /cmd_vel_champ -> CHAMP
```

Lite3 预留命令链为：

```text
Nav2 velocity_smoother /cmd_vel
  -> active_perception_node /cmd_vel_lite3_safe
  -> 云深处 SDK 安全桥（尚未实现/验收）
```

这种分离保证 Nav2 与主动感知不会竞争发布同一个 `/cmd_vel`。在 SDK 安全桥不存在时，
`/cmd_vel_lite3_safe` 没有执行端，机器人应保持不动。

进度检查已与最低 30% 速度倍率对齐为“20 秒内移动 0.15 m”，避免安全低速运动被误判
为卡死；真实障碍碰撞仍由 footprint、VoxelLayer 和 DWB 拒绝，不能靠放宽进度门限掩盖。

## 4. 主要话题接口

| 话题 | 类型 | 作用 |
| --- | --- | --- |
| `/text_query` | `std_msgs/String` | 文本查询入口 |
| `/clip_logits` | `Float32MultiArray` | CLIP 网格类别分数 |
| `/clip_features` | `Float32MultiArray` | CLIP 网格特征 |
| `/query_feature` | `Float32MultiArray` | CLIP 文本特征 |
| `/segformer/project_posterior` | `sensor_msgs/Image` (`16FC13`) | 原生解码分辨率完整项目类别后验 |
| `/segformer/class_mask` | `sensor_msgs/Image` | 13 类逐像素掩码 |
| `/segformer/confidence` | `sensor_msgs/Image` | 逐像素置信度 |
| `/segformer/color_mask` | `sensor_msgs/Image` | RViz 语义预览 |
| `/segformer/source_image` | `sensor_msgs/Image` | 与掩码同时间戳的 RGB 图像 |
| `/semantic_cloud` | `sensor_msgs/PointCloud2` | 类别着色三维语义图 |
| `/uncertainty_cloud` | `sensor_msgs/PointCloud2` | 不确定性着色点云 |
| `/voxel_entropy_data` | `sensor_msgs/PointCloud2` | 主动感知使用的体素熵 |
| `/semantic_cost_map` | `nav_msgs/OccupancyGrid` | Nav2 全局语义代价层 |
| `/query_target_pose` | `geometry_msgs/PoseStamped` | 观测到的目标表面簇估计位置 |
| `/goal_pose` | `geometry_msgs/PoseStamped` | 目标附近的安全接近点 |
| `/nav_goal_bridge/status` | `std_msgs/String` (JSON) | Nav2 目标事务状态与终态 |
| `/path_entropy` | `std_msgs/Float32` | 当前局部路径平均熵 |
| `/semantic_speed_scale` | `std_msgs/Float32` | 主动感知速度缩放比例 |

`/query_target_pose` 和 `/goal_pose` 含义不同，不应混用。前者是观测到的目标表面簇位置，
不保证等于车辆几何中心；后者是安全接近点。`nav_goal_bridge_node` 会把后者转换为 Nav2
`NavigateToPose` action 请求，并发布接收、发送、接受、取消、超时与终态事件；
是否真正移动还取决于 Nav2 生命周期、规划器、速度链路和底盘控制器均正常。

## 5. 三套配置

| 配置 | 点云 | IMU | 图像 | 用途 |
| --- | --- | --- | --- | --- |
| `semantic_mapping_m2dgr.yaml` | `/velodyne_points` PointCloud2 | `/handsfree/imu` | 压缩图像 | M2DGR Bag |
| `semantic_mapping_sim_livox.yaml` | `/livox/lidar` CustomMsg | `/imu/data` | `/d435i/image_raw` + `/d435i/camera_info` | Gazebo Go2（以运行时 publisher 为准） |
| `semantic_mapping_lite3_real.yaml` | `/timefix/lidar` CustomMsg | `/timefix/imu` | `/camera/color/image_raw` + CameraInfo | Lite3 已确认传感器接口，运动失败关闭 |

M2DGR 与 Gazebo 配置默认 `semantic_backend: clip`。运行 SegFormer 时通过命令行设置：

```bash
-p semantic_backend:=segformer
```

Lite3 配置默认 `semantic_backend: segformer`，并要求 `/camera/color/camera_info` 有效、
畸变系数为零或输入已经校正。运行时内参会覆盖 YAML 中的回退 `camera_k`。该配置中的
LiDAR-相机外参仍是占位值，因此 `projection_calibration_verified` 必须保持 `false`；
坐标系、CUDA/FP16 算法负载和完整闭环仍必须在实机上确认。

## 6. 已完成验证

截至本文更新时已完成：

- Python 语法检查通过；
- 三套 YAML 可解析，类别、代价、尺寸约束长度均为 13；
- 39 个 Lite3 采集、时间戳和离线烟测专项测试通过；
- 2026-08-11 的 SegFormer 完整后验、类别、微调、主动感知、Goal Bridge、连续时间
  体素融合和导航配置回归已纳入测试；分组运行等价完整结果为
  `184 passed, 1 skipped`，flake8/pep257 均通过。警告仅来自已知的 SciPy/NumPy
  版本范围不一致；
- `colcon build --symlink-install --packages-select semantic_mapping` 通过；
- `ros2 pkg executables semantic_mapping` 可安装 12 个入口，包含
  `nav_goal_bridge_node`、`segformer_dataset`、`segformer_finetune`、
  `segformer_checkpoint` 和 `segformer_image`；
- 标准 ROS 2 launch 能从 package share 找到配置；
- M2DGR Bag 中 FAST-LIO、CLIP、SegFormer 和语义点云有实际运行记录；
- M2DGR SegFormer 路线已成功查询 `car`：目标簇为 2 个体素、累计证据 6.0，
  `/query_target_pose` 为约 `(13.20, -2.75)`，`/goal_pose` 为约
  `(14.15, -3.15)`，水平接近距离约 1.03 m；
- Gazebo 中 Go2、Mid360、D435i、Nav2 手动目标和 car benchmark 有实际运行记录。
- GA-BSVM 已按点云消息时间查询 `odom <- pointcloud_frame`，历史 TF 暂时未到时进入
  有界重试队列；无时间戳或重试超时的帧会被丢弃，不再使用最新位姿污染地图；
- `school_parking_lot.world` 中旧版 `white truck` 曾错误选择红车局部白色附件；簇级
  颜色支持率修复及自动测试已完成，干净 Gazebo 回归尚待执行，不能提前记为实测通过；
- CARLA 0.9.16 的采集和评测工具已能生成车辆、行人和两轮车数据；二维
  CLIP/SegFormer 基准与三维可靠性基准入口均可运行。2026-08-13 已完成 stationary、
  constant-velocity、turning 三组 Motion V2 时间偏移评测并归档报告；Motion V2 的
  offset 响应方向成立，但 turning 体素 uncertainty 主验收未通过，参数未冻结，
  不得直接迁移到实机 runtime。详细结果见
  [CARLA Motion V2 实验记录](results/carla_motion_v2_20260813/README.md)；这些结果
  仍是受控 CARLA 验证，不能替代 Gazebo 和实机验证；
- 已新增独立 `carla_reliability_v1` 采集与离线评测链路：普通 LiDAR 作为算法输入，
  Semantic LiDAR 仅提供互为最近邻且覆盖率门控后的点级 GT，历史 RGB pose 可构造
  0/20/50/100/150 ms 时间偏移；五个可靠度因子共用 GA-BSVM 正式公式，并可调用
  同一 `VoxelMap` 做六组短序列消融。当前仅通过自动化与合成数据回归，尚未完成真实
  CARLA server 全规模采集和最终论文指标，不能提前宣称参数已标定；
- Lite3 Jetson 已确认 Ubuntu 20.04/ROS 2 Foxy、Mid360 和 D435I 驱动可用；
- Lite3 `/timefix/imu` 约 200 Hz、`/timefix/lidar` 约 10 Hz，D435I 原始 RGB
  实录约 9.4 Hz，CameraInfo 与图像计数基本一致；
- 已新增三个独立 recorder、实际消息预检、SQLite 自动验收、消息时间戳审计和 USB
  分级检查工具；
- 2026-07-23 完成 70 秒正式三路并发采集：四话题共同时间窗约 66.99 秒，三个
  recorder 均正常定时退出，D435I 录制前后存在且内核 USB 检查通过；三路同时出现
  0.46–0.74 秒 rosbag 接收调度告警；
- 同一 Bag 的消息时间审计确认 IMU header 约 200 Hz、LiDAR header 约 10 Hz，均单调
  且无零时间戳，LiDAR `timebase == header.stamp`；相机存在一次约 0.532 秒真实缺口，
  当前数据结论为 `SENSOR_STATIC_PASS_WITH_WARN`；
- 2026-07-26 的复测再次确认 IMU/LiDAR header 健康，但给 Image/CameraInfo 强制
  `reliable`、`depth: 100` 会在原始 RGB 写盘吞吐低于发布频率时积压旧帧，末尾相机
  header 最多落后 LiDAR 约 6.81 秒，1 ms 内 Image/CameraInfo 配对率仅 74%。录制脚本
  已恢复使用相机发布者自身 QoS，并继续使用 `--max-cache-size 0`；该 Bag 保留作队列
  积压回归样本，不能作为运动就绪证据；
- 2026-07-26 随后使用 D435I 原生 `424x240@15Hz` 模式和修正后的相机 QoS 完成
  67.5 秒四话题共同窗口采集：Image/CameraInfo 均为 1014 条、header 配对率 100%，
  LiDAR–图像最近时间差 p99 约 32.9 ms、最大约 33.3 ms，LiDAR 周围 IMU 覆盖率
  100%，严格结论为 `SENSOR_HEADER_ALIGNMENT=PASS`。rosbag 接收时间仍有
  0.52–0.69 秒调度间隔，但消息 header 连续（IMU 最大约 12.2 ms、LiDAR 最大约
  103.8 ms、相机最大约 66.8 ms），因此记录为接收调度警告，不解释为传感器断流；
  LiDAR 到相机光学坐标系的静态 TF 仍缺失，故 `CALIBRATION_STRUCTURE=NOT_READY`、
  `MOTION_READY=NO`；
- 已增加开发电脑离线结构烟测：单一合并 Bag、隔离 DDS、FAST-LIO + CPU CLIP +
  GA-BSVM 自动编排、定向清理和输出验收。工具通过测试后仍需用上述真实 Bag 运行，
  结果上限为 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`。

上述 M2DGR 坐标证明“查询 -> 聚类 -> 目标表面簇位置 -> 安全接近点”接口已经工作，但没有
物体真值，不能据此宣称定位误差达标；Bag 也没有机器人执行器，不能验证导航成功率。

当前 Python 环境仍会报告 SciPy 1.8 要求 NumPy `<1.25`、实际 NumPy 为 `1.26.4` 的
警告。仓库已新增 `requirements-runtime-common.txt`，固定为 NumPy 1.26.4 +
SciPy 1.11.4，并为 SegFormer/CLIP 分开声明模型依赖；当前全局环境未被自动修改。正式
实验前应在隔离环境安装锁定组合并保存 `pip freeze`。

## 7. 未实现或未充分验证

### P0：完成闭环前必须解决

1. 增加语义查询本身的自动重试与地图增长触发；Goal Bridge action 事务、超时和结果
   反馈已经完成，但语义查询仍是一次性快照；
2. Gazebo 和 Lite3 均只允许一个 `odom -> base_link` TF 发布者；
3. 统一 FAST-LIO `/Odometry`、Nav2 `/odom` 和机器人本体状态估计；
4. 实测 Lite3 完整 TF 树、相机内参和 LiDAR-相机外参；
5. 实现 Lite3 官方运动控制接口与 ROS `Twist` 之间的安全桥接，包括急停和限速。
6. 将 CLIP logits/features 改为携带源图 Header 的原子消息，避免融合“最新缓存”；
7. 用 Lite3 受控运动 Bag 验证现有历史 TF 查询、重试队列和掉帧统计，确认真实 frame
   ID 与 `pointcloud_frame` 配置一致。

### P1：论文实验和可靠性

1. 在 car/person/bicycle 等可测目标上建立定位误差和导航成功率指标；
2. 对 CLIP、SegFormer、SegFormer+GA-BSVM、无主动降速等方案做消融；
3. 补充 LiDAR 射线自由空间清除和动态实例跟踪；连续时间衰减、TTL 与权重封顶已完成；
4. 标定颜色阈值和类别阈值，避免把阴影、低照度和反光当成颜色属性；
5. 记录 CPU/GPU 延迟、带宽、内存和端到端控制频率。

### P2：扩展能力

- SAM/YOLO-World 等实例分割尚未接入；
- 3D 实例图、跨帧物体关联和物体持久 ID 尚未实现；
- “蓝色汽车”等属性目前依赖体素平均 RGB，不适合遮挡、粘连和多色物体；
- 地图持久化、任务恢复和多机器人/服务器通信尚未实现。

## 8. 推荐的下一阶段顺序

1. 先用正式 Lite3 静止 Bag 完成电脑端
   `ALGORITHM_STATIC_PASS_NON_GEOMETRIC` 烟测；
2. 在进入实机 reliability 融合前，先保留 Motion V2 为诊断量；验收
   Camera/LiDAR 时间戳、CameraInfo、静态 TF 和 LiDAR-相机外参后，才把
   `projection_calibration_verified` 改为 `true`；
3. 修复 CLIP 源时间戳后，使用 `fast_lio_lite3_real.yaml` 采集并验证受控低速运动 Bag；
4. 先完成已实现 Nav2 action 事务和簇级颜色查询的 Gazebo 闭环验收，再实现语义查询
   随地图增长自动重试；
5. 解决 Gazebo/Lite3 单一里程计与 TF 权威源；
6. 完成运动安全桥后，才进行架空、低速空场和障碍环境实机测试。

完整运行步骤见 [RUNBOOK.md](RUNBOOK.md)。

## 9. Git 状态说明

本文更新时，`~/ws/src/semantic_mapping` 位于本地 `main`，基线 HEAD 为 `969affa`。
2026-08-08 的 CameraInfo/投影失败关闭、Lite3 速度链、可重写 Nav2 里程计话题以及
`fast_lio_lite3_real.yaml` 仍是尚未提交的工作区修改。新实验开始前应先完成一次明确的
提交和远端备份，否则 Git HEAD 不能代表本文描述的版本。
