# 当前版本状态

更新日期：2026-08-04

本文描述 `~/ws` 当前工作区中的实际代码。后续讨论、实验和新对话应以本文、
`RUNBOOK.md` 及当前源码为准，不再以旧聊天记录中的命令为准。

## 0. 新对话交接摘要

新对话开始时，先阅读本文、[RUNBOOK.md](RUNBOOK.md) 和项目根目录的
[README.md](../README.md)。当前最重要的交接信息如下：

- 主路线是 Cityscapes `SegFormer + GA-BSVM`；`CLIP + GA-BSVM` 是开放词汇对比
  路线，同一次实验只启动其中一个语义前端；
- 当前统一 12 类：road、building、tree、person、car、truck、bus、bicycle、
  motorcycle、chair、bench、unknown background；
- 图像入口已统一按 RGB 处理，OpenCLIP `openai` 权重会自动选用 QuickGELU 模型；
  M2DGR、Gazebo 与 CARLA 基准的 CLIP 网格已按 4x6 对齐；Lite3 配置仍为 3x4，
  正式做跨平台对比前需要统一；
- 带颜色查询已改为“先按类别形成完整空间簇，再计算整个簇的颜色支持率”，参数
  `query_min_color_support_ratio` 默认是 `0.3`。这用于避免白色车灯、车牌或轮毂把
  红车误判为 `white truck`；
- `nav_goal_bridge_node` 已实现 `/goal_pose` 到 Nav2 `/navigate_to_pose` action 的桥接，
  并由 `nav_with_remap.launch.py` 默认启动；
- 黄车回归暴露的“接近点落在车辆另一侧、沿车身擦行、主动降速改变 DWB 轨迹曲率”已完成
  代码修复：障碍类目标要求机器人同侧 road 接近点，优先直线路径净空，仿真不再使用
  未验证回退点；Nav2 改用包含腿部扫掠的多边形 footprint，主动感知同比缩放线角速度；
  自动测试已覆盖原复现坐标，仍需全新 Gazebo 进程做闭环验收；
- 本轮导航安全定向回归为 `19 passed`；排除现有 flake8/pep257 门禁后的完整功能测试为
  `104 passed, 1 skipped`。`semantic_mapping` 单包 colcon 构建和 launch 参数检查通过；
  当前安装应包含 8 个 console scripts；
- Git 当前位于本地 `master`，HEAD 为 `ada6240`，本文描述的多项功能仍在未提交工作区，
  开始新实验前不要清理或覆盖这些修改；
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
- Lite3 的传感器话题和单路录包链路已确认，但相机内参、LiDAR-相机外参、完整 TF
  和底盘速度接口尚未确认；
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
| Lite3 实机语义导航 | 标定、定位 TF、运动安全桥和 Nav2 闭环未完成 |

## 3. 当前系统组成

### 3.1 定位与点云前端

- 算法：FAST-LIO；
- Bag 配置：`fast_lio/config/velodyne.yaml`；
- Gazebo 配置：`fast_lio/config/sim_mid360.yaml`；
- 主要输出：`/Odometry`、`/cloud_registered`、`/cloud_registered_body`、TF
  `odom -> base_link`。

注意：FAST-LIO 发布的是大写 `/Odometry`，当前 Nav2 参数读取小写 `/odom`。
Gazebo 中 `/odom` 通常由 CHAMP/robot_localization 提供。实机部署前必须统一里程计
话题和 TF 权威源。

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
- 输出逐像素类别、置信度、彩色预览和参与推理的源图像；
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
9. `motorcycle`
10. `chair`
11. `bench`
12. `unknown background`

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
地图不确定性。这样 `query_min_class_prob` 不会因为类别从 6 类扩展到 12 类而系统性压低
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

`active_perception_node` 根据 `/plan` 附近体素的熵调节速度。为避免后处理改变 DWB
已经碰撞检查的转弯半径，当前默认按同一比例缩放线速度和角速度。Gazebo 配置中的
命令链为：

```text
Nav2 /cmd_vel -> active_perception_node -> /cmd_vel_champ -> CHAMP
```

进度检查已与最低 30% 速度倍率对齐为“20 秒内移动 0.15 m”，避免安全低速运动被误判
为卡死；真实障碍碰撞仍由 footprint、VoxelLayer 和 DWB 拒绝，不能靠放宽进度门限掩盖。

## 4. 主要话题接口

| 话题 | 类型 | 作用 |
| --- | --- | --- |
| `/text_query` | `std_msgs/String` | 文本查询入口 |
| `/clip_logits` | `Float32MultiArray` | CLIP 网格类别分数 |
| `/clip_features` | `Float32MultiArray` | CLIP 网格特征 |
| `/query_feature` | `Float32MultiArray` | CLIP 文本特征 |
| `/segformer/class_mask` | `sensor_msgs/Image` | 12 类逐像素掩码 |
| `/segformer/confidence` | `sensor_msgs/Image` | 逐像素置信度 |
| `/segformer/color_mask` | `sensor_msgs/Image` | RViz 语义预览 |
| `/segformer/source_image` | `sensor_msgs/Image` | 与掩码同时间戳的 RGB 图像 |
| `/semantic_cloud` | `sensor_msgs/PointCloud2` | 类别着色三维语义图 |
| `/uncertainty_cloud` | `sensor_msgs/PointCloud2` | 不确定性着色点云 |
| `/voxel_entropy_data` | `sensor_msgs/PointCloud2` | 主动感知使用的体素熵 |
| `/semantic_cost_map` | `nav_msgs/OccupancyGrid` | Nav2 全局语义代价层 |
| `/query_target_pose` | `geometry_msgs/PoseStamped` | 观测到的目标表面簇估计位置 |
| `/goal_pose` | `geometry_msgs/PoseStamped` | 目标附近的安全接近点 |
| `/path_entropy` | `std_msgs/Float32` | 当前局部路径平均熵 |
| `/semantic_speed_scale` | `std_msgs/Float32` | 主动感知速度缩放比例 |

`/query_target_pose` 和 `/goal_pose` 含义不同，不应混用。前者是观测到的目标表面簇位置，
不保证等于车辆几何中心；后者是安全接近点。`nav_goal_bridge_node` 会把后者转换为 Nav2
`NavigateToPose` action 请求；
是否真正移动还取决于 Nav2 生命周期、规划器、速度链路和底盘控制器均正常。

## 5. 三套配置

| 配置 | 点云 | IMU | 图像 | 用途 |
| --- | --- | --- | --- | --- |
| `semantic_mapping_m2dgr.yaml` | `/velodyne_points` PointCloud2 | `/handsfree/imu` | 压缩图像 | M2DGR Bag |
| `semantic_mapping_sim_livox.yaml` | `/livox/lidar` CustomMsg | `/imu/data` | `/d435i/image_raw` | Gazebo Go2 |
| `semantic_mapping_lite3_real.yaml` | `/timefix/lidar` CustomMsg | `/timefix/imu` | `/camera/color/image_raw` | Lite3 已确认传感器接口 |

三套配置默认 `semantic_backend: clip`。运行 SegFormer 时必须通过命令行设置：

```bash
-p semantic_backend:=segformer
```

Lite3 文件中的四个输入话题已用实机数据确认；相机内参、LiDAR-相机外参、坐标系和
`device: cuda` 的完整算法负载仍必须在实机上确认。

## 6. 已完成验证

截至本文更新时已完成：

- Python 语法检查通过；
- 三套 YAML 可解析，类别、代价、尺寸约束长度均为 12；
- 39 个 Lite3 采集、时间戳和离线烟测专项测试通过；
- 本轮导航安全定向回归为 `19 passed, 1 warning`；排除现有 flake8/pep257 门禁后的
  完整功能测试为 `104 passed, 1 skipped, 3 warnings`。警告来自已知的 SciPy/NumPy
  版本范围不一致和未注册 pytest 标记；
- `colcon build --symlink-install --packages-select semantic_mapping` 通过；
- `ros2 pkg executables semantic_mapping` 可安装 8 个入口，包含
  `nav_goal_bridge_node`；
- 标准 ROS 2 launch 能从 package share 找到配置；
- M2DGR Bag 中 FAST-LIO、CLIP、SegFormer 和语义点云有实际运行记录；
- M2DGR SegFormer 路线已成功查询 `car`：目标簇为 2 个体素、累计证据 6.0，
  `/query_target_pose` 为约 `(13.20, -2.75)`，`/goal_pose` 为约
  `(14.15, -3.15)`，水平接近距离约 1.03 m；
- Gazebo 中 Go2、Mid360、D435i、Nav2 手动目标和 car benchmark 有实际运行记录。
- `school_parking_lot.world` 中旧版 `white truck` 曾错误选择红车局部白色附件；簇级
  颜色支持率修复及自动测试已完成，干净 Gazebo 回归尚待执行，不能提前记为实测通过；
- CARLA 0.9.16 的采集和评测工具已能生成车辆、行人和两轮车数据；已有有效采集记录，
  但最终 CLIP/SegFormer 对比指标报告尚未完成，不能作为论文最终结果；
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

当前 Python 环境仍会报告 SciPy 要求 NumPy `<1.25`、实际 NumPy 为 `1.26.4` 的警告。
核心测试可通过，但正式实验前应使用隔离环境固定依赖版本。

## 7. 未实现或未充分验证

### P0：完成闭环前必须解决

1. 增加查询状态、自动重试、超时和结果反馈，避免用户反复手动发布 `/text_query`；
2. Gazebo 和 Lite3 均只允许一个 `odom -> base_link` TF 发布者；
3. 统一 FAST-LIO `/Odometry`、Nav2 `/odom` 和机器人本体状态估计；
4. 实测 Lite3 完整 TF 树、相机内参和 LiDAR-相机外参；
5. 实现 Lite3 官方运动控制接口与 ROS `Twist` 之间的安全桥接，包括急停和限速。
6. 将 CLIP logits/features 改为携带源图 Header 的原子消息，避免融合“最新缓存”；
7. 让 GA-BSVM 按点云时间查询 `odom <- pointcloud_frame`，不能继续把 LiDAR 点直接
   套用最新 `odom <- base_link`。

### P1：论文实验和可靠性

1. 在 car/person/bicycle 等可测目标上建立定位误差和导航成功率指标；
2. 对 CLIP、SegFormer、SegFormer+GA-BSVM、无主动降速等方案做消融；
3. 解决动态目标留下历史体素的问题；
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
2. 完成 Lite3 CameraInfo、静态 TF 和 LiDAR-相机外参标定；
3. 修复 CLIP 源时间戳和 GA-BSVM 点云坐标变换后，采集受控低速运动 Bag；
4. 先完成已实现 Nav2 action 桥接和簇级颜色查询的 Gazebo 闭环验收，再实现查询
   状态机、自动重试、超时与取消；
5. 解决 Gazebo/Lite3 单一里程计与 TF 权威源；
6. 完成运动安全桥后，才进行架空、低速空场和障碍环境实机测试。

完整运行步骤见 [RUNBOOK.md](RUNBOOK.md)。

## 9. Git 状态说明

本文更新时，`~/ws/src/semantic_mapping` 位于本地 `master`，HEAD 为 `ada6240`，
但当前功能主要存在于尚未提交的工作区修改中。新实验开始前应先完成一次明确的提交和
远端备份，否则 Git HEAD 不能代表本文描述的版本。
