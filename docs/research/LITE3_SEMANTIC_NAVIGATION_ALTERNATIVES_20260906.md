# Lite3 语义/语言导航替代路线研究

日期：2026-09-06

## 结论摘要

用户提供的论文是 Steve Wilhelm 的 UCF 2025 硕士论文 *Autonomous Navigation and Real-Time 3D Reconstruction of Interior Spaces Using a Quadruped Robot*。它展示的是 Unitree Go2 EDU + Jetson Orin NX 16GB + Hesai XT32/D435i 的室内几何导航和三维重建，不是闭集语言导航或开放词汇语义导航：论文使用 LIO-SAM/DLIO 做 3D LiDAR-惯性 SLAM，使用点云转 LaserScan，再用 SLAM Toolbox 和 Nav2 做 2D 建图与导航；目标主要从 RViz 设定。

对 Lite3 最有价值的启示是拆分实时职责：

```text
实时几何层：MID-360/IMU -> FAST-LIO -> 2D LaserScan/障碍物 -> Nav2
语义层：D435I -> 低频闭集分割或按需开放词汇查询 -> 目标/语义代价
控制层：Nav2 -> /cmd_vel_lite3_safe -> 厂商安全桥
```

这比让 CLIP/SegFormer、三维语义体素、Nav2 和高分辨率重建同时成为硬实时链更适合 Xavier NX。论文明确报告：2D/3D SLAM 与实时可视化同时运行会增加计算负载；复杂、长时间扫描会出现漂移和重建误差；建议减少实时高分辨率可视化、缩短扫描，并考虑外部处理。

## 1. 论文实际做了什么

根据论文摘要、第 28--39 页、第 52--64 页：

- 平台是 Unitree Go2 EDU，扩展坞使用 Jetson Orin NX 16GB，论文写明最高 100 TOPS、8 核 CPU、10--25 W；不是 Lite3 的 Xavier NX。
- 传感器是 Unitree L1、Hesai XT32、Intel RealSense D435i；论文实验主要使用 XT32 做高密度 3D 重建，并使用 L1 或点云转 2D scan 支撑导航。
- 3D 层测试 LIO-SAM 与 DLIO。LIO-SAM 还做了 RGB 颜色化，但论文指出标定和 IMU 预积分仍有限制。
- 2D 层使用点云转 `LaserScan`，再由 SLAM Toolbox 建图/定位，Nav2 负责规划和导航。
- 论文没有实现 CLIP、SegFormer、语言查询、开放词汇检测、目标实例跟踪或“语言到目标接近点”的闭环。论文中的导航目标主要由 RViz 设置。
- 论文发现 2D/3D SLAM 和实时可视化同时运行计算负载较高；低质量传感器、长距离和复杂扫描会导致地图漂移或对齐误差。

因此，不应把论文当作现成的语言导航算法；应把它当作一个**几何导航与高质量重建解耦**的系统架构参考。

## 2. 和当前 Lite3 的对照

### 当前路线保留的部分

- MID-360 + IMU 的 FAST-LIO 定位基础；FAST-LIO 已经适配逐点时间的 `CustomMsg`。
- Nav2 和 Lite3 的速度安全链设计。
- D435I 作为低频语义传感器。
- `SemanticProfile` 的闭集类别契约、fail-closed 查询能力门和 indoor/outdoor profile。
- `/query_target_pose` 与 `/goal_pose` 分离，以及同侧可通行接近点门禁。
- 语义结果对速度调节和语义 costmap 的辅助作用。

### 最值得放弃或降级的部分

1. **不要把高密度实时 3D 重建作为导航前置条件。** 论文也显示这是计算和漂移的主要来源之一。导航只需要可靠 odom、2D obstacle scan 和 Nav2 costmap；高分辨率点云保存改为按需或离线。
2. **不要让每个 LiDAR 帧都进入完整语义体素融合。** 保持 LiDAR/IMU 10/200 Hz，语义图像推理独立低频运行；目标出现时再提升短时查询频率。
3. **不要同时把 CLIP、SegFormer、颜色、特征和完整语义云都作为常驻硬实时链。** 每次只选一个语义后端；开放词汇查询应按需启用。
4. **不要把语义识别直接变成速度控制。** 语义层输出类别、目标簇或区域约束；Nav2/几何障碍层和安全桥继续负责运动。
5. **可以放弃 GA-BSVM 的完整 feature/颜色累积作为第一阶段导航依赖。** 若目标是先跑通闭集导航，可以改为“2D/3D 目标簇 + 时间滤波 + 目标确认”轻量路径；完整 Dirichlet/feature map 保留为研究模式，而不是运动必需模块。

## 3. 三条可落地路线

### 路线 A：低算力闭集语言导航，推荐作为 Lite3 主路线

目标示例：`去找人`、`去汽车旁边`、`到椅子附近`。

```text
SegFormer-B0 低频推理
-> project posterior / class mask
-> LiDAR 投影或深度辅助的目标点
-> 3D/2D 时间滤波与聚类
-> traversable/same-side approach
-> Nav2 NavigateToPose
```

建议：

- 室内使用 ADE20K 或针对 Lite3 场景微调的轻量 checkpoint；室外使用现有 outdoor13。
- 语义推理 1--3 Hz 起步，FAST-LIO 和障碍更新保持实时。
- 不需要每帧发布 semantic cloud；只在目标更新或调试时发布。
- 目标类别、颜色、可查询性和安全接近条件由现有 `SemanticProfile` 管理。
- 只对确认后的目标生成 `/goal_pose`，未知类别和 checkpoint 不支持类别继续 fail-closed。

优点：CPU/GPU 负载最低，容易验证，和现有 indoor7/outdoor13 profile 最接近。
缺点：语言表达受类别表限制；不能可靠处理任意新物体或复杂关系。

### 路线 B：开放词汇按需导航，保留 CLIP 但不让它常驻避障

目标示例：`去红色背包旁边`、`找灭火器`、`去蓝色箱子附近`。

```text
用户文本
-> CLIP/OpenCLIP 查询 embedding
-> 图像网格/候选区域低频相似度
-> LiDAR 投影 + 空间聚类
-> 多帧确认、颜色/尺寸/可通行性过滤
-> 安全接近点 -> Nav2
```

建议：

- CLIP 只在收到查询后运行；查询结束后停止或降低到很低频率。
- 不把单个最高相似度网格当作目标；至少要求空间簇、连续观测、点数/证据阈值和可通行接近点。
- 保留现有颜色支持率和源图时间身份约束；它们是防止车灯/反光区域误判的重要机制。
- 没有可靠目标或接近点时只返回 pending/rejected，不发布 Nav2 goal。
- 避障仍使用几何点云/2D scan，不使用 CLIP 相似度代替碰撞层。

优点：支持开放词汇，能逐步增加目标类型，不需要立即训练新分割模型。
缺点：Xavier NX 上查询延迟不稳定；CLIP 网格空间分辨率有限，不能替代实例分割；目标遮挡和重叠场景需要更强检测器。

### 路线 C：伴随计算机上的完整语言导航

```text
Lite3：驱动、状态、底盘安全桥、必要的 FAST-LIO/数据转发
伴随 Orin/x86：FAST-LIO、SegFormer/开放词汇检测、语义融合、Nav2、语言解析
```

语言模型或 VLM 只输出结构化任务：

```json
{
  "target_class": "chair",
  "attributes": ["red"],
  "relation": "near",
  "approach": "same_side_traversable"
}
```

随后仍由目标选择器、Nav2 和安全桥执行。不要让 VLM 直接输出 `/cmd_vel`。

优点：能使用更强的 Grounding DINO/YOLO-World/SAM 或 VLM，语言表达能力最高；机器狗上的 CPU 竞争最小。
缺点：增加网络、时间同步、DDS/序列化和断链处理；安全桥必须在网络断开时自动停车；系统验收复杂度最高。

## 4. 推荐的取舍

### 如果目标是尽快在 Lite3 上部署

选择路线 A：

- 保留 FAST-LIO + Nav2 + 2D 几何障碍层；
- 保留 `SemanticProfile`、SegFormer 和目标接近点逻辑；
- 暂停完整 CLIP feature map、颜色/feature 常驻融合和高频 semantic cloud；
- 把语义推理和目标更新做成 1--3 Hz 的异步层；
- 先只验证 `person`、`chair`、`car` 等 2--4 个类别，不追求全类别；
- 语义目标只作为 Nav2 goal 候选，不作为避障唯一来源。

### 如果目标是保留开放语义能力

选择路线 B，必要时再升级到路线 C：

- 用现有闭集 SegFormer 保证几何导航与通行判断；
- CLIP 只服务于用户查询，不负责连续避障；
- 查询前先把文本解析成颜色/类别/属性；
- 用多帧 3D 簇和安全接近点消除单帧 CLIP 假阳性；
- 复杂关系或长指令交给伴随计算机/VLM，Lite3 只接收结构化目标。

## 5. 建议的实施顺序

1. **先做几何导航基线**：FAST-LIO -> 2D LaserScan/障碍层 -> Nav2；暂时关闭语义节点和高密度点云保存。目标是确认实时性、TF 唯一性和安全桥，而不是语言能力。
2. **接入低频闭集语义**：ADE20K/室外 checkpoint 每 0.5--1 秒推理一次；只发布目标候选和必要的 semantic costmap，不发布完整语义云。
3. **接入目标确认与接近点**：保留现有同侧、直线净空、traversable 和 fail-closed 门禁；先用离线 Bag 和仿真测试。
4. **再接开放查询**：CLIP 只在 `/text_query` 到达后运行；测试 `red car` 等已有能力，不直接增加 VLM。
5. **最后评估伴随计算机**：只有当路线 A/B 在 Xavier NX 上仍无法满足延迟和最大 gap，再把语义或 FAST-LIO 迁移出去。

## 6. 和论文结论的边界

论文使用 Orin NX 16GB 和不同 LiDAR/机器人，不能证明 Xavier NX + Foxy + MID-360 的性能。论文也没有报告闭集/开放词汇语言导航成功率，因此不能直接作为语义导航验收证据。

论文可迁移的强结论是：高分辨率 3D 重建、2D SLAM、Nav2 和实时可视化并发会消耗大量算力；应缩短扫描、减少实时可视化、分离导航所需数据和重建所需数据，并在需要时使用外部计算。

Lite3 仍必须保持：

```text
projection_calibration_verified=false
goal_bridge_enabled=false
MOTION_READY=NO
```

直到真实外参、唯一 TF、受控运动、SDK 安全桥、watchdog、网络断链停车和低速空场验收全部通过。

## 参考

- Wilhelm, Steve. *Autonomous Navigation and Real-Time 3D Reconstruction of Interior Spaces Using a Quadruped Robot*, University of Central Florida, 2025. 用户上传 PDF，第 18--39、52--64 页。
- 项目当前状态：[PROJECT_STATUS.md](../PROJECT_STATUS.md)
- Lite3 交接：[LITE3_REAL_HANDOFF.md](../LITE3_REAL_HANDOFF.md)
- 室内实现与证据：[INDOOR_GAZEBO_BENCHMARK.md](../INDOOR_GAZEBO_BENCHMARK.md)
- 室内 profile 与当前验证结论：[INDOOR_GAZEBO_BENCHMARK.md](../INDOOR_GAZEBO_BENCHMARK.md)
- Livox/Fast-LIO 减载调研：[LITE3_OPEN_SOURCE_LIVOX_CPU_PRACTICES_20260906.md](LITE3_OPEN_SOURCE_LIVOX_CPU_PRACTICES_20260906.md)
