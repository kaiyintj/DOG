# 当前能力与待办

更新日期：2026-09-16

本文只回答三件事：当前代码能做什么、哪些结果已经验证、下一步缺什么。
具体命令见 [RUNBOOK.md](RUNBOOK.md)，室内实验细节见
[INDOOR_GAZEBO_BENCHMARK.md](INDOOR_GAZEBO_BENCHMARK.md)。

## 当前主线

当前正式仿真主线是 `SegFormer + GA-BSVM`：

```text
RGB + LiDAR + IMU
  -> FAST-LIO
  -> SegFormer full posterior
  -> camera-LiDAR projection
  -> reliability-weighted Dirichlet fusion (GA-BSVM)
  -> semantic map
  -> language query
  -> safe approach goal
  -> Nav2
```

CLIP 仍保留为开放词汇对比后端，但不与 SegFormer 结果混为同一个 benchmark。

一次运行必须显式选择一个 semantic profile：

| Profile | 用途 | 可通行角色 | 默认 checkpoint |
| --- | --- | --- | --- |
| `outdoor13` | 原户外 13 类 | `road` | Cityscapes B0 |
| `indoor7` | floor/wall/door/chair/table/shelf/bed + unknown | `floor` | ADE20K B0 |

Profile 决定类别顺序、posterior 通道数、查询别名和导航角色。运行中不切换 profile；
切换时重新启动语义地图，避免不同维度的证据混合。

## 能力矩阵

| 场景或能力 | 当前结论 | 不能据此宣称的内容 |
| --- | --- | --- |
| M2DGR Bag 感知与语义建图 | 已有运行验证 | Bag 不能验证机器人导航执行 |
| 户外 Go2 Gazebo | 已有建图、查询与导航工作路径 | 不代表室内参数自动适用 |
| Small House 传感器启动 | RGB、Livox、IMU、FAST-LIO 启动门已工作 | 不代表长时间或运动定位精度 |
| Indoor-7 SegFormer | ADE20K checkpoint 的七类标签均真实存在 | 标签存在不等于 Gazebo 场景识别准确 |
| Indoor GA-BSVM | K=8 posterior、profile handshake、融合与查询失败关闭已验证 | 尚无完整室内类别精度评估 |
| 室内负例 | nonqueryable、world-absent、unsupported、unresolved 均未错误发布 goal | 不证明正例导航成功 |
| 室内 chair 正例 | 历史默认入口已有一次自动到达记录；也存在失败和无终态运行 | 不代表最新代码重复性或正式成功率通过 |
| 室内 table 正例 | 查询可接受 | 当前场景未形成满足阈值的目标簇 |
| Safe approach | 使用 traversable role、机器人同侧和距离约束；路径与碰撞检查交给 Nav2 | 尚无正式 Valid Goal Rate/碰撞率统计 |
| 主动感知减速 | 节点、配置、状态和速度缩放链已接通 | 尚未完成 Recovery OFF/ON 收益对比 |
| Lite3 | 电脑端配置与失败关闭保护保留 | 未授权真实运动；当前任务不验收实机 |
| CARLA | 现有采集和可靠性实验资料保留 | 不属于当前 Gazebo 室内主线 |

## 核心实现边界

### 语义前端

- SegFormer 发布项目类别的完整 posterior，而不是只传 argmax mask。
- raw checkpoint 类别先聚合为所选 project ontology，再进行硬判决。
- checkpoint 不支持的查询必须返回 unsupported，不伪造概率。
- CLIP 作为独立 backend 使用；选择 CLIP 时不同时启动 SegFormer。

### GA-BSVM 与查询

- GA-BSVM 只有在 profile 名称、类别顺序和 posterior 维度一致时才融合。
- 可靠度参与 Dirichlet 证据更新；无有效 IMU 对齐时使用保守可靠度。
- `/query_target_pose` 是语义目标估计位置。
- `/goal_pose` 是经过 traversability、侧向关系和距离约束的接近点；路径与碰撞检查由 Nav2 完成。
- nonqueryable、unsupported、unresolved 和 not-ready 查询均失败关闭。
- 日常查询使用后端无关的 `semantic_query` 一次性命令；旧 `clip_query` 仅作兼容别名。

### 导航与速度链

- Gazebo Nav2 使用 `nav_sim.launch.py`，定位输入可选 `/Odometry`；该入口默认加载
  `nav2_sim_params.yaml`。该仿真配置保留点云和语义障碍层，但关闭 RPP 的预测碰撞 veto，
  让语义接近点能够完成最后一段到达；真实机入口继续使用 `nav2_params.yaml`。
- `nav_goal_bridge_node` 把最新有效 `/goal_pose` 转换为一个活动 Nav2 action。
- 主动感知节点根据路径附近熵和输入新鲜度发布 `/semantic_speed_scale`。
- 仿真中键盘建图模式与自动导航模式不同时争用 `/cmd_vel_champ`。
- 不增加绕过 Nav2 或底盘单写者约束的新命令通道。

## 室内 Small House 当前证据

组合入口 `semantic_sim.launch.py` 会等待控制器、连续稳定 IMU 窗口和新鲜非空
LiDAR 后再启动 FAST-LIO。这个门解决了机器人出生/落地瞬态导致的初始化发散。

已确认：

- Small House 资产由 `go2_config` 管理；
- benchmark manifest 由 `semantic_mapping` 管理；
- SegFormer 发布 `16FC8` Indoor-7 posterior；
- GA-BSVM 生成语义点云、熵和语义代价信息；
- floor、场景不存在的 bed、profile 外的 car、checkpoint 不支持的 table 组合均按预期
  失败关闭；
- chair 可形成目标估计；使用 PointCloud2 FAST-LIO 输入和 Gazebo 专用 Nav2 配置的最近一次
  隔离试验以 `SUCCEEDED` 结束，机器人实际位移约 1.40 m，`/Odometry` 未出现跳变；
- 保留 RPP 预测碰撞 veto 的对照配置会在接近椅子时反复报告 `collision ahead` 并 `ABORTED`，
  因而不能把该对照结果当作定位或速度故障；
- table 正例尚未形成满足当前门限的稳定目标簇。

这些值是诊断证据，不是完整 benchmark 统计。原始运行日志不进入源码仓库；需要复现实验时
由 runner 在 `/home/yk/ws/indoor_benchmark_runs/` 重新生成。

## 2026-09-16 查询改动与验收边界

- 同类候选按现有评分顺序最多尝试两个；规划器返回接近点决策后，ROS 节点统一发布。
- 未找到接近点的候选不发布 target/goal；待处理查询仅在地图 revision 改变且达到重试间隔后重新评估。
- 接近点快照先筛选本轮候选的搜索邻域，再计算语义置信度；共享只读快照，避免重复复制。
- 查询与地图专项测试 54 项通过，完整测试 388 项通过、2 项跳过（含 benchmark 进程测试），
  ROS 包构建通过；本次修改后尚未重跑 Gazebo。
- 排序和重试状态仍主要由节点管理，查询模块架构整理尚未全部完成。

2026-09-15 性能基线实际记录建图/导航阶段平均 RTF 约 0.469/0.398，
GA-BSVM 平均单核 CPU 约 76.2%/92.3%。该次 chair 查询被 Nav2 接收，
但日志缺少终态，因此不能计为成功或超时失败。原始采集与分析位于
`/home/yk/ws/indoor_benchmark_runs/aws_small_house/20260915_045858_performance_baseline/README.md`。
这些值是修改前基线，不证明本次优化已经改善运行性能。

除真机外，仍需以本次发布版本补齐：固定流程的 chair 重复到达、两个同类目标的
接近点回退、全部候选失败及地图变化后的重试、table 失败层定位、修改前后性能对照，
以及独立的目标误差/有效接近点/无碰撞到达统计。主动减速和融合输入表示的收益
仍需匹配条件的对照实验；跨场景泛化属于后续扩展验收。

## 当前优先级

1. 在 Small House 获得至少一个 chair 的可重复闭环到达案例。
2. 判断 table 失败来自模型场景表现、可视范围还是融合/聚类阈值，不先放宽失败关闭门。
3. 为成功案例补齐 target error、goal validity、collision-free arrival 和 approach distance。
4. 用同一 case 对比主动减速关闭与开启；在这之前不宣称速度策略提升成功率。
5. Small House 稳定后再接 Bookstore，Hospital 放在其后。
6. Lite3 实机工作继续保持隔离和失败关闭，不阻塞当前室内仿真开发。

## 专题文档

- [RUNBOOK.md](RUNBOOK.md)：当前可执行命令与排障。
- [INDOOR_GAZEBO_BENCHMARK.md](INDOOR_GAZEBO_BENCHMARK.md)：Indoor-7、资产、manifest 和实验结论。
- [LITE3_REAL_HANDOFF.md](LITE3_REAL_HANDOFF.md)：实机交接和运动安全门。
- [CARLA_IMAGE_BENCHMARK.md](CARLA_IMAGE_BENCHMARK.md)：CARLA 图像采集。
- [CARLA_RELIABILITY_BENCHMARK.md](CARLA_RELIABILITY_BENCHMARK.md)：CARLA 可靠性实验。
- [SEGFORMER_EBIKE_FINETUNE.md](SEGFORMER_EBIKE_FINETUNE.md)：电动自行车类别训练。

`docs/research/` 只保存有来源价值的背景调研，不是运行说明或当前状态来源。

## 查看实际工作区状态

文档中的测试数字可能随代码继续变化。需要交接或发布时，以当前 Git 状态和一次新的定向检查
为准：

```bash
cd ~/ws/src/semantic_mapping
git status --short
git rev-parse --short HEAD
```

发布后仍应核对当前工作区是否包含新增修改，不应把某个历史 commit 的测试结果套用到全部现状。
