# 当前能力与待办

核对日期：2026-09-29。本文是当前状态入口；运行命令见 [RUNBOOK](RUNBOOK.md)。

## 当前主线

`RGB/LiDAR/IMU → FAST-LIO → SegFormer 完整后验 → GA-BSVM → 类别查询/接近点 → Nav2`。
CLIP 保留为独立对照后端。短类别/颜色查询不等于任意自然语言理解，空间聚类不提供稳定实例 ID。

| 配置 | 场景与类别 | 默认模型 |
| --- | --- | --- |
| `indoor7` | Small House；floor/wall/door/chair/table/shelf/bed + unknown，8 通道 | ADE20K B0 |
| `outdoor13` | School Parking Lot、M2DGR；13 类，road 可通行 | Cityscapes B0 |

一次运行固定一个 profile；切换时重启地图。Bookstore、Hospital 尚未接入。

## 当前能力与证据

| 内容 | 已有证据 | 仍待验证 |
| --- | --- | --- |
| 室内感知与融合 | 8 通道完整后验、profile 检查、同帧检查和失败关闭；模型支持七类标签 | 场景识别精度与融合收益 |
| 室内 chair 导航 | 9 月 16、18 日已有到达与停止录包 | 当前代码复测、重复成功率、碰撞率、有效接近点 |
| 室内 table | 查询可接受，历史运行未形成满足门限的簇 | 区分模型、视角、融合与聚类问题 |
| 户外/M2DGR | 保留建图、查询及户外 Gazebo 导航入口 | 当前版本复验；Bag 不证明导航执行 |
| 主动感知 | 不确定性调速、输入过期降速、命令超时零速 | 匹配条件下 OFF/ON 收益 |
| Lite3 | 静止采集/标定候选；9 月 26 日分别观察到隔离里程计桥、FAST-LIO 和实时语义融合输出 | 全链几何、运动同步、TF 权威、SDK 安全桥与导航 |
| CARLA | 图像评估、三维可靠性和 Motion V2 离线诊断入口 | 当前 AURC 定义下重新评估；Motion V2 未进入在线主线 |

Lite3 的 9 月 26 日融合检查在固定 `rslidar` 系进行，未同时运行 FAST-LIO/Nav2。
用户报告低电量趴下、时刻未确认，因此不能作为全程静止或运动几何验收。
IMU 历史对齐告警尚未查明；停止时的 Livox 转换异常已做限定修复与真实输入退出检查。
详见 [实机交接](LITE3_REAL_HANDOFF.md)。

## 当前实现边界

- 完整 posterior 先聚合项目类别概率；hard-mask 对照路线映射 raw argmax，两者输入表示不同。
- 融合要求 profile、类别顺序、通道数一致；图像产物须同帧，LiDAR 允许近似同步。
- 不支持、不可查询或证据未就绪时不发布目标。日常入口为 `semantic_query`，`clip_query` 是兼容别名。
- 同类候选最多尝试两个，接近点有效后同时发布目标估计与导航目标；不包含 Nav2 失败后自动换实例。
- 接近点保留可通行类别、置信度、物体距离与同侧约束；路径和碰撞检查由 Nav2 负责。
- Gazebo 的 `nav2_sim_params.yaml` 保留障碍层、关闭 RPP 预测碰撞 veto；这不是无碰撞到达证明。
- Lite3 保持 `projection_calibration_verified=false`、`goal_bridge_enabled=false`、`MOTION_READY=NO`；
  `/cmd_vel_lite3_safe` 尚无底盘执行桥。

## 最新修改与验证

9 月 29 日修复：地图为空时发布更新以清除旧障碍；不确定性点云使用最新融合观测的源时间；
AURC 对同分点使用排列期望。旧 AURC 与新结果比较前需要重算，历史报告不改写。

新增 3 项回归测试。当前 429 项测试均通过：426 项在沙箱内通过，3 项本机通信/进程夹具
因权限限制在沙箱外重跑通过。`git diff --check` 通过。本次未重跑构建、完整仿真或实机验收。

此前已完成 hard-mask 同帧检查和目录整理：训练/单图工具在 `semantic_mapping/offline/`，
仿真启动门在 `gazebo/`，共享模型检查在 `runtime/segformer_checkpoint.py`；ROS 命令名不变。
历史清理和逐文件工作空间清单只保存在本机，不作为跨系统论文资料上传。

## 保留的室内实验依据

- [9 月 15 日性能基线](results/indoor_gazebo_20260915_20260918/performance_20260915.md)：
  建图/导航平均 RTF 约 0.469/0.398；缺导航终态，不计为成功或失败。
- [9 月 16 日诊断](results/indoor_gazebo_20260915_20260918/chair_navigation_20260916.md)：到达与停止案例。
- [9 月 18 日复测](results/indoor_gazebo_20260915_20260918/cloud_stride3_20260918.md)：
  `SUCCEEDED`；墙钟 178.305 s、仿真时间 71.080 s。stride 配置由用户报告，Bag 无完整参数转储。

两次成功运行路线与录制内容不同，不能用作性能对照，也不覆盖后续代码修改。
新实验写入 `/home/yk/ws/indoor_benchmark_runs/<world>/<timestamp>_<purpose>/`。

## 当前优先级

1. 固定起点、观测/建图流程与输入，复测当前版本 chair 查询、到达和停止。
2. 覆盖同类候选回退、全部候选失败及地图变化后重试；定位 table 失败层。
3. 补齐目标误差、有效接近点、无碰撞到达与重复成功率，再比较融合输入和主动减速收益。
4. Small House 稳定后扩展 Bookstore，再考虑 Hospital。
5. Lite3 按交接文档推进 TF、同步、几何和 SDK 桥验收。

## 文档分工

| 文档 | 用途 |
| --- | --- |
| [RUNBOOK](RUNBOOK.md) | 当前启动、查询、排障和采集命令 |
| [室内基准](INDOOR_GAZEBO_BENCHMARK.md) | 类别、资产、实验定义与验收指标 |
| [Lite3 交接](LITE3_REAL_HANDOFF.md) | 实机已有证据与下一步门禁 |
| [CARLA 图像](CARLA_IMAGE_BENCHMARK.md) / [可靠性](CARLA_RELIABILITY_BENCHMARK.md) | 两种离线评估流程 |
| [电动自行车微调](SEGFORMER_EBIKE_FINETUNE.md) | 数据集、训练与模型验收；不表示已有合格模型 |
| [室内结果摘要](results/indoor_gazebo_20260915_20260918/README.md) / [Lite3 观测摘要](results/lite3_20260924_20260926/README.md) | 可在其他系统查阅的历史结果与边界 |

`research/` 是对应日期的调研；`results/` 是历史实验摘要；CARLA 原数据集与大型逐点文件当前缺失，不能直接复跑。两者均不替代当前状态或运行说明。
`agents/` 保存协作流程。原始录包、图像和本机清单不随仓库同步。

## 查看实际工作区状态

```bash
cd /home/yk/ws/src/semantic_mapping
git status --short
git rev-parse --short HEAD
```

以命令显示的实际分支和工作区为准；历史测试通过不能自动代表后续修改已验收。
