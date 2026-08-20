---
title: CARLA semantic reliability research contract
evaluation_type: simulation_only
status: active
created: 2026-08-17T13:03:18+08:00
ground_truth: CARLA Semantic LiDAR and camera semantic/instance annotations
primary_plan: refine-logs/EXPERIMENT_PLAN.md
---

# 研究契约：CARLA 语义可靠性与体素融合

## 评价边界

本研究只给出 CARLA 仿真与离线评测证据。它不能证明 Lite3 实机外参、时钟同步、网络、算力负载、地面动力学、Nav2 路径安全或真实导航成功率。所有结论必须标注为 `simulation_only`，不得外推为实机已验证。

CARLA GT 的允许用途：

- 普通 LiDAR 是算法点云；Semantic LiDAR 经 mutual-nearest、距离容差与 frame-level fail-closed 对齐后，只为离线点级/体素级评分提供标签。
- semantic camera 与 instance camera 用于二维 benchmark 真值和审计。
- CARLA actor metadata 可提供 car/truck/bus/bicycle/motorcycle 细类；缺失 actor metadata 时不得由 generic vehicle tag 猜细类。

CARLA GT 的禁止用途：

- 不得进入 SegFormer 输入、posterior、reliability factor、VoxelMap 预测更新或在线 runtime。
- 不得根据预测结果重映射 GT、删除困难点/类别、改变对齐门槛或用另一个模型输出充当 GT。

## 可证伪 claims

| Claim ID | 可证伪陈述 | 证据与失败条件 |
| --- | --- | --- |
| C1 | 固定 revision 的 SegFormer 在 CARLA Town05、seed 11/42/73 上达到 road IoU ≥ 0.70、vehicle IoU ≥ 0.25，且五个导航车辆类各自 detection rate 与 class accuracy ≥ 0.70。 | 以 format-v2 image manifest 和逐类实例结果评分；任何目标类低于门槛即该类 claim 失败，不能由总体均值覆盖。 |
| C2 | 静态 ego、静态目标下，`r_semantic` 对支持类点的 correctness AUROC 高于 0.5，且至少 2/3 seed 同方向。 | 以 seed/frame 为重复单位或 cluster/bootstrap；95% CI 包含 0.5 时只能判为证据不足。 |
| C3 | 至少一个受控 reliability 组合在相同点、posterior、timestamp 和 GT support 下改善 VoxelMap 错误检测信号，同时 all-GT accuracy 下降不超过 1 个百分点、coverage 相对前一组合下降不超过 0.5 个百分点。 | 六组 `none → semantic → semantic_range → semantic_range_density → semantic_range_density_view → full`；若只改善点级排序而不改善体素门，则 claim 失败。 |
| C4 | Motion V2 在 stationary 时不随 offset 无故下降，在 constant_velocity/turning 时随实际相对运动增加而下降。 | 这是机制 claim；若 paired common-point 的 pose/motion 响应不符合预期则失败。它不等价于体素效果 claim。 |
| C5 | 被选候选参数在 seed 11/42/73 中至少 2/3 方向一致，且不存在一个 seed 的显著反向退化。 | 逐 seed 报告均值、标准差和 95% CI；不把每个点当独立重复。 |

## 已知否定结果与 unsupported 类

- Motion V2 当前状态：机制响应通过，但 turning 的体素主验收失败；wrong-minus-correct uncertainty gap 随 Motion V2 下降。因此 Motion V2 只能保留为 CARLA diagnostic，参数未冻结，不得接入 `runtime/ga_bsvm_node.py`。
- `electric_bicycle`：默认 Cityscapes SegFormer checkpoint 没有独立输出通道，CARLA 也没有独立电动车 GT。状态固定为 `unsupported`；禁止把 `bicycle` 伪装或重命名为 `electric_bicycle`。
- 点级 weighted accuracy 只是排序诊断；只有 VoxelMap 六组消融可支持“融合改善”陈述。

## 停止规则

P0 任一硬门失败则后续 P0/P1 为 `NOT_RUN`；环境或资源不可用为 `BLOCKED`，实验完成但低于质量门为 `FAIL`，GT/manifest/类别支持不成立为 `INVALID`。不得为了获得正结果改变本契约。
