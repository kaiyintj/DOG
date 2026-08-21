# ARIS / Auto-Claude CARLA 执行约束

## Pipeline Status

```yaml
language: zh
workflow: ARIS_W1.5_experiment_bridge
orchestrator: wanshuiyin/Auto-claude-code-research-in-sleep
backend: local
platform: Linux
ros_distro: humble
python: "3.10"
accelerator: CUDA_single_gpu
wandb: false
code_sync: local
AUTO_DEPLOY: true
SANITY_FIRST: true
MAX_PARALLEL_RUNS: 1
planning_model: gpt-5.6-sol
review_model: gpt-5.6-sol
execution_model: gpt-5.6-luna
execution_reasoning_effort: max
scope: CARLA_simulation_and_offline_evaluation_only
p2_auto_deploy: false
```

## 范围约束（HERO：反过度防御）

来源：https://github.com/wanshuiyin/HERO-Anti-OverDefense

这些约束管“提议什么修法”，不管“找什么”：这里真的有问题都要报，包括听起来罕见但本项目确实会产生的情况。报完后把修法收在范围内：

1. 这不是安全攻防论文：默认操作者是自己机器上的合作者。校验欢迎，过度防御禁止。
2. 不加哈希/校验和/指纹，除非它替代了一个实质上更贵的操作，并且结果会改变下一步做什么。
3. 不为这里不会发生的情况加 feature flag、迁移框架、兼容层、包装层或守卫。
4. 冷门编码、符号链接竞态、毫秒级竞态等一律不在范围内，除非经由本项目受支持的用法可达——文档示例、公开接口、真实数据。可达即可，不需要复现；“理论上构造得出”不算。
5. 该判断的地方就判断，不要换成评分表、检查清单，或对已经定论的东西再跑一遍校验/审计。
6. 以上不覆盖用户、本项目约定或更高优先级规则明确要求的安全、迁移、校验与审阅。本文件已有的结构硬门、质量门、tracker 状态和用户指定流程是被要求的，属于工作本身，继续执行。

过度防御的四种形状，用于校准而不是当清单：Hashing（哈希）、Edge cases（边界情况）、Rubrics（机械判断）、Overbuild（过度建设）。一个真问题不会因为“长得像其中一条”就被驳回。

跑任何检查前先回答：这次运行会检测出什么具体失败？真出现了我会做什么不同的事？答不上来就别跑。对的就说对，不要为了交差硬找问题。

## 当前任务

- 固定入口：`refine-logs/EXPERIMENT_PLAN.md` 与 `refine-logs/EXPERIMENT_TRACKER.md`。
- 严格按 P0 → P1 执行；P2 只保留预算和矩阵估算，未经用户再次确认不得启动。
- `AUTO_DEPLOY=true` 表示结构硬门通过后自动继续 P0/P1。质量门失败必须保留为负结果，但不得据此停止后续独立诊断或消融；P2 hold 仍需用户另行确认。
- 单 GPU 上 CARLA server 与 SegFormer 都占显存；采集和离线推理串行，`MAX_PARALLEL_RUNS=1`。
- W&B 禁用。状态、命令、退出码、日志与结果只写本地带时间戳运行目录。

## 不可违反的研究边界

- `evaluation_type=simulation_only`。Semantic LiDAR 仅是离线 GT；不得进入 SegFormer 输入、可靠性公式或 VoxelMap 预测输入。
- 不改变 GT 映射，不删困难样本，不偷降阈值，不把 unsupported 类计入主指标。
- `electric_bicycle` unsupported；禁止用 `bicycle` 代替。
- Motion V2 当前只通过机制响应，未通过体素主门；禁止接入 `semantic_mapping/runtime/ga_bsvm_node.py` 或正式实机权重。
- 不修改 `config/semantic_mapping_sim_livox.yaml`。参数扫描只修改 `$AUTOTEST_ROOT/configs/` 下的副本。
- 不运行 Gazebo、Nav2 或实机节点；不提交、不推送、不删除旧数据、不清理用户 dirty 修改。

## 执行与故障纪律

- 每个子运行使用唯一输出目录；不得覆盖既有结果。只信 `manifest.json` 中的地图、版本、seed 和传感器配置。
- **结构硬门（可停止依赖链）**：环境/资源不可用；build、定向测试、代码入口或命令执行失败；manifest/schema/固定 revision 不成立；GT alignment 无效；数据或支持 GT 为空；必需结构化产物缺失/不可解析。只把真正依赖该无效产物的 run 标记 `NOT_RUN`；不相关分支继续。
- **质量门（不可停止自主测试）**：SegFormer IoU、逐类 detection/class accuracy 低于目标，reliability AUROC/CI 不支持 claim，VoxelMap accuracy/coverage/gap/AUROC 未改善，或 Motion V2 效果门失败。这些 run 记为 `FAIL`/negative/inconclusive，仍继续后续独立 seed、因素诊断、motion 和六组消融。
- 先读 traceback/stderr/主日志。禁止无改动重试。
- 对有日志证据的实验工具缺陷，执行者最多做 2 次最小修复并复测；每次必须保存 diff、命令、退出码和新日志。该授权不包括改变 GT、验收阈值、正式 YAML、实验假设或把 Motion V2 接入 runtime。
- 普通环境差异由执行模型诊断并做必要的最小处置；SciPy/NumPy warning 在没有异常时只记录，不当作失败。
- 端口被占用时先识别进程。禁止 `pkill`、`killall` 和无法确认归属的终止操作。
- 状态只使用 `PASS`、`FAIL`、`BLOCKED`、`INVALID`、`NOT_RUN`；tracker 初始状态保持 `TODO`。其中 `FAIL` 默认表示已得到可用但未达质量门的负结果，不传播停止；`BLOCKED`/`INVALID`/执行失败只阻断实际依赖该产物的 run。

## 工作树保护基线（2026-08-22，B 盘迁移）

算法与实验基线 commit 为
`d27c1032f97d8e744c3ee2f2ef196c00ea6bac7e`。A 盘文档来源为分支
`docs/lite3-handoff-20260821` 的 commit
`f9b75ded0d5d8cbe207d22c5491b04800b1f8801`；该提交只同步 Lite3 Bag/离线烟测证据、
交接入口和默认搜索规则，不改变算法源码、正式 YAML、测试或既有实验结论。

用户已授权把上述文档变更迁入 B 盘并适配 B 盘精简归档，以及移除与时间戳副本逐字节相同的
`refine-logs/P1_OFFLINE_DIAGNOSTIC_TRACE.md` alias；规范审计副本
`P1_OFFLINE_DIAGNOSTIC_TRACE_20260817_135410.md` 必须保留。当前修改须保持未提交，等待
用户最终审查；该授权不包括删除或移动其他历史文件，也不包括提交或推送。

历史审查、时间戳 plan/tracker、规范 P1 trace 与 `.aris` 记录仍受 Git 跟踪，只通过
`.rgignore` 从默认全文搜索中排除。执行任何新任务前必须重新记录
`git status --short --branch`；上述获批迁移文件是当前预期 dirty 内容，任何其他新增
dirty 内容均应先视为用户修改并保护。

B 盘当前不可变原始 Bag 位于
`/home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7`，精简烟测证据位于
`/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN`。精简目录故意
不含 `merged/` 和 `output_bag/`；其历史 manifest/source_path 中的 A 盘路径属于运行时
来源记录，不得改写。验证方法以 `docs/RUNBOOK.md` 第 8.9 节为准。
