# CARLA 实验预部署代码/计划审查

审查时间：2026-08-17 13:03 +0800  
审查角色：ARIS W1.5 planning/review（gpt-5.6-sol）  
范围：指定计划、benchmark 文档、Motion V2 归档、`semantic_mapping/carla/` 全部真实源码、`setup.py`、正式 YAML、现有数据与 Git dirty 状态。未运行实验，未修改源码或正式 YAML。

## 结论

代码入口、参数名、数据 schema、GT fail-closed、strict JSON、六组 VoxelMap 消融和模型 revision 参数均与计划主链匹配。两个实际阻断都已在部署前解除；执行器仍必须从 P0 硬门重新确认，不能跳过。

## BLOCKING

### B1 — 当前审查 sandbox 看不到 CUDA/NVIDIA driver（已由外层预检解除）

受限审查 sandbox 的只读预检得到 `torch.cuda.is_available() == False`，`nvidia-smi` 返回无法连接 NVIDIA driver；这会阻断在 sandbox 内部署。主线程随后已在非受限本地执行上下文确认：`nvidia-smi` 正常，RTX 5070 8 GB 约占用 617 MiB，Torch `2.12.0.dev+cu128` 且 `torch.cuda.is_available() == True`，CARLA Python 包可导入。因此它是审查隔离造成的假阴性，不是当前开放阻断。

处置：执行模型的第一条动作仍必须在实际本地部署上下文重新运行 P0-A preflight并保存日志。只有当次 `nvidia-smi`、PyTorch CUDA 和 CARLA 0.9.16 client/server 都通过才继续；否则写 `BLOCKED`，不改为 CPU、不静默换后端。

### B2 — 原计划 turning point/voxel 可能复用非空 output（已在新工件中修正）

`carla_evaluate_reliability.evaluate()` 会在 `--output` 已存在且非空时直接报错。原计划先把 turning point-only 写到 `regression_turning`，随后仅说“再执行一次 `--voxel-eval`”；若沿用相同输出路径，第二次必失败。

处置：新计划和 tracker 已固定为两个独立目录：`regression_turning_point` 与 `regression_turning_voxel`。执行器不得合并。

## 非阻断环境记录

- 历史归档在 Python 3.10.12、NumPy 1.26.4、SciPy 1.8.0、Pillow 12.2.0、Torch CUDA build、Transformers 4.46.3 上完成；SciPy/NumPy 版本 warning 当时未导致异常。按任务约束只记录，若实际运行没有 traceback/非零退出码，不升级为故障。
- 当前本地已存在指定 SegFormer revision `21b3847fae21ddee674abd31129307b6a1235bd9` 的 config、processor 与模型权重快照；模型 revision 本身不是阻断。
- 三个 P0-B 旧数据目录及 CARLA 0.9.16 安装目录均存在；普通路径/环境差异由执行者从日志做最小诊断。

## 修复权限边界

执行者仅可对有 traceback/stderr 证据的实验工具问题做最多 2 次最小修复并复测；每次保存 diff 与日志。禁止无改动重试、改变 GT、降低硬门、改正式 YAML、删样本或把 Motion V2 接入 runtime。
