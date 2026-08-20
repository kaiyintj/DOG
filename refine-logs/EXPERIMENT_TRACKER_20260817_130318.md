---
tracker_id: carla_aris_w15_tracker_20260817_130318
plan_id: carla_aris_w15_20260817_130318
status: initialized
default_run_status: TODO
created: 2026-08-17T13:03:18+08:00
---

# CARLA ARIS W1.5 Experiment Tracker

固定计划：`refine-logs/EXPERIMENT_PLAN.md`。所有 run 初始为 `TODO`；只有执行模型在写入对应 `run_records/<RUN_ID>.json` 后才能改为 `PASS/FAIL/BLOCKED/INVALID/NOT_RUN`。P2 保持 `TODO (HOLD_P2)`。

## 命令代码表

下表中的命令是实际入口，完整公共参数与 server 生命周期见计划对应章节；变量值由 run ID/矩阵列唯一确定。

| Code | 实际命令 |
| --- | --- |
| C_BOOT | 计划 §2 的 `set -euo pipefail ... mkdir ... AUTOTEST_STATUS.json` 完整命令块 |
| C_PREFLIGHT | 计划 §4 P0-A-001 的 Git/GPU/CARLA/model-cache preflight 命令块 |
| C_BUILD_TEST | 计划 §4 P0-A-002 的 `colcon build` + pytest + flake8 + pep257 命令块 |
| C_ENTRY | `for tool in carla_capture_benchmark carla_evaluate_benchmark carla_capture_reliability carla_evaluate_reliability; do ros2 run semantic_mapping "$tool" --help; done` |
| C_REG_POINT | `ros2 run semantic_mapping carla_evaluate_reliability --dataset "$dataset" --output "$output" --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml --segformer-model "$SEGFORMER_MODEL" --segformer-revision "$SEGFORMER_REVISION" --time-offset-ms 0 20 50 100 150 --frame-tolerance-ms 6 --device cuda --fp16 --max-points-per-frame 1000 --posterior-storage none --no-voxel-eval` |
| C_REG_VOXEL | 与 C_REG_POINT 相同，但唯一 output 为 `regression_turning_voxel` 且末项 `--voxel-eval` |
| C_SERVER | 计划 §6 的端口/进程只读检查；端口空闲时执行 `nohup ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-port=2000 ... &`，并保存 owned PID |
| C_IMG_CAP | `ros2 run semantic_mapping carla_capture_benchmark --host 127.0.0.1 --port 2000 --map Town05 --output "$dataset" --cache-dir /home/yk/ws/.cache/carla --frames "$frames" --save-every "$save_every" --vehicles "$vehicles" --vehicle-classes car truck bus bicycle motorcycle --walkers "$walkers" --width 640 --height 480 --grid-rows 4 --grid-cols 6 --seed "$seed" --stationary-ego` |
| C_IMG_EVAL | `ros2 run semantic_mapping carla_evaluate_benchmark --dataset "$dataset" --output "$output" --backend segformer --segformer-model "$SEGFORMER_MODEL" --segformer-revision "$SEGFORMER_REVISION" --segformer-confidence "$threshold" --device cuda --fp16 --save-overlay-every "$overlay_every"` |
| C_REL_CAP | 计划 §8 的 `carla_capture_reliability` 完整命令；变量 seed/channels/PPS/profile/output 唯一化，Town05/ClearNoon/静态目标/300 帧固定 |
| C_REL_EVAL | `ros2 run semantic_mapping carla_evaluate_reliability --dataset "$dataset" --output "$output" --params-file "$yaml_copy" --segformer-model "$SEGFORMER_MODEL" --segformer-revision "$SEGFORMER_REVISION" --segformer-temperature "$temperature" --time-offset-ms $offsets --frame-tolerance-ms 6 --device cuda --fp16 --posterior-storage none --voxel-eval` |
| C_YAML_COPY | 计划 §9.1 的 PyYAML 命令；从正式 YAML 读取，在 `$AUTOTEST_ROOT/configs/` 写唯一副本并只改 `{field}={value}` |
| C_AGG | `python3 "$AUTOTEST_ROOT/summary/aggregate_results.py" --root "$AUTOTEST_ROOT" ...`，按计划 §11/§12 输出 strict JSON/CSV/MD |

## P0 — sanity 与 baseline

| Run ID | 阶段 | 依赖 | 状态 | 命令/参数 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-BOOT-001 | sanity | 无 | TODO | C_BOOT | `$AUTOTEST_ROOT/{run.env,AUTOTEST_STATUS.json,commands.log}` | 新 timestamp 目录且不覆盖旧结果 |
| P0-A-001 | sanity | P0-BOOT-001 | TODO | C_PREFLIGHT | `preflight/P0-A-001.log` | Git dirty 已记录；CUDA/CARLA/model cache/disk 全通过 |
| P0-A-002 | sanity | P0-A-001 | TODO | C_BUILD_TEST | `logs/P0-A-002-*.log` | build、6 test files、flake8、pep257 exit 0 |
| P0-A-003 | sanity | P0-A-002 | TODO | C_ENTRY | `logs/P0-A-003-entry-help.log` | 四入口 exit 0 |
| P0-B-STA-POINT | baseline | P0-A-003 | TODO | C_REG_POINT，stationary 归档数据 | `reliability/regression_stationary_point/{report.json,factor_summary.csv}` | flow 与归档一致、确定性差≤1e-6 |
| P0-B-CV-POINT | baseline | P0-A-003 | TODO | C_REG_POINT，constant 归档数据 | `reliability/regression_constant_point/*` | 同上 |
| P0-B-TURN-POINT | baseline | P0-A-003 | TODO | C_REG_POINT，turning 归档数据 | `reliability/regression_turning_point/*` | 同上 |
| P0-B-TURN-VOXEL | baseline | P0-B-TURN-POINT | TODO | C_REG_VOXEL，独立 output | `reliability/regression_turning_voxel/*` | Motion V2 机制通过、体素主门仍失败 |
| P0-C-SERVER-IMG | sanity | P0-B-STA-POINT,P0-B-CV-POINT,P0-B-TURN-POINT,P0-B-TURN-VOXEL | TODO | C_SERVER + 0.9.16 connect probe | `preflight/carla_server*.{log,pid}` | client/server 0.9.16，端口归属明确 |
| P0-C-IMG-CAP | sanity | P0-C-SERVER-IMG | TODO | C_IMG_CAP，frames=30/save_every=1/vehicles=12/walkers=18/seed=42 | `image/smoke_seed42/manifest.json` 等 | format-v2、Town05、frames非空 |
| P0-C-SERVER-IMG-STOP | sanity | P0-C-IMG-CAP | TODO | 计划 §6 owned-PID `kill`；非 owned 请求 owner 正常停止，否则 BLOCKED | server退出日志 | 推理前CARLA已释放GPU，未误杀他人进程 |
| P0-C-IMG-EVAL | sanity | P0-C-SERVER-IMG-STOP | TODO | C_IMG_EVAL，threshold=0.45/overlay_every=1 | `image/smoke_seed42/results/segformer_smoke/*` | strict report、per_frame、overlay、exit 0 |
| P0-C-SERVER-REL | sanity | P0-C-IMG-EVAL | TODO | C_SERVER + 0.9.16 connect probe | server log/PID | CARLA重新就绪且GPU无SegFormer任务 |
| P0-C-REL-CAP | sanity | P0-C-SERVER-REL | TODO | 计划 §6 reliability 30-frame 命令 | `reliability/smoke_seed42/manifest.json` 等 | sensor/metadata 完整同帧 |
| P0-C-SERVER-REL-STOP | sanity | P0-C-REL-CAP | TODO | 计划 §6 owned-PID stop | server退出日志 | 推理前释放CARLA GPU |
| P0-C-REL-EVAL | sanity | P0-C-SERVER-REL-STOP | TODO | 计划 §6 evaluator，offsets=0,20,50,100,150 | `reliability/smoke_eval/{report.json,per_point.csv,factor_summary.csv}` | alignment≥0.98、rejected=0、finite metrics |
| P0-D-S11-CAP | baseline | P0-C-REL-EVAL | TODO | C_IMG_CAP，400/5/40 vehicles/25 walkers/seed11 | `image/town05_clear_seed11/` | format-v2、五类可见 |
| P0-D-S42-CAP | baseline | P0-C-REL-EVAL | TODO | C_IMG_CAP，seed42，其余同上 | `image/town05_clear_seed42/` | 同上 |
| P0-D-S73-CAP | baseline | P0-C-REL-EVAL | TODO | C_IMG_CAP，seed73，其余同上 | `image/town05_clear_seed73/` | 同上 |
| P0-D-S11-C035 | baseline | P0-D-S11-CAP | TODO | C_IMG_EVAL，threshold=.35 | `...seed11/results/segformer_c0p35/` | report/per-class/overlay |
| P0-D-S11-C045 | baseline | P0-D-S11-CAP | TODO | C_IMG_EVAL，threshold=.45 | `...seed11/results/segformer_c0p45/` | 同上 |
| P0-D-S11-C055 | baseline | P0-D-S11-CAP | TODO | C_IMG_EVAL，threshold=.55 | `...seed11/results/segformer_c0p55/` | 同上 |
| P0-D-S42-C035 | baseline | P0-D-S42-CAP | TODO | C_IMG_EVAL，threshold=.35 | `...seed42/results/segformer_c0p35/` | 同上 |
| P0-D-S42-C045 | baseline | P0-D-S42-CAP | TODO | C_IMG_EVAL，threshold=.45 | `...seed42/results/segformer_c0p45/` | 同上 |
| P0-D-S42-C055 | baseline | P0-D-S42-CAP | TODO | C_IMG_EVAL，threshold=.55 | `...seed42/results/segformer_c0p55/` | 同上 |
| P0-D-S73-C035 | baseline | P0-D-S73-CAP | TODO | C_IMG_EVAL，threshold=.35 | `...seed73/results/segformer_c0p35/` | 同上 |
| P0-D-S73-C045 | baseline | P0-D-S73-CAP | TODO | C_IMG_EVAL，threshold=.45 | `...seed73/results/segformer_c0p45/` | 同上 |
| P0-D-S73-C055 | baseline | P0-D-S73-CAP | TODO | C_IMG_EVAL，threshold=.55 | `...seed73/results/segformer_c0p55/` | road/vehicle与每细类门逐项判定 |
| P0-E-S11-CAP | baseline | P0-D-S11-C035/P0-D-S11-C045/P0-D-S11-C055/P0-D-S42-C035/P0-D-S42-C045/P0-D-S42-C055/P0-D-S73-C035/P0-D-S73-C045/P0-D-S73-C055 | TODO | C_REL_CAP，64ch/600k/stationary/seed11 | `reliability/static_town05_clear_seed11/` | 300完整帧、manifest事实正确 |
| P0-E-S11-EVAL | baseline | P0-E-S11-CAP | TODO | C_REL_EVAL，正式YAML副本/temp1/offset0 | `reliability/static_eval_seed11/` | alignment、finite metrics、voxel六组 |
| P0-E-S42-CAP | baseline | P0-E-S11-CAP | TODO | C_REL_CAP，seed42 | `reliability/static_town05_clear_seed42/` | 同上 |
| P0-E-S42-EVAL | baseline | P0-E-S42-CAP | TODO | C_REL_EVAL，offset0 | `reliability/static_eval_seed42/` | 同上 |
| P0-E-S73-CAP | baseline | P0-E-S42-CAP | TODO | C_REL_CAP，seed73 | `reliability/static_town05_clear_seed73/` | 同上 |
| P0-E-S73-EVAL | baseline | P0-E-S73-CAP | TODO | C_REL_EVAL，offset0 | `reliability/static_eval_seed73/` | `r_semantic` claim 可统计 |

## P1-F — 单因素 main runs

每个以下 ID 是一个独立 evaluator run，依赖 `P0-E-S<seed>-CAP` 与对应 `C_YAML_COPY`。状态均为 TODO；输出为 `$AUTOTEST_ROOT/reliability/p1f/<ID>/`；验收统一为 strict report + 六组 voxel + one-factor-at-a-time + alignment 硬门。

| Family | 独立 Run IDs | 命令变量 | 依赖 | 状态 | 验收 |
| --- | --- | --- | --- | --- | --- |
| Range | P1-F-RANGE-S11-V10, P1-F-RANGE-S11-V20, P1-F-RANGE-S11-V30, P1-F-RANGE-S42-V10, P1-F-RANGE-S42-V20, P1-F-RANGE-S42-V30, P1-F-RANGE-S73-V10, P1-F-RANGE-S73-V20, P1-F-RANGE-S73-V30 | C_YAML_COPY `range_scale_m=V`; C_REL_EVAL temp=1,offset0 | P0-E 对应 seed | TODO | ≥2/3 seed方向一致；voxel不退化 |
| Density weight | P1-F-DW-S11-V4, P1-F-DW-S11-V8, P1-F-DW-S11-V16, P1-F-DW-S42-V4, P1-F-DW-S42-V8, P1-F-DW-S42-V16, P1-F-DW-S73-V4, P1-F-DW-S73-V8, P1-F-DW-S73-V16 | `density_scale=V` | 同上 | TODO | 同上 |
| View | P1-F-VIEW-S11-V0, P1-F-VIEW-S11-V0P2, P1-F-VIEW-S11-V0P4, P1-F-VIEW-S42-V0, P1-F-VIEW-S42-V0P2, P1-F-VIEW-S42-V0P4, P1-F-VIEW-S73-V0, P1-F-VIEW-S73-V0P2, P1-F-VIEW-S73-V0P4 | `view_edge_penalty=0,.2,.4` | 同上 | TODO | 非零值必须在 voxel 层有证据 |
| Semantic floor | P1-F-SF-S11-V005, P1-F-SF-S11-V015, P1-F-SF-S11-V030, P1-F-SF-S42-V005, P1-F-SF-S42-V015, P1-F-SF-S42-V030, P1-F-SF-S73-V005, P1-F-SF-S73-V015, P1-F-SF-S73-V030 | `semantic_confidence_floor=.05,.15,.30` | 同上 | TODO | AUROC/AURC/gap + voxel联合 |
| Temperature | P1-F-TEMP-S11-V075, P1-F-TEMP-S11-V100, P1-F-TEMP-S11-V125, P1-F-TEMP-S11-V150, P1-F-TEMP-S42-V075, P1-F-TEMP-S42-V100, P1-F-TEMP-S42-V125, P1-F-TEMP-S42-V150, P1-F-TEMP-S73-V075, P1-F-TEMP-S73-V100, P1-F-TEMP-S73-V125, P1-F-TEMP-S73-V150 | 正式YAML副本；C_REL_EVAL `temperature=.75,1,1.25,1.5` | 同上 | TODO | ECE/NLL/Brier与disjoint解释，不宣称自动校准 |

### Density 传感器控制 runs

| Run ID | 依赖 | 状态 | 命令/参数 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- |
| P1-F-D32-S11-CAP / P1-F-D32-S42-CAP / P1-F-D32-S73-CAP | P0-E all PASS | TODO | C_REL_CAP，32ch/300k，各 seed | `reliability/density32_seed<seed>/` | 300帧、其余控制量相同 |
| P1-F-D32-S11-EVAL / P1-F-D32-S42-EVAL / P1-F-D32-S73-EVAL | 对应 CAP | TODO | C_REL_EVAL，temp1/offset0/正式YAML副本 | `reliability/p1f/<ID>/` | alignment与voxel硬门 |
| P1-F-D16-S11-CAP / P1-F-D16-S42-CAP / P1-F-D16-S73-CAP | P0-E all PASS | TODO | C_REL_CAP，16ch/150k，各 seed | `reliability/density16_seed<seed>/` | 300帧、其余控制量相同 |
| P1-F-D16-S11-EVAL / P1-F-D16-S42-EVAL / P1-F-D16-S73-EVAL | 对应 CAP | TODO | C_REL_EVAL，temp1/offset0/正式YAML副本 | `reliability/p1f/<ID>/` | alignment与voxel硬门 |

## P1-G — Motion/offset main runs

| Run ID | 依赖 | 状态 | 命令/参数 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- |
| P1-G-STA-S11-EVAL / P1-G-STA-S42-EVAL / P1-G-STA-S73-EVAL | P0-E对应CAP | TODO | C_REL_EVAL，offsets=`0 20 50 100 150` | `reliability/p1g/<ID>/` | stationary V2不随offset降权 |
| P1-G-CV-S11-CAP / P1-G-CV-S42-CAP / P1-G-CV-S73-CAP | P0-E all PASS | TODO | C_REL_CAP，profile=constant_velocity, speed=5 | `reliability/motion_cv_seed<seed>/` | 300完整帧，静态目标 |
| P1-G-CV-S11-EVAL / P1-G-CV-S42-EVAL / P1-G-CV-S73-EVAL | 对应CAP | TODO | C_REL_EVAL，五offset | `reliability/p1g/<ID>/` | paired common points、pose/V1/V2/voxel完整 |
| P1-G-TURN-S11-CAP / P1-G-TURN-S42-CAP / P1-G-TURN-S73-CAP | P0-E all PASS | TODO | C_REL_CAP，profile=turning, throttle=.45,steer=.35 | `reliability/motion_turn_seed<seed>/` | 300完整帧，静态目标 |
| P1-G-TURN-S11-EVAL / P1-G-TURN-S42-EVAL / P1-G-TURN-S73-EVAL | 对应CAP | TODO | C_REL_EVAL，五offset | `reliability/p1g/<ID>/` | 机制/效果分层；不得调scale凑门 |

## P1-H/I — ablation 与统计

| Run ID | 阶段 | 依赖 | 状态 | 命令 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-H-AGG-001 | ablation | P0-E,P1-F,P1-G 所有已完成 evaluator | TODO | C_AGG，计划 §11 | `summary/P1_H_voxel_ablation.{json,csv}`, `P1_H_decision.md` | 六组同点/同GT；accuracy/coverage/gap/AUROC门逐seed |
| P1-I-BOOTSTRAP-001 | main | P1-H-AGG-001 | TODO | C_AGG，bootstrap seed=20260817,reps=2000 | `summary/FINAL_REPORT.{json,csv,md}` | seed/frame cluster；均值/std/95%CI；≥2/3方向 |

## P2 — HOLD，仅估算

每个格子都有唯一 run ID `P2-J-{TOWN}-{WEATHER}-S{SEED}`：

- `TOWN ∈ {T03,T05,T10}` 对应 Town03/Town05/Town10HD_Opt；
- `WEATHER ∈ {CLEAR,CLOUDY,WET}`；
- `SEED ∈ {11,42,73}`；共 27 个 IDs。

| IDs | 依赖 | 状态 | 命令/输出 | 验收 |
| --- | --- | --- | --- | --- |
| `P2-J-{T03,T05,T10}-{CLEAR,CLOUDY,WET}-S{11,42,73}` | P1-I + 用户新确认 | TODO (HOLD_P2) | 仅预注册：C_REL_CAP 30-frame smoke→300-frame；C_REL_EVAL；输出 `p2/<ID>/` | 当前禁止启动；预算 24–40 GPU·h、35–55GB |
| P2-K-CLIP | P1-I + 用户新确认 | TODO (HOLD_P2) | `carla_evaluate_benchmark --backend clip`，复用image数据 | 不替代SegFormer |
| P2-K-PERF | P1-I + 用户新确认 | TODO (HOLD_P2) | CPU/CUDA、FP32/FP16与长序列压力测试 | CARLA与模型不并发测时 |

## 失败传播

- P0 任一硬门失败：所有未开始 P0/P1 改为 `NOT_RUN`，不得仍保持“自动继续”。
- `FAIL` 代表实验完成但质量不达标；不修阈值、不重采“直到成功”。
- `INVALID` 代表 manifest/GT/类别支持/控制变量不成立；该 run 不进入统计。
- 工具异常最多 2 次有差异的最小修复+复测；无改动重试禁止。
