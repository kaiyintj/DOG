---
tracker_id: carla_aris_w15_tracker_20260817_131318
plan_id: carla_aris_w15_20260817_130318
status: p1_offline_diagnostic_rescue_planned
default_run_status: TODO
created: 2026-08-17T13:13:18+08:00
revised: 2026-08-17T13:54:10+08:00
---

> **2026-08-17 15:49 live crash override：**当前允许的 live runs 与状态唯一来源为 [`CARLA_CRASH_DIAG_TRACKER.md`](CARLA_CRASH_DIAG_TRACKER.md)。旧 tracker 中 `LIVE_CAPTURE_BLOCKED`/“不再启动 CARLA”不阻断这组诊断；不代表恢复原 P0-D/E、P1/P2。

# CARLA ARIS W1.5 Experiment Tracker

固定计划：`refine-logs/EXPERIMENT_PLAN.md`。所有 run 初始为 `TODO`；只有执行模型在写入对应 `run_records/<RUN_ID>.json` 后才能改为 `PASS/FAIL/BLOCKED/INVALID/NOT_RUN`。P2 保持 `TODO (HOLD_P2)`。

门类型：`S`=结构硬门，仅 `BLOCKED`/`INVALID`/命令执行失败沿真实依赖边停止；`Q`=质量门，低于阈值记 `FAIL`/negative/inconclusive，但不得停止后续独立 seed、因素诊断、motion 或消融。`FAIL` 不传播 `NOT_RUN`。

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
| P0-BOOT-001 | sanity | 无 | PASS | C_BOOT | `$AUTOTEST_ROOT/{run.env,AUTOTEST_STATUS.json,commands.log}` | 新 timestamp 目录且不覆盖旧结果 |
| P0-A-001 | sanity | P0-BOOT-001 | PASS | C_PREFLIGHT | `preflight/P0-A-001*.log` | 本机 GPU/CUDA/CARLA/model cache/disk 全通过；隔离 false-negative 已保留 |
| P0-A-002 | sanity | P0-A-001 | PASS | C_BUILD_TEST | `logs/P0-A-002-*.log` | build、73 个定向测试、flake8、pep257 exit 0；SciPy/NumPy warning 仅记录 |
| P0-A-003 | sanity | P0-A-002 | PASS | C_ENTRY | `logs/P0-A-003-entry-help.log` | 四入口 exit 0 |
| P0-B-STA-POINT | baseline | P0-A-003 | PASS | C_REG_POINT，stationary 归档数据 | `reliability/regression_stationary_point/{report.json,factor_summary.csv}` | S/Q 通过；flow一致，最大数值漂移 `1.31e-10` |
| P0-B-CV-POINT | baseline | P0-A-003 | PASS | C_REG_POINT，constant 归档数据 | `reliability/regression_constant_point/*` | S/Q 通过；flow一致，最大数值漂移 `<4e-13` |
| P0-B-TURN-POINT | baseline | P0-A-003 | PASS | C_REG_POINT，turning 归档数据 | `reliability/regression_turning_point/*` | S/Q 通过；flow一致，最大数值漂移 `4.56e-10` |
| P0-B-TURN-VOXEL | baseline | P0-B-TURN-POINT产物结构有效（PASS或Q-FAIL） | PASS | C_REG_VOXEL，独立 output | `reliability/regression_turning_voxel/*` | S/Q 通过；flow一致，最大数值漂移 `1.64e-12`；历史 Motion V2 结论保持 |
| P0-C-SERVER-IMG | sanity | P0-A-003；P0-B仅顺序前置、不构成产物依赖 | PASS | C_SERVER + 0.9.16 connect probe | `preflight/carla_server*.{log,pid}` | client/server 0.9.16、owned PID 明确；后续命令结束时 server 生命周期被清理并留痕 |
| P0-C-IMG-CAP | sanity | P0-C-SERVER-IMG | BLOCKED | C_IMG_CAP，frames=30/save_every=1/vehicles=12/walkers=18/seed=42 | `image/smoke_seed42*/manifest.json` | 原始与两次最小修复均 load_world 超时；R1 sensor-timeout=60 仍 0 帧 |
| P0-C-SERVER-IMG-STOP | sanity | P0-C-IMG-CAP | PASS | 计划 §6 owned-PID `kill`；非 owned 请求 owner 正常停止，否则 BLOCKED | server退出日志 | R1 owned stop exit 0，推理前 GPU 已释放 |
| P0-C-IMG-EVAL | sanity | P0-C-SERVER-IMG-STOP | NOT_RUN | C_IMG_EVAL，threshold=0.45/overlay_every=1 | 无有效 image smoke 输出 | capture 结构门 BLOCKED，未启动 SegFormer |
| P0-C-SERVER-REL | sanity | P0-C-IMG-EVAL | PASS | C_SERVER + 0.9.16 connect probe | server log/PID | reliability 分支 owned server 探针通过，版本 0.9.16 |
| P0-C-REL-CAP | sanity | P0-C-SERVER-REL | BLOCKED | 计划 §6 reliability 30-frame 命令 | `reliability/smoke_seed42_r1/manifest.json` | sensor-timeout=60 后 load_world 仍超时，0 帧；owned stop exit 0 |
| P0-C-SERVER-REL-STOP | sanity | P0-C-REL-CAP | PASS | 计划 §6 owned-PID stop | server退出日志 | owned stop exit 0，推理前释放 CARLA GPU |
| P0-C-REL-EVAL | sanity | P0-C-SERVER-REL-STOP | NOT_RUN | 计划 §6 evaluator，offsets=0,20,50,100,150 | 无有效 reliability smoke 输出 | capture 结构门 BLOCKED，未启动 evaluator |

### P0-C unplanned diagnostic fallback（不替代 Town05 计划结果）

| Run ID | 阶段 | 依赖 | 状态 | 命令/参数 | 输出 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-C-FALLBACK-T10-SERVER-IMG | diagnostic fallback | P0-C-SERVER-IMG | PASS | owned CARLA 0.9.16 + probe；目标 Town10HD_Opt | `preflight/carla_server_fallback_img_owned.pid`, server log | 明确标记 `unplanned_diagnostic_fallback` |
| P0-C-FALLBACK-T10-IMG-CAP | diagnostic fallback | P0-C-FALLBACK-T10-SERVER-IMG | BLOCKED | Town10HD_Opt image 30 帧、walkers=0、sensor-timeout=60 | `image/fallback_town10hd_seed42/manifest.json` | server Signal 11，exit 250，format-v2 但 frames=0；不冒充 Town05 |
| P0-C-FALLBACK-T10-SERVER-IMG-STOP | diagnostic fallback | P0-C-FALLBACK-T10-IMG-CAP | PASS | owned PID identity-checked stop | fallback server log | stop exit 0，推理前释放 GPU |
| P0-C-FALLBACK-T10-IMG-EVAL | diagnostic fallback | P0-C-FALLBACK-T10-SERVER-IMG-STOP | NOT_RUN | SegFormer/CLIP image eval | 无有效输出 | capture 结构门 BLOCKED |
| P0-C-FALLBACK-T10-REL-CAP | diagnostic fallback | P0-C-FALLBACK-T10-IMG-EVAL | NOT_RUN | Town10HD_Opt reliability capture | 无有效输出 | image fallback 失败后按依赖规则不启动第二条 capture |
| P0-C-FALLBACK-T10-REL-EVAL | diagnostic fallback | P0-C-FALLBACK-T10-REL-CAP | NOT_RUN | reliability evaluator | 无有效输出 | reliability capture 未运行 |
| P0-D-S11-CAP | baseline | P0-C-IMG-EVAL结构有效 | NOT_RUN | C_IMG_CAP，400/5/40 vehicles/25 walkers/seed11 | 无输出 | P0-C image capture BLOCKED；live capture 未启动 |
| P0-D-S42-CAP | baseline | P0-C-IMG-EVAL结构有效 | NOT_RUN | C_IMG_CAP，seed42，其余同上 | 无输出 | P0-C image capture BLOCKED；live capture 未启动 |
| P0-D-S73-CAP | baseline | P0-C-IMG-EVAL结构有效 | NOT_RUN | C_IMG_CAP，seed73，其余同上 | 无输出 | P0-C image capture BLOCKED；live capture 未启动 |
| P0-D-S11-C035 | baseline | P0-D-S11-CAP | NOT_RUN | C_IMG_EVAL，threshold=.35 | 无输出 | capture dependency NOT_RUN |
| P0-D-S11-C045 | baseline | P0-D-S11-CAP | NOT_RUN | C_IMG_EVAL，threshold=.45 | 无输出 | capture dependency NOT_RUN |
| P0-D-S11-C055 | baseline | P0-D-S11-CAP | NOT_RUN | C_IMG_EVAL，threshold=.55 | 无输出 | capture dependency NOT_RUN |
| P0-D-S42-C035 | baseline | P0-D-S42-CAP | NOT_RUN | C_IMG_EVAL，threshold=.35 | 无输出 | capture dependency NOT_RUN |
| P0-D-S42-C045 | baseline | P0-D-S42-CAP | NOT_RUN | C_IMG_EVAL，threshold=.45 | 无输出 | capture dependency NOT_RUN |
| P0-D-S42-C055 | baseline | P0-D-S42-CAP | NOT_RUN | C_IMG_EVAL，threshold=.55 | 无输出 | capture dependency NOT_RUN |
| P0-D-S73-C035 | baseline | P0-D-S73-CAP | NOT_RUN | C_IMG_EVAL，threshold=.35 | 无输出 | capture dependency NOT_RUN |
| P0-D-S73-C045 | baseline | P0-D-S73-CAP | NOT_RUN | C_IMG_EVAL，threshold=.45 | 无输出 | capture dependency NOT_RUN |
| P0-D-S73-C055 | baseline | P0-D-S73-CAP | NOT_RUN | C_IMG_EVAL，threshold=.55 | 无输出 | capture dependency NOT_RUN |
| P0-E-S11-CAP | baseline | P0-C-REL-EVAL结构有效；不依赖P0-D质量结果 | NOT_RUN | C_REL_CAP，64ch/600k/stationary/seed11 | 无输出 | P0-C reliability capture BLOCKED；live capture 未启动 |
| P0-E-S11-EVAL | baseline | P0-E-S11-CAP | NOT_RUN | C_REL_EVAL，正式YAML副本/temp1/offset0 | 无输出 | capture dependency NOT_RUN |
| P0-E-S42-CAP | baseline | P0-C-REL-EVAL结构有效 | NOT_RUN | C_REL_CAP，seed42 | 无输出 | P0-C reliability capture BLOCKED；live capture 未启动 |
| P0-E-S42-EVAL | baseline | P0-E-S42-CAP | NOT_RUN | C_REL_EVAL，offset0 | 无输出 | capture dependency NOT_RUN |
| P0-E-S73-CAP | baseline | P0-C-REL-EVAL结构有效 | NOT_RUN | C_REL_CAP，seed73 | 无输出 | P0-C reliability capture BLOCKED；live capture 未启动 |
| P0-E-S73-EVAL | baseline | P0-E-S73-CAP | NOT_RUN | C_REL_EVAL，offset0 | 无输出 | capture dependency NOT_RUN |

## P1-F — 单因素 main runs

每个以下 ID 是一个独立 evaluator run，只依赖对应 `P0-E-S<seed>-CAP` 的结构有效数据与对应 `C_YAML_COPY`；P0-D 或 P0-E 的质量 `FAIL` 不阻断。状态均为 TODO；输出为 `$AUTOTEST_ROOT/reliability/p1f/<ID>/`。S 门是 strict report、同一控制变量、GT alignment 与非空数据；Q 门是各 factor/voxel 效果，Q 失败仍继续其他矩阵单元。

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
| P1-F-D32-S11-CAP / P1-F-D32-S42-CAP / P1-F-D32-S73-CAP | P0-C-REL-EVAL结构有效 | TODO | C_REL_CAP，32ch/300k，各 seed | `reliability/density32_seed<seed>/` | S：300帧、其余控制量相同 |
| P1-F-D32-S11-EVAL / P1-F-D32-S42-EVAL / P1-F-D32-S73-EVAL | 对应 CAP | TODO | C_REL_EVAL，temp1/offset0/正式YAML副本 | `reliability/p1f/<ID>/` | S：alignment/output；Q：voxel效果，失败继续其他密度 |
| P1-F-D16-S11-CAP / P1-F-D16-S42-CAP / P1-F-D16-S73-CAP | P0-C-REL-EVAL结构有效 | TODO | C_REL_CAP，16ch/150k，各 seed | `reliability/density16_seed<seed>/` | S：300帧、其余控制量相同 |
| P1-F-D16-S11-EVAL / P1-F-D16-S42-EVAL / P1-F-D16-S73-EVAL | 对应 CAP | TODO | C_REL_EVAL，temp1/offset0/正式YAML副本 | `reliability/p1f/<ID>/` | S：alignment/output；Q：voxel效果，失败继续其他密度 |

## P1-G — Motion/offset main runs

| Run ID | 依赖 | 状态 | 命令/参数 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- |
| P1-G-STA-S11-EVAL / P1-G-STA-S42-EVAL / P1-G-STA-S73-EVAL | P0-E对应CAP结构有效 | TODO | C_REL_EVAL，offsets=`0 20 50 100 150` | `reliability/p1g/<ID>/` | Q：stationary V2响应；失败记录后继续CV/turning |
| P1-G-CV-S11-CAP / P1-G-CV-S42-CAP / P1-G-CV-S73-CAP | P0-C-REL-EVAL结构有效 | TODO | C_REL_CAP，profile=constant_velocity, speed=5 | `reliability/motion_cv_seed<seed>/` | S：300完整帧，静态目标 |
| P1-G-CV-S11-EVAL / P1-G-CV-S42-EVAL / P1-G-CV-S73-EVAL | 对应CAP | TODO | C_REL_EVAL，五offset | `reliability/p1g/<ID>/` | paired common points、pose/V1/V2/voxel完整 |
| P1-G-TURN-S11-CAP / P1-G-TURN-S42-CAP / P1-G-TURN-S73-CAP | P0-C-REL-EVAL结构有效 | TODO | C_REL_CAP，profile=turning, throttle=.45,steer=.35 | `reliability/motion_turn_seed<seed>/` | S：300完整帧，静态目标 |
| P1-G-TURN-S11-EVAL / P1-G-TURN-S42-EVAL / P1-G-TURN-S73-EVAL | 对应CAP | TODO | C_REL_EVAL，五offset | `reliability/p1g/<ID>/` | 机制/效果分层；不得调scale凑门 |

## P1-H/I — ablation 与统计

| Run ID | 阶段 | 依赖 | 状态 | 命令 | 输出 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-H-AGG-001 | ablation | P0-E/P1-F/P1-G 中所有结构有效 evaluator；允许Q-FAIL和部分seed INVALID | TODO | C_AGG，计划 §11 | `summary/P1_H_voxel_ablation.{json,csv}`, `P1_H_decision.md` | S：六组同点/同GT；Q：accuracy/coverage/gap/AUROC，失败仍继续P1-I |
| P1-I-BOOTSTRAP-001 | main | P1-H-AGG-001结构化产物有效（PASS或Q-FAIL） | TODO | C_AGG，bootstrap seed=20260817,reps=2000 | `summary/FINAL_REPORT.{json,csv,md}` | 对全部有效正/负结果做cluster CI；不足明确inconclusive |

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

- `FAIL` 是结构有效但质量不达标的负结果；不传播停止，不修阈值、不重采“直到成功”，继续所有后续独立诊断/消融并纳入负结果汇总。
- 只有 `BLOCKED`、`INVALID` 或命令/入口执行失败沿真实依赖边把 run 改为 `NOT_RUN`。不得因为一个质量 claim 失败取消整个 P0/P1。
- 一个 seed/dataset 的 manifest/schema/GT/control 无效，只阻断消费该产物的 runs；其他 seed、image/reliability 独立分支继续。
- `INVALID` run 不进入数值聚合，但必须进入最终状态/缺失性报告；有效的 `FAIL` run 必须进入聚合。
- 工具异常最多 2 次有差异的最小修复+复测；无改动重试禁止。

## P1-OFFLINE-DIAGNOSTIC rescue override

P0-A/B 已 `PASS`，P0-C live capture 在 Town05 与 Town10HD_Opt 均因 CARLA server Signal 11 结构阻断。此前 P0-D/E 与 P1-F/G/H/I 中依赖新 capture 的 TODO 项现在统一解释为 `NOT_RUN (LIVE_CAPTURE_BLOCKED)`；不再启动 CARLA。下列 `P1-OD-*` 是唯一获准继续的 rescue runs，全部使用历史 Town10HD_Opt/ClearNoon/seed42 数据，不能冒充新采集、多 seed 或正式参数冻结。

### Setup 与 point-only OFAT

| Run ID | 依赖 | 状态 | 参数/数据 | 命令 | 输出 | S/Q 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-OD-SETUP-001 | P0-A/B PASS | PASS | 三 manifest；YAML副本 | 计划 §16.2 | `configs/p1_offline_diagnostic/*.yaml`, run record | S 通过：100/100/60帧、Town10/seed42/schema/模型可读 |
| P1-OD-PT-RANGE-10 | SETUP | PASS | stationary；range=10,temp=1,offset0 | §16.3 point command | `p1_offline_diagnostic/P1-OD-PT-RANGE-10/` | S 通过；Q 诊断记录 |
| P1-OD-PT-RANGE-30 | SETUP | PASS | stationary；range=30,temp=1,offset0 | 同上 | `.../P1-OD-PT-RANGE-30/` | S 通过；worth-later-retest 仅 single-seed diagnostic |
| P1-OD-PT-DENSITY-4 | SETUP | PASS | stationary；density=4,temp=1,offset0 | 同上 | `.../P1-OD-PT-DENSITY-4/` | S 通过；nonphysical density-weight diagnostic |
| P1-OD-PT-DENSITY-16 | SETUP | PASS | stationary；density=16,temp=1,offset0 | 同上 | `.../P1-OD-PT-DENSITY-16/` | S 通过；Q 负向诊断不传播 |
| P1-OD-PT-VIEW-0P2 | SETUP | PASS | stationary；view=.2,temp=1,offset0 | 同上 | `.../P1-OD-PT-VIEW-0P2/` | S 通过；view signal 退化诊断 |
| P1-OD-PT-VIEW-0P4 | SETUP | PASS | stationary；view=.4,temp=1,offset0 | 同上 | `.../P1-OD-PT-VIEW-0P4/` | S 通过；view signal 退化诊断 |
| P1-OD-PT-SEMFLOOR-0P05 | SETUP | PASS | stationary；floor=.05,temp=1,offset0 | 同上 | `.../P1-OD-PT-SEMFLOOR-0P05/` | S 通过；worth-later-retest 仅 single-seed diagnostic |
| P1-OD-PT-SEMFLOOR-0P30 | SETUP | PASS | stationary；floor=.30,temp=1,offset0 | 同上 | `.../P1-OD-PT-SEMFLOOR-0P30/` | S 通过；Q 负向诊断不传播 |
| P1-OD-PT-TEMP-0P75 | SETUP | PASS | stationary；base YAML,temp=.75,offset0 | 同上 | `.../P1-OD-PT-TEMP-0P75/` | S 通过；不支持单 seed calibration claim |
| P1-OD-PT-TEMP-1P50 | SETUP | PASS | stationary；base YAML,temp=1.5,offset0 | 同上 | `.../P1-OD-PT-TEMP-1P50/` | S 通过；不支持单 seed calibration claim |

默认 range20/density8/view0/floor.15/temp1 已有 P0-B stationary 结果，不创建重复 run。

### Motion 复用与 representative voxel

| Run ID | 依赖 | 状态 | 参数/数据 | 命令 | 输出 | S/Q 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-OD-MOTION-SUMMARY-001 | 四个 P0-B reports PASS | PASS | stationary/constant/turning，既有 offsets 0/20/50/100/150 | 计划 §16.4，纯解析不推理；superseding paired-key correction | `summary/p1_offline_diagnostic/motion_v1_v2_by_offset.{json,csv,md}` | S 通过；(lidar_frame,point_index) supported common counts 24459/23472/13820；`matched_projected_points` 单列，不作 paired count；旧输出保留 `.pre_pairing_correction` |
| P1-OD-VX-STA-DEFAULT | SETUP | PASS | stationary；base,temp1,offsets 0/150 | §16.5 voxel command | `.../P1-OD-VX-STA-DEFAULT/` | S 通过：六组齐；Q 诊断 |
| P1-OD-VX-CV-DEFAULT | SETUP | PASS | constant；base,temp1,offsets 0/150 | 同上 | `.../P1-OD-VX-CV-DEFAULT/` | S 通过：六组齐；Q 诊断 |
| P1-OD-VX-RANGE-30 | P1-OD-PT-RANGE-30结构有效 | PASS | stationary；range30,temp1,offset0 | 同上 | `.../P1-OD-VX-RANGE-30/` | S 通过；coverage/accuracy/gap/AUROC delta 已汇总 |
| P1-OD-VX-DENSITY-4 | P1-OD-PT-DENSITY-4结构有效 | PASS | stationary；density4,temp1,offset0 | 同上 | `.../P1-OD-VX-DENSITY-4/` | S 通过；非物理 density-weight claim 限制 |
| P1-OD-VX-VIEW-0P4 | P1-OD-PT-VIEW-0P4结构有效 | PASS | stationary；view.4,temp1,offset0 | 同上 | `.../P1-OD-VX-VIEW-0P4/` | S 通过；Q 诊断 |
| P1-OD-VX-SEMFLOOR-0P30 | P1-OD-PT-SEMFLOOR-0P30结构有效 | PASS | stationary；floor.30,temp1,offset0 | 同上 | `.../P1-OD-VX-SEMFLOOR-0P30/` | S 通过；Q 诊断 |
| P1-OD-VX-TEMP-0P75 | P1-OD-PT-TEMP-0P75结构有效 | PASS | stationary；base,temp.75,offset0 | 同上 | `.../P1-OD-VX-TEMP-0P75/` | S 通过；Q 诊断 |
| P1-OD-SUMMARY-001 | 所有已完成结构有效 P1-OD runs；允许Q-FAIL | PASS | 全部正/负结果；superseding paired-motion summary | 计划 §16.6，纯离线重生成 | `summary/p1_offline_diagnostic/{run_metrics.csv,ofat_point_comparison.json,voxel_comparison.json,P1_OFFLINE_DIAGNOSTIC_REPORT.md}` | S 通过；62 canonical = 34 PASS + 5 BLOCKED + 23 NOT_RUN；39 attempted；`run_metrics.csv` 19 input rows（不含 summary自身）；代表性 full AUROC 全部低于 none，semantic-only gap/AUROC 小幅正向 |

### Rescue 传播与停止

- live CARLA 永久保持 `BLOCKED/NOT_RUN`，本分支不得改变 P0-C/D/E 状态。
- `FAIL` 只代表质量负结果，继续所有独立 `P1-OD-*` 并进入汇总。
- `BLOCKED`/`INVALID`/执行失败只阻断消费同一数据或同一参数产物的 run；其他 profile/参数继续。
- turning/default voxel 已由 P0-B 完成，不重复；Motion summary 只解析既有结果。
- 累计 wall time 达 60 分钟时，尚未开始的五个 factor voxel 可记 `NOT_RUN (RUNTIME_CAP)`，随后仍执行 P1-OD-SUMMARY-001。
