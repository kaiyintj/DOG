# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.
> This is a historical audit ledger, not the current project-status source. For current state use
> `docs/PROJECT_STATUS.md`; for current commands use `docs/RUNBOOK.md`; for current Lite3 work use
> `docs/LITE3_REAL_HANDOFF.md`; for current ARIS execution use the un-timestamped plan/tracker in
> `refine-logs/`. Historical entries and old environment paths below must not override those files.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-17 13:03 | /experiment-bridge | AGENTS.md | implementation | Auto-Claude/ARIS 本地单 GPU 执行约束与模型配置 |
| 2026-08-17 13:03 | /experiment-bridge | idea-stage/docs/research_contract.md | implementation | simulation-only、CARLA GT 边界与可证伪 claims |
| 2026-08-17 13:03 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260817_130318.md | implementation | P0→P1 CARLA sanity/baseline/main/ablation 可执行计划 |
| 2026-08-17 13:03 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest copy |
| 2026-08-17 13:03 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260817_130318.md | implementation | 唯一 run ID、依赖、TODO 状态、命令、输出与验收 |
| 2026-08-17 13:03 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest copy |
| 2026-08-17 13:03 | /experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW.md | implementation | CARLA 源码/计划预部署阻断审查与修复边界 |
| 2026-08-17 13:13 | /experiment-bridge | AGENTS.md | implementation | 修订：结构硬门可阻断真实依赖；质量 FAIL 继续独立诊断与消融 |
| 2026-08-17 13:13 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260817_131318.md | implementation | 修订版：拆分 structural/quality gates，负结果不停止自主测试 |
| 2026-08-17 13:13 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest copy：同步 structural/quality gate 修订 |
| 2026-08-17 13:13 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260817_131318.md | implementation | 修订版：按真实依赖传播 BLOCKED/INVALID，FAIL 不传播 |
| 2026-08-17 13:13 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest copy：同步失败传播修订 |
| 2026-08-17 13:23 | /experiment-bridge | .aris/compute/env_spec.yaml | execution | 本机 GPU/CUDA/CARLA/模型缓存声明式环境账本与 seeded CUDA witness |
| 2026-08-17 13:23 | /experiment-bridge | .aris/compute/local.md | execution | 本机实际版本、RTX 5070、模型 revision、dirty 基线与证据路径 |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_20260817_134900.json` | execution | P0-A/B 真实结果、P0-C Town05/诊断 fallback 状态与未完成项 |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_20260817_134900.md` | execution | timestamped initial results summary（机器事实来源于 run_records/report/manifest） |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_latest.json` | execution | latest initial results summary pointer |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_latest.md` | execution | latest initial results summary pointer |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/AUTOTEST_STATUS.json` | execution | P0-A/B PASS、P0-C capture BLOCKED、fallback BLOCKED 的真实状态 |
| 2026-08-17 13:50 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/P0_B_regression_comparison.json` | execution | 四个旧数据回归的真实 flow/数值漂移比较 |
| 2026-08-17 13:52 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_20260817_135300.json` | execution | 更新：含 Town05/Town10 crash evidence、P0-D/E NOT_RUN run records |
| 2026-08-17 13:52 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_20260817_135300.md` | execution | 更新 timestamped/latest 初始结果摘要 |
| 2026-08-17 14:00 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/setup_validation.json` | execution | P1-OD setup：三历史 manifest、模型 revision、YAML 副本与 SHA-256 |
| 2026-08-17 14:03 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/reliability/p1_offline_diagnostic/P1-OD-PT-*` | execution | 10 个 single-seed historical point-only OFAT reports/CSV |
| 2026-08-17 14:13 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/reliability/p1_offline_diagnostic/P1-OD-VX-*` | execution | 7 个 representative voxel reports/CSV，六组结构校验 |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/motion_v1_v2_by_offset.json` | execution | P0-B 三 profile 五 offset 纯解析 Motion V1/V2 summary |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/motion_v1_v2_by_offset.csv` | execution | machine-readable motion offset metrics |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/P1_OFFLINE_DIAGNOSTIC_REPORT.md` | execution | P1-OD strict final report：single-seed diagnostic ranking与claim限制 |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/run_metrics.csv` | execution | P1-OD all-run status/exit/output/metric index |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/ofat_point_comparison.json` | execution | 10 OFAT points vs P0-B offset-0 baseline |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/voxel_comparison.json` | execution | 7 voxel runs六组 delta vs none |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/build_motion_summary.py` | execution | 可审计的 P0-B motion summary parser |
| 2026-08-17 14:18 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/build_p1_summary.py` | execution | 可审计的 P1-OD strict aggregate builder |
| 2026-08-17 13:54 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260817_135410.md | implementation | Rescue：CARLA live BLOCKED 后的 single-seed historical P1 offline diagnostic 增补 |
| 2026-08-17 13:54 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest copy：禁止继续 live CARLA，启用有限离线 OFAT/motion/voxel 分支 |
| 2026-08-17 13:54 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260817_135410.md | implementation | Rescue tracker：唯一 P1-OD run IDs、参数、依赖、S/Q 门和 60 分钟上限 |
| 2026-08-17 13:54 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest copy：同步 P1-OFFLINE-DIAGNOSTIC 执行清单 |
| 2026-08-17 13:54 | /experiment-bridge | refine-logs/P1_OFFLINE_DIAGNOSTIC_TRACE_20260817_135410.md | implementation | Rescue planning evidence、选择理由、claim 边界与产物协议；无时间戳 alias 与本文件逐字节相同，已在 2026-08-21 文档整理中移出当前树，可由 `d27c103` 恢复 |
| 2026-08-17 14:39 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/motion_v1_v2_by_offset.{json,csv,md}` | execution | superseding paired common `(lidar_frame,point_index)` correction；supported counts stationary 24459、constant_velocity 23472、turning 13820；JSON SHA-256 `f4dd66c9a6fd3ee73cbea9dac7d1c539baf9edf3663fce4cb1e4186708726a8c`；prior outputs retained as `.pre_pairing_correction` |
| 2026-08-17 14:39 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/{run_metrics.csv,ofat_point_comparison.json,voxel_comparison.json,P1_OFFLINE_DIAGNOSTIC_REPORT.md}` | execution | corrected P1 offline final summary；19 input rows、summary record excluded；canonical ledger 62 = 34 PASS + 5 BLOCKED + 23 NOT_RUN；39 attempted；representative full AUROC all lower than none，semantic-only gap/AUROC small positive；report SHA-256 `b271a1c21ff240f699961ef7d362300aa0e4ebfa209df30503e60c538b8c4102` |
| 2026-08-17 14:39 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/p1_offline_diagnostic/*pre_pairing_correction` | execution | immutable pre-correction artifacts retained for audit；correction records reference old/new SHA-256 |
| 2026-08-17 14:39 | /experiment-bridge | `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/{AUTOTEST_STATUS.json,AUTOTEST_STATUS.md,commands.log}` | execution | state synchronization after offline correction；live CARLA remains BLOCKED/NOT_RUN |
| 2026-08-17 15:49 | /experiment-bridge | refine-logs/CARLA_CRASH_DIAG_PLAN_20260817_154949.md | implementation | 用户授权的 CARLA 0.9.16 Signal 11 live override；12 个 OFAT、active/shutdown crash 分流、owned PID 与可恢复 settings 协议 |
| 2026-08-17 15:49 | /experiment-bridge | refine-logs/CARLA_CRASH_DIAG_PLAN.md | implementation | latest pointer：当前 CARLA crash diagnosis 计划 |
| 2026-08-17 15:49 | /experiment-bridge | refine-logs/CARLA_CRASH_DIAG_TRACKER_20260817_154949.md | implementation | D00–D11 + strict summary tracker；唯一端口、依赖、状态与验收 |
| 2026-08-17 15:49 | /experiment-bridge | refine-logs/CARLA_CRASH_DIAG_TRACKER.md | implementation | latest pointer：当前 CARLA crash tracker |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401/summary/diagnosis.json` | execution | D00–D19 strict aggregate：4 active Town05 skeletal crashes、16 stop-boundary Vulkan shutdown records、strict NVIDIA Xid/OOM=0 |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401/summary/DIAGNOSIS.md` | execution | final supported/inconclusive diagnosis；D14 harness blocked、D15–D18 active crash、D19 30-frame project PASS |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401/summary/matrix.csv` | execution | machine-readable D00–D19 status/ports/active-stop boundary/frames/actor counts |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `refine-logs/CARLA_CRASH_DIAG_EXECUTION_20260817_172827.md` | execution | timestamped live execution trace with evidence boundaries and final cleanup |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `refine-logs/CARLA_CRASH_DIAG_EXECUTION_LATEST.md` | execution | latest execution trace pointer |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `refine-logs/CARLA_CRASH_DIAG_TRACKER_20260817_172827.md` | execution | final D00–D19 tracker; all independent runs retained |
| 2026-08-17 17:28 | ARIS CARLA crash-diagnosis executor | `refine-logs/CARLA_CRASH_DIAG_TRACKER.md` | execution | latest tracker pointer |
| 2026-08-17 17:29 | ARIS CARLA crash-diagnosis executor | `/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401/summary/final_cleanup.json` | execution | final read-only cleanup: no CARLA executable, all run/TM localhost ports closed, settings hash restored; ss permission limitation recorded |
