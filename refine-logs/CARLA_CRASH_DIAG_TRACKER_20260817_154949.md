---
tracker_id: carla_0916_signal11_diag_tracker_20260817_154949
plan_id: carla_0916_signal11_diag_20260817_154949
status: planned
default_run_status: TODO
created: 2026-08-17T15:49:49+08:00
---

# CARLA 0.9.16 Signal 11 Diagnostic Tracker

固定计划：`refine-logs/CARLA_CRASH_DIAG_PLAN.md`。执行根：`/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_154949`。只有执行模型在对应 `run_records/<RUN_ID>.json` 落盘后才能修改状态。

| Run ID | 依赖 | 状态 | Port | 唯一变化 | 必需输出 | 判定 |
| --- | --- | --- | ---: | --- | --- | --- |
| D00-BASE-T10-TICK | preflight | TODO | 21000 | Town10，无actor/sensor | record, server/client/gpu/kernel/crash diff | 120 ticks；active crash=FAIL |
| D01-LOAD-T05-TICK | D00结构完成 | TODO | 21010 | load Town05 | 同上 | load+120 ticks |
| D02-NORENDER-FULL | D01进入Town05；否则显式T10 fallback | TODO | 21020 | no-rendering，full actors+3cam | 同上 | 120 ticks；不验 camera 内容 |
| D03-RGB-EGO | D01或fallback | TODO | 21030 | RGB only | record + frame timestamps | 30 frames |
| D04-SEM-EGO | 同上 | TODO | 21040 | semantic only | 同上 | 30 frames |
| D05-INSTANCE-EGO | 同上 | TODO | 21050 | instance only | 同上 | 30 frames |
| D06-3CAM-EGO | D03–D05均结构完成，允许FAIL | TODO | 21060 | 三 camera | record + bundle timestamps | 30 bundles |
| D07-3CAM-CARS | D06结构完成，允许FAIL | TODO | 21070 | +4 cars | record + blueprint/actor ids | 30 bundles |
| D08-3CAM-2WHEEL | D06结构完成，允许FAIL | TODO | 21080 | +2 bicycle/+2 motorcycle | 同上 | 30 bundles |
| D09-FULL-DEFAULT | D06结构完成，允许D07/D08 FAIL | TODO | 21090 | full default reproducer | record + exact classes/counts | 30 bundles或active crash |
| D10-FULL-NORHI | D09结构完成，允许FAIL | TODO | 21100 | 仅 `-norhithread` | 与D09同schema | 配对解释 |
| D11-FULL-FRESH-SETTINGS | D09结构完成；settings可恢复 | TODO | 21110 | 仅 fresh settings | backup/new/restored hashes + record | 配对解释且恢复hash一致 |
| D12-SUMMARY | 所有已运行records | TODO | — | strict aggregate | `summary/crash_matrix.*`, report/status | active/shutdown分离、控制变量可审计 |

## Preflight checklist

- [ ] 新建唯一 `$CRASH_ROOT`，保存 git dirty 基线；不改/清理用户文件。
- [ ] 记录 GPU/Vulkan/ICD/CARLA hash/driver/open-kernel-module/RAM/swap。
- [ ] 记录原始 crash-dir 清单和 `GameUserSettings.ini` hash/backup。
- [ ] 确认 21000–21112 计划端口在各 run 启动时空闲。
- [ ] harness 只写 `$CRASH_ROOT/harness/`；每 run 独立目录与 owned PID identity。
- [ ] `SDL_VIDEODRIVER` unset；禁止 `-opengl`。
- [ ] 设置每 run readiness 60 s、client 30 s、active 120 s timeout。

## Stop / crash classification checklist（每 run）

- [ ] 发 stop 前已写 `active_phase_completed`、server alive、last tick/frame 和 `stop_signal_epoch`。
- [ ] 只停止 run record 中、cmdline/start-time/port 一致的 owned PID；未使用 `pkill`/`killall`。
- [ ] 新 crash dir 已按 timestamp/PID 分为 ACTIVE/SHUTDOWN/UNKNOWN；UNKNOWN 不参与根因结论。
- [ ] 保存 GPU peak 与 kernel Xid/OOM 查询可用性及原始输出。
- [ ] shutdown crash 不改变 active PASS/FAIL，只写独立字段。

## 失败传播

- active `Signal 11` 是有价值的 `FAIL`，不阻断独立 OFAT。
- `BLOCKED/INVALID` 仅沿真实依赖传播；Town05 load 阻断时统一标注 fallback 到既有 Town10，不静默混图。
- Xid/GPU reset、连续两个不可确认遗留进程、settings无法保证恢复才触发全局停止。
- D11 cleanup/restoration失败时，立即停止并向用户报告原始/backup路径与hash；不得继续启动 CARLA。

