---
tracker_id: carla_0916_signal11_diag_tracker_20260817_172827
plan_id: carla_0916_signal11_diag_20260817_154949
status: complete
created: 2026-08-17T17:28:27+08:00
root: /home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401
---

# CARLA 0.9.16 Signal 11 Diagnostic Tracker — final

| Run ID | Port | Status | Active phase | Active crash | Shutdown crash | Notes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| D00-BASE-T10-TICK | 21020 | PASS | PASS | 0 | 1 | Town10 boot/tick |
| D01-LOAD-T05-TICK | 21010 | PASS | PASS | 0 | 1 | Town05 load/tick |
| D02-NORENDER-FULL | 21050 | PASS | PASS | 0 | 1 | no-render full static actors |
| D03-RGB-EGO | 21030 | PASS | PASS | 0 | 1 | RGB 30/30 |
| D04-SEM-EGO | 21040 | PASS | PASS | 0 | 1 | semantic 30/30 |
| D05-INSTANCE-EGO | 21060 | PASS | PASS | 0 | 1 | instance 30/30 |
| D06-3CAM-EGO | 21070 | PASS | PASS | 0 | 1 | three-camera 30 bundles |
| D07-3CAM-CARS | 21080 | PASS | PASS | 0 | 1 | cars |
| D08-3CAM-2WHEEL | 21090 | PASS | PASS | 0 | 1 | two-wheel |
| D09-FULL-DEFAULT | 21100 | PASS | PASS | 0 | 1 | 320x240 full |
| D10-FULL-NORHI | 21110 | PASS | PASS | 0 | 1 | `-norhithread`, no mitigation claim |
| D11-FULL-FRESH-SETTINGS | 21130 | PASS | PASS | 0 | 1 | settings hash restored |
| D12-HISTORICAL-LOAD | 21140 | PASS | PASS | 0 | 1 | 640x480/warmup40/12 actors/60 measured |
| D13-HISTORICAL-REPEAT | 21150 | PASS | PASS | 0 | 1 | repeat |
| D14-PROJECT-IMAGE-SMOKE | 21160 | BLOCKED | not entered | 0 | 1 | harness-invalid TM/output attempts retained |
| D15-PROJECT-IMAGE-SMOKE-CORRECT-PORT | 21160 | FAIL | ACTIVE_CRASH | 1 | 0 | Town05 skeletal active crash |
| D16-PROJECT-NO-WALKERS | 21170 | FAIL | ACTIVE_CRASH | 1 | 0 | walkers=0, same symbol |
| D17-PROJECT-WALKERS-ONLY | 21180 | FAIL | ACTIVE_CRASH | 1 | 0 | vehicles=0, same symbol |
| D18-AUTOPILOT-4W | 21180 | FAIL | ACTIVE_CRASH | 1 | 0 | four-wheel autopilot |
| D19-AUTOPILOT-2W | 21190 | PASS | PASS | 0 | 1 | 30-frame manifest; shutdown crash only |

Final aggregate: 4 active crash directories, 16 shutdown crash directories
(all with the `VulkanRHI::FMemoryManager::DumpMemory` stop-boundary stack),
zero strict NVIDIA `NVRM: Xid`, zero NVIDIA OOM/reset matches. The active
Town05 skeletal crash remains reproducible in D15–D18; active root cause and
global remediation are inconclusive. D18 and D19 both use autopilot
(`stationary_ego=false`) and are internally comparable; D16/D17 use stationary
ego. All records and artifacts are retained;
no more CARLA runs are planned.
