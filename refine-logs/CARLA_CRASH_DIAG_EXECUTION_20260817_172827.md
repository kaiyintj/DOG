---
execution_id: carla_0916_signal11_diag_execution_20260817_172827
plan_id: carla_0916_signal11_diag_20260817_154949
status: complete
started: 2026-08-17T16:03:42+08:00
ended: 2026-08-17T17:28:27+08:00
root: /home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401
carla: /home/yk/ws/third_party/CARLA_0.9.16
---

# CARLA 0.9.16 crash-diagnosis execution trace

This trace is the execution record for the authorized P0/P1 live diagnosis. It
uses the Sol plan `CARLA_CRASH_DIAG_PLAN_20260817_154949.md`; the earlier
planned tracker is retained unchanged. No formal YAML, project source, driver,
CARLA installation, or non-owned process was modified. No commit or push was
performed, and no P2 workload was run.

## Baseline and controls

- Unique run root: `/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_155401`.
- CARLA 0.9.16 binary SHA-256: `03bcd413615fa1fc61a5b846342dbdc4e6b3541320a40d6bf17ff927039731f9`.
- Git dirty status and diff SHA-256 were captured in `run.env`/`harness/baseline.json`; existing dirty files were preserved.
- GPU: RTX 5070 Laptop, driver `nvidia-open 580.173.02`; explicit NVIDIA ICD/env is recorded in `run.env`.
- RHI remained Vulkan; `-opengl` was not attempted.
- Original `GameUserSettings.ini` SHA-256 `0e23feac25b539a9875eff20d6f681893a3fff02df7a1faa652c5fe266304448` was restored unconditionally after D11; final hash matches.
- Every server used a fresh port and an owned `setsid` process group. D14's three invalid/harness-blocked attempts are retained, not counted as active workload failures.

## Final run matrix

| Run | Port / TM | Result | Active phase | New crash | Main evidence |
|---|---:|---|---|---|---|
| D00 | 21020 / — | PASS | PASS | shutdown Vulkan | Town10 boot/get_world + 120 ticks |
| D01 | 21010 / — | PASS | PASS | shutdown Vulkan | Town05 load + 120 ticks |
| D02 | 21050 / — | PASS | PASS | shutdown Vulkan | no-rendering, full static actors, 120 ticks |
| D03 | 21030 / — | PASS | PASS | shutdown Vulkan | RGB 30/30 frames |
| D04 | 21040 / — | PASS | PASS | shutdown Vulkan | semantic 30/30 frames; corrected stop-boundary classification retained |
| D05 | 21060 / — | PASS | PASS | shutdown Vulkan | instance 30/30 frames |
| D06 | 21070 / — | PASS | PASS | shutdown Vulkan | three cameras, 30 bundles |
| D07 | 21080 / — | PASS | PASS | shutdown Vulkan | three cameras + cars, 30 bundles |
| D08 | 21090 / — | PASS | PASS | shutdown Vulkan | three cameras + two-wheel, 30 bundles |
| D09 | 21100 / — | PASS | PASS | shutdown Vulkan | full 320x240 static workload |
| D10 | 21110 / — | PASS | PASS | shutdown Vulkan | D09 workload + `-norhithread` |
| D11 | 21130 / — | PASS | PASS | shutdown Vulkan | temporary settings rebuild; hash restored |
| D12 | 21140 / — | PASS | PASS | shutdown Vulkan | historical 640x480, warmup40, 12 actors, 60 measured bundles |
| D13 | 21150 / — | PASS | PASS | shutdown Vulkan | D12 repeat, same result |
| D14 | 21160 / 21161,21162 | BLOCKED | not entered | shutdown Vulkan | TM port and non-empty-output harness faults; all 3 attempts retained |
| D15 | 21160 / 22000 | FAIL | ACTIVE_CRASH | `FSkeletalMeshSceneProxy::GetMeshElementsConditionallySelectable` | real ROS2 entry, 0 frames before crash |
| D16 | 21170 / 22010 | FAIL | ACTIVE_CRASH | same skeletal symbol | real entry with walkers=0; 12 vehicles |
| D17 | 21180 / 22020 | FAIL | ACTIVE_CRASH | same skeletal symbol | real entry with vehicles=0; 13 walkers spawned |
| D18 | 21180 / 22020 | FAIL | ACTIVE_CRASH | same skeletal symbol | autopilot, 12 four-wheel vehicles |
| D19 | 21190 / 22030 | PASS | PASS | shutdown Vulkan | autopilot, two-wheel only; format-v2 manifest, 30 frames |

## Evidence summary

- Four active crashes were observed, all in the real ROS2 entrypoint runs
  D15–D18 and all with the Town05 skeletal render-thread symbol. This supports
  that the historical Town05 active path is still reproducible in the tested
  project workload; it does not establish a complete root cause or a global
  fix.
- D16 shows walkers are not necessary in this tested workload. D17 also reaches
  the same symbol with vehicles disabled, so the actor-family split is not a
  unique isolation of the fault.
- D18 failed for the four-wheel autopilot subset while D19 two-wheel autopilot
  captured all 30 frames. Both D18 and D19 used `stationary_ego=false` and
  autopilot, so that pair is internally comparable; D16/D17 used stationary
  ego and are not direct controls for the pair. This is local evidence toward
  the tested class or interaction path, not proof of a particular mesh or
  autopilot root cause.
- D00–D13 custom probe and historical-load runs completed their active phases;
  D09, D10, D11 all passed, so neither `-norhithread` nor settings rebuild is
  demonstrated as a mitigation.
- Sixteen stop-boundary crash directories were recorded, all with the
  `VulkanRHI::FMemoryManager::DumpMemory` shutdown stack. D19's only new crash was emitted at the owned stop boundary and is classified
  `SHUTDOWN_CRASH`; the repeated `VulkanRHI::FMemoryManager::DumpMemory`
  stack is shutdown evidence, never an active capture root cause.
- Strict kernel filtering matched zero `NVRM: Xid` and zero NVIDIA OOM/reset
  lines in this run. The unrelated network `XID 541` text remains in raw
  journal artifacts but is excluded from `summary/gpu_xid_summary.json`.

## Finalization

All 20 run records, server/client logs, copied `Diagnostics.txt` and
`CrashContext.runtime-xml`, GPU samples, raw journals, image manifests, and
summary artifacts are retained under the unique run root. The final CARLA
process check found no actual `CarlaUE4`/`carla_server` process. The unprivileged
`ss -ltnp` check reported netlink permission unavailable; run ports were also
checked with localhost connection probes and no run-owned listener remained.
No further CARLA test is authorized or required for this execution.
