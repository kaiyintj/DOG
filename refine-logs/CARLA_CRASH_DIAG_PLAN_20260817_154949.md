---
plan_id: carla_0916_signal11_diag_20260817_154949
workflow: ARIS_W1.5_experiment_bridge_crash_override
status: planned
created: 2026-08-17T15:49:49+08:00
planning_model: gpt-5.6-sol
execution_model: gpt-5.6-luna
execution_reasoning_effort: max
max_parallel_runs: 1
scope: CARLA_0.9.16_live_crash_diagnosis_only
supersedes: refine-logs/EXPERIMENT_PLAN_20260817_135410.md#live-carla-prohibition
---

# CARLA 0.9.16 Signal 11 自主诊断 override

## 1. 目的、授权与边界

本计划只解决本机 CARLA 0.9.16/UE4.26.2 live capture 的 `Signal 11`。用户已明确授权恢复 live 诊断，因此本文件**仅在本轮 crash diagnosis 范围内**覆盖旧 rescue 计划中的“不得再次启动 CARLA”；不恢复 P0-D/E、P1/P2 大矩阵，也不授权改驱动、重装 CARLA、改系统包、改 GT、改正式 YAML、提交或推送。

已知环境：Ubuntu 22.04、RTX 5070 Laptop、`nvidia-open 580.173.02`、显式 NVIDIA Vulkan ICD。已排除的粗粒度解释：两个 active crash 均 `bIsOOM=0`，崩溃窗口无 NVIDIA Xid，内存/显存没有 OOM 证据。

必须区分三类事件：

1. **ACTIVE_RENDER_CRASH**：在启动、load world、tick 或传感器取帧阶段，停止信号发出前发生；才用于定位 capture 根因。
2. **SHUTDOWN_CRASH**：active phase 已判定完成、owned stop 发出后出现。已见 `VulkanRHI::FMemoryManager::DumpMemory`/`RHIExit` 栈；单独记录，不得把它倒灌成 capture 失败。
3. **HARNESS_FAILURE**：端口、探针、脚本或日志结构失败，但 CARLA 本身没有 active crash；按结构问题处理。

真实 active 栈先验：

- Town05：`FSkeletalMeshSceneProxy::GetMeshElementsConditionallySelectable` → scene capture/render thread；
- Town10：`FPackedUniformBuffers::Init` → Vulkan pipeline/RHI thread；
- 停服：`FVulkanResourceHeap::~` / `FMemoryManager::Deinit` / `RHIExit`，只算 shutdown crash。

## 2. 网络资料与可检验假设

| 来源 | 级别 | 可用于本轮的结论 | 不允许推出 |
| --- | --- | --- | --- |
| [CARLA rendering options](https://carla.readthedocs.io/en/0.9.12/adv_rendering_options/) | 官方文档 | UE4.26 Linux 使用 Vulkan；`-RenderOffScreen` 是支持入口；no-rendering 下 GPU camera 输出为空；图形故障可尝试重建 `GameUserSettings.ini` | 不能据此认定 offscreen 或配置文件必然是根因 |
| [CARLA issue #4560](https://github.com/carla-simulator/carla/issues/4560) | 维护者讨论 | UE4.26 不应使用 `-opengl`；headless 环境应避免强设 `SDL_VIDEODRIVER` | 不得把 `-opengl` 当修复；本机 `SDL_VIDEODRIVER` 已未设置 |
| [CARLA discussion #7616](https://github.com/carla-simulator/carla/discussions/7616) | 社区个案 | NVIDIA 驱动变化可能影响 CARLA，是后续解释线索 | 不能据单个个案自主降级/重装驱动 |
| [CARLA issue #8043](https://github.com/carla-simulator/carla/issues/8043)、[#5970](https://github.com/carla-simulator/carla/issues/5970) | 未决问题 | 混合显卡、Vulkan、offscreen+Python 触发崩溃有相似报告 | 不作为已验证解决方案 |

本轮检验四个互斥度较高的假设：H1=world/load 本身；H2=GPU scene-capture/render path；H3=某传感器组合或 actor skeletal mesh；H4=RHI 并发或持久图形配置。`-norhithread` 与配置重建只在同一 full reproducer 上比较，属于诊断性 mitigation，不先声称根因。

## 3. 统一运行协议

执行根目录：

```bash
CRASH_ROOT=/home/yk/ws/carla_benchmark_data/carla_crash_diag_20260817_154949
CARLA_ROOT=/home/yk/ws/third_party/CARLA_0.9.16
```

执行模型先在 `$CRASH_ROOT` 下创建 `logs/`, `run_records/`, `crashes/`, `gpu/`, `settings_backup/`, `harness/`，并保存：git dirty 基线、`nvidia-smi -q`、`vulkaninfo --summary`、ICD/env、CARLA binary SHA-256、当前 `GameUserSettings.ini` SHA-256、测试前 crash-dir 清单。不得修改项目源码；若确需 harness，写在 `$CRASH_ROOT/harness/`。

每个 run：

- 串行执行，独占 GPU；端口为 `21000 + 10 * run_index`，并在启动前检查该端口及 `port+1/+2` 均空闲。发现占用只记录 owner，不终止非 owned 进程。
- 启动环境固定：`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`、`__NV_PRIME_RENDER_OFFLOAD=1`、`__GLX_VENDOR_LIBRARY_NAME=nvidia`、`__VK_LAYER_NV_optimus=NVIDIA_only`、`SDL_VIDEODRIVER` unset；server 固定 `-RenderOffScreen -nosound -quality-level=Low -carla-port=<unique>`，仅 D10 在 server CLI 加 `-norhithread`。D02 的 no-rendering 必须通过 Python API 设置 `world.get_settings().no_rendering_mode=True`，不得臆造 server CLI 参数。
- 保存 wrapper PID、child PID、`/proc/<pid>/cmdline`、进程开始时间和端口绑定。后续只操作这组 positively-owned PID；禁止 `pkill`/`killall`。
- readiness 最多 60 s；Python client timeout 30 s；active phase 最多 120 s；目标 120 synchronous ticks（fixed delta 0.05 s）或 30 个三传感器 frame bundles。每 5 s 采一份 GPU memory/utilization；记录 active phase 起止 wall time。
- active phase 结束先写 `active_phase_completed=true/false`、server 是否存活、最后 tick/frame、client exception，再发 stop。先销毁本 run 创建的 sensors/actors并恢复 world settings；然后向 positively-owned wrapper/child 发 TERM，10 s 后仍存活才可对同一已核验 PID 发 KILL，并记录原因。
- 对比 run 前后 `/home/yk/.config/Epic/CarlaUE4/Saved/Crashes/crashinfo-*`，复制新增 crash 目录索引/`CrashContext.runtime-xml`/调用栈到本 run 目录；按 crash timestamp 与 `stop_signal_epoch` 分类 ACTIVE 或 SHUTDOWN。
- 每个 run 前后保存 `journalctl -k` 的 NVIDIA/Xid/OOM 过滤结果。若无权限，状态写 `unavailable`，不得伪写为“无 Xid”。
- `run_records/<RUN_ID>.json` 必含：完整 server/client 命令、控制变量、PIDs/ports、timestamps、exit codes、tick/frame counts、server_alive_before_stop、new crash dirs、active/shutdown 分类、首个 symbol、GPU peak、Xid/OOM、status 和解释。

统一状态：

- `PASS`：完成目标 ticks/frames，停止信号前 server 存活；即使 stop 后有 shutdown crash，active 结果仍 PASS，另记 `shutdown_crash=true`。
- `FAIL`：结构完整且复现 active crash；这是有价值的阳性诊断，不停止后续独立 run。
- `BLOCKED`：环境/端口/readiness 等无法进入被测阶段。
- `INVALID`：harness/控制变量/记录不成立。
- `NOT_RUN`：只允许真实依赖被阻断或总安全门触发。

全局安全门只有三项：出现 Xid/GPU reset、连续两个 run 遗留无法确认归属的 CARLA process，或设置备份无法保证恢复。触发后停止后续 live runs并保存证据；单个 `Signal 11`、低帧率或一次质量失败不是全局停止理由。

## 4. 12 个 OFAT runs

除表中唯一变化外，所有 camera 使用与失败 capture 一致的 640×480、FOV 90、sensor tick 0、ego stationary、ClearNoon、seed 42；full actor set 固定为 car 4、truck 2、bus 2、bicycle 2、motorcycle 2（共 12）、walkers=0。固定首选蓝图分别为 `vehicle.audi.a2`、`vehicle.carlamotors.carlacola`、`vehicle.mitsubishi.fusorosa`、`vehicle.bh.crossbike`、`vehicle.harley-davidson.low_rider`，实际 actor ID/blueprint 必须落盘。D03–D09 使用默认 RHI 与原始 settings。若固定 blueprint 不存在或无法生成，run 记录 `INVALID`，不得悄悄换类别。

| Run ID | 唯一被测因素 | 场景/传感器/actor | 成功条件 | 失败如何解释 |
| --- | --- | --- | --- | --- |
| D00-BASE-T10-TICK | server+tick baseline | 当前 Town10HD_Opt；无新 actor、无 sensor；120 ticks | active phase 完成且 server alive | 崩溃则 H1 成立，后续 camera 结果不可单独归因；独立 no-rendering仍可做 |
| D01-LOAD-T05-TICK | `load_world(Town05)` | 无新 actor、无 sensor；load 后 120 ticks | load/ticks 完成 | 只此失败说明 Town05 load/map path 有问题，不是 camera 充分证据 |
| D02-NORENDER-FULL | rendering off isolation | Town05；full actors；创建同样三 camera 但不要求图像；120 ticks | server alive；camera 空是预期 | D09 active crash而本 run稳定，支持 GPU render path；本 run也崩溃则 actor/world path仍可能 |
| D03-RGB-EGO | RGB camera | Town05；ego only；RGB；30 frames | 30 RGB frames | 复现表示最小 scene capture 已足够 |
| D04-SEM-EGO | semantic camera | 同 D03，仅 semantic | 30 semantic frames | 只此失败指向 semantic pass/material pipeline |
| D05-INSTANCE-EGO | instance camera | 同 D03，仅 instance | 30 instance frames | 只此失败指向 instance pass/material pipeline |
| D06-3CAM-EGO | camera concurrency | ego only；RGB+semantic+instance | 30 对齐 bundles | singles PASS、此 FAIL 支持多 capture/RHI 组合问题 |
| D07-3CAM-CARS | wheeled skeletal family A | D06 + 4 个 car，固定蓝图列表 | 30 bundles | 相对 D06 失败说明 actor mesh/visibility 参与；不等于所有汽车都有问题 |
| D08-3CAM-2WHEEL | wheeled skeletal family B | D06 + bicycle 2 + motorcycle 2，固定蓝图 | 30 bundles | D07 PASS、此 FAIL 强支持 two-wheel skeletal mesh/blueprint family |
| D09-FULL-DEFAULT | known reproducer | Town05；三 camera；full actor set；30 bundles | 30 bundles | FAIL 是正式 active reproducer；首栈按 skeletal/RHI/other分类 |
| D10-FULL-NORHI | RHI thread diagnostic | 与 D09 完全相同，仅 server 加 `-norhithread` | 30 bundles | D09 FAIL、D10 PASS 支持 RHI 并发 mitigation；不是驱动根因证明 |
| D11-FULL-FRESH-SETTINGS | persistent graphics config | 与 D09 完全相同；仅临时移出 `GameUserSettings.ini` 让 UE 重建 | 30 bundles 且原文件最终恢复 | D09 FAIL、D11 PASS 支持 stale settings；两者同类 FAIL 则否定该简单解释 |

执行顺序固定 D00→D11。D01 `BLOCKED` 只阻断 Town05 的直接依赖；D00 与 D01 均通过后按序继续。质量/active crash `FAIL` 不传播停止，因为后续 run 是为归因而设计。若 D01 load Town05 始终无法进入 active phase，D02–D11 改用**已记录的原始 Town10HD_Opt**，并在所有 record 写 `map_fallback=true`；不得混合两张地图做无标注比较。

## 5. D11 可恢复配置协议

目标文件：`/home/yk/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/GameUserSettings.ini`。

1. 启动任何 CARLA 前，把原文件复制到 `$CRASH_ROOT/settings_backup/`，保存 mode/mtime/SHA-256。
2. D00–D10 只读原文件。
3. D11 前确认无 CARLA owned/non-owned process；将原文件**移动**为同目录唯一 `.aris_backup_20260817_154949`，不删除。
4. D11 无论 PASS/FAIL/timeout，都在 cleanup trap 中先归档新生成文件及其 SHA-256，再把原文件恢复到原路径并校验 SHA-256。
5. 原文件不能备份或 cleanup trap 未安装时，D11=`NOT_RUN`，其余结果仍有效。

## 6. 结论矩阵与交付

| 观测模式 | 最窄可支持结论 | 下一步（本轮不自动执行） |
| --- | --- | --- |
| D00/D01 FAIL | world/load 或基本 Vulkan tick 已不稳定 | 用完整 stdout/crash stack 上游报 issue；不做模型实验 |
| D02 PASS，D03–D09 任一 FAIL | GPU scene-capture/render path 被隔离 | 根据 singles/combination/actors继续缩小 |
| D03–D05 PASS，D06 FAIL | 多 GPU camera/RHI combination | 降并发或错峰传感器作为候选 workaround |
| D06 PASS，D07/D08 FAIL | actor/mesh family参与 | 固定到具体 blueprint 的后续二分 |
| D09 FAIL，D10 PASS | `-norhithread` 是本机可复现 mitigation | 再做重复性与性能验证后才进入正式命令 |
| D09 FAIL，D11 PASS | settings 重建是本机可复现 mitigation | 审计新旧 ini 差异；不直接覆盖用户配置 |
| active 均 PASS，仅 stop 崩溃 | capture 不再复现；shutdown 生命周期独立问题 | 改进停止协议，不能宣称渲染根因已修复 |
| active crash + Xid | 与当前“无 Xid”前提不同，升级为 GPU fault | 停止自主 live runs，交由用户决定驱动/系统动作 |

最终写入 `$CRASH_ROOT/summary/`：`crash_matrix.{json,csv,md}`、`stack_index.json`、`gpu_xid_summary.json`、`settings_diff.md`（若 D11 运行）、`CARLA_CRASH_DIAG_REPORT.md` 和 `AUTOTEST_STATUS.json`。报告必须列出正/负结果、未运行项、active/shutdown crash 数量、每类首栈、最小 reproducer、被支持/被否定/尚未区分的假设。网络线索只用于 debug discovery，不进入论文参考文献或研究 claim。
