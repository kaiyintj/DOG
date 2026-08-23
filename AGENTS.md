# Repository instructions

## Default workflow

1. Read [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for the current capability and
   migration state. Its Git-status commands are the only realtime source for branch,
   HEAD, remote divergence and dirty files.
2. For Lite3 collection, calibration, mapping, SDK or navigation work, also read
   [docs/LITE3_REAL_HANDOFF.md](docs/LITE3_REAL_HANDOFF.md) and the relevant section of
   [docs/RUNBOOK.md](docs/RUNBOOK.md). Lite3 is the default active branch of work.
3. For CARLA/ARIS work, read [refine-logs/AGENTS.md](refine-logs/AGENTS.md),
   [refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md) and
   [refine-logs/EXPERIMENT_TRACKER.md](refine-logs/EXPERIMENT_TRACKER.md) only after
   the user explicitly resumes that experiment and its excluded installation/data are
   restored. On the migrated B-disk snapshot, `AUTO_DEPLOY=false`.

## Stable safety invariants

- Keep `projection_calibration_verified=false`, `goal_bridge_enabled=false` and
  `MOTION_READY=NO` until the real LiDAR-camera calibration, TF authority, SDK safety
  bridge and controlled-motion gates in the handoff are accepted.
- Keep the Lite3 command path fail-closed: `/cmd_vel_lite3_safe` has no chassis
  executor until an accepted vendor-SDK bridge provides limits, mode/state checks,
  watchdog zeroing, network-loss stopping and an independent emergency stop.
- Motion V2 remains CARLA diagnostic evidence and must not enter the real-robot runtime
  or formal real-robot weights without new evidence and user approval.
- Semantic LiDAR is offline CARLA ground truth only; it must not enter model or runtime
  prediction inputs.

## Working-tree protection

- Before changing any repository under `/home/yk/ws/src`, record its branch, HEAD and
  `git status --short --branch`. Several migrated dependency repositories intentionally
  contain preserved tracked and untracked user changes.
- Treat existing dirty and untracked files as user-owned. Preserve them and avoid broad
  cleanup, reset or overwrite operations.
- Scope staging to the reviewed paths. A local commit, a history rewrite and a remote
  push are separate actions and each requires user authorization.

## B-disk Lite3 evidence

- Algorithm/evidence baseline: `d27c1032f97d8e744c3ee2f2ef196c00ea6bac7e`.
- Immutable source Bag:
  `/home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7`.
- Compact smoke evidence:
  `/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN`.
- The compact archive intentionally omits only the four `merged/` and `output_bag/`
  payload files recorded by the migration validator. Historical manifest/source paths
  are provenance and must not be rewritten.
- Validate this evidence through the migration command in RUNBOOK section 8.9 before
  claiming that the archive is complete.
- Before claiming that B-disk ROS execution works, require
  `scripts/check_b_disk_runtime.py --backend <clip|segformer>` to report
  `OVERALL=B_DISK_RUNTIME_READY`; archive integrity and runtime readiness are separate.

## Completion criteria

- The requested change is complete only when intended paths, relevant validation,
  documentation links and `git diff --check` pass.
- Report skipped builds, launches or hardware checks explicitly. Historical A-disk
  results do not prove that the rebuilt B-disk environment can execute them.

## 范围约束（HERO：反过度防御）

来源：https://github.com/wanshuiyin/HERO-Anti-OverDefense

=== 范围约束(约束你提议什么修法,不约束你找什么)===
凡是这里真的有问题,都要报——包括听起来罕见但本项目确实会产生的情况。
然后把修法收在范围内:
1. 这不是一篇安全攻防论文。可以校验,禁止过度防御。除非本项目另有说明,默认操作者是
   自己机器上的合作者;如果它真有对手,它会写明,以那个范围为准。
2. 不要加哈希/校验和/指纹,除非它替代了一个实质上更贵的操作,并且结果会改变下一步做什么。
3. 禁止防御性脚手架:不为这里不会发生的情况加 feature flag、迁移框架、兼容层、包装层。
4. 禁止钻牛角尖:冷门编码、符号链接竞态、RTL 文本、毫秒级竞态一律不在范围内,
   除非该情况经由本项目**受支持的用法**可达——它的文档示例、它公开的接口、它真实的
   数据。可达即可,不需要你复现出来;但"理论上构造得出"不算。
5. 该判断的地方就判断,不要换成评分表、检查清单,或对已经定论的东西再跑一遍校验。
6. 以上都不覆盖用户、本项目自己的约定、或更高优先级规则明确要求的安全、迁移、校验与
   审阅。那些是被要求的,是活儿本身,不算范围外。
已经见过的形状,供你校准。是例子不是清单——一个真问题不会因为"长得像其中一条"就被驳回:
  H  为了比对两个表格的差异,给每一行都算哈希——直接比单元格就能回答
  H  写下一堆校验和文件,而没有任何代码会去读它们
  E  给一个没有用户、没有部署的应用做账号安全加固
  R  用一整夜对自己的补丁反复审计,而功能一行没写
  R  一个对任何提交都给不通过的审阅者
  O  一层守卫的理由是上一层守卫,而不是需求
另有两种长得像上面、但不是的。这些要报:
  ✓  用摘要比对来跳过重读一个你已经有的大文件
  ✓  本项目自己的文档示例就会产生的那种"听起来罕见"的输入
跑任何检查之前先回答:这次运行会检测出什么具体的失败?真出现了我下一步会做什么不同的事?
答不上来就别跑。
对的就说对。不要为了交差硬找问题。
