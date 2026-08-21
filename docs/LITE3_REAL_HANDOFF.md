# Lite3 实机下一阶段交接

更新日期：2026-08-22

本文是开启新 Codex 窗口时的 Lite3 专用交接入口。新窗口应先读本文、项目根目录的
[README.md](../README.md) 与 [AGENTS.md](../AGENTS.md)，再按需查阅
[PROJECT_STATUS.md](PROJECT_STATUS.md) 和 [RUNBOOK.md](RUNBOOK.md)。旧聊天中的临时命令、
旧 Bag 路径和旧脚本哈希不得作为当前依据。

## 1. 当前结论

当前已经完成并确认：

- Lite3 机载环境为 Ubuntu 20.04、ROS 2 Foxy、aarch64；
- Mid360 点云 `/timefix/lidar` 约 10 Hz，Mid360 IMU `/timefix/imu` 约 200 Hz；
- D435I 使用 `424x240@15Hz` 彩色和深度启动模式时，彩色图像和 CameraInfo 可稳定录制；
- 三个独立 recorder、SQLite 检查、频率验收、消息时间戳审计和 USB 分级检查工具已经实现；
- 2026-08-18 推荐静止 Bag 的四路消息频率、计数、共同时间窗和 header 连续性通过；
- 录包接收时间有 0.43--0.68 秒调度间隔，只记为 `PASS_WITH_RECEIPT_WARN`；消息自身
  header 连续，因此不解释为传感器断流；
- Image 与 CameraInfo 在 1 ms 内匹配率为 100%，共同窗口内 LiDAR 到最近图像的 p99
  时间差约 32.6 ms；
- CameraInfo 内参稳定，但 `/tf_static` 中仍没有 `rslidar -> camera_color_optical_frame`
  链路；
- 2026-08-21 已在开发电脑上用该 Bag 完成 FAST-LIO + CPU CLIP + GA-BSVM 离线烟测，
  并得到 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`；
- 烟测运行图中 `/cmd_vel` 发布者为 0，没有启动 Nav2、主动感知或运动桥。

当前严格状态为：

```text
SENSOR_HEADER_ALIGNMENT=PASS
ALGORITHM_STATIC_PASS_NON_GEOMETRIC
CALIBRATION_STRUCTURE=NOT_READY
MOTION_READY=NO
```

因此，静止采集和电脑离线软件链已经形成可复现基线；下一步转向外参、TF、SDK 安全桥
和受控运动 Bag，仍不能据此让机器狗行走。

## 2. 当前必须保留的数据

电脑上的推荐原始数据归档是：

```text
/home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7
```

该目录约 563 MiB，是当前静止离线实验的推荐输入，不要修改或删除。主要统计为：

| 话题 | 消息数 | 频率 | header 最大间隔 |
| --- | ---: | ---: | ---: |
| `/timefix/imu` | 13545 | 200.006 Hz | 0.0161 s |
| `/timefix/lidar` | 679 | 10.008 Hz | 0.1031 s |
| `/camera/color/image_raw` | 1014 | 14.990 Hz | 0.0681 s |
| `/camera/color/camera_info` | 1014 | 14.989 Hz | 0.0681 s |

四话题共同接收时间窗约 67.544 秒。机器狗上的原始副本是：

```text
/home/ysc/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7
```

目录用途不要混淆：

- `/home/yk/ws/lite3_bags`：B 盘机器狗原始 Bag 归档，相当于实验“底片”，必须保留；
- `/home/yk/ws/lite3_offline_runs`：B 盘迁移后的精简烟测证据归档；
- 未来重新执行脚本时，新的完整运行仍默认写到 `/home/yk/lite3_offline_runs`，不得与
  B 盘精简归档混淆；
- 当前推荐离线证据是
  `/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN`。它约 1.2 MiB，
  保留 manifest、配置、日志、验收报告和哈希账本，故意不保留 `merged/` 与
  `output_bag/`，不能直接重放；manifest 绑定 `d27c103`、`git_dirty=false` 和
  `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`。

manifest 与 `source_path.txt` 中的 `/home/yk/lite3_*` 是 2026-08-21 运行时的 A 盘来源
记录，不是 B 盘当前路径，不得为了迁移而改写。B 盘完整性使用 RUNBOOK 第 8.9 节的
源 Bag、现存制品和实现文件三组哈希验证。

推荐 Bag 的基础验收为 `OVERALL=PASS`，并带四路接收调度警告；严格 header 审计为
`SENSOR_HEADER_ALIGNMENT=PASS`。审计同时确认 `/tf_static` 缺少 LiDAR 到相机光学帧的
链路，因此不能把静止软件链通过解释成投影标定通过。

## 3. 当前代码入口

项目根目录：

```text
/home/yk/ws/src/semantic_mapping
```

重要文件：

- 机器狗录制：`scripts/record_lite3_sensors.sh`；
- 基础 Bag 验收：`scripts/verify_lite3_capture.py`；
- 严格时间戳和标定结构审计：`scripts/audit_lite3_timestamps.py`；
- 电脑离线烟测：`scripts/run_lite3_offline_smoke.sh`；
- 静止 FAST-LIO 配置：`config/fast_lio_lite3_offline.yaml`；
- 受控运动数据候选配置：`config/fast_lio_lite3_real.yaml`；
- 实机语义配置：`config/semantic_mapping_lite3_real.yaml`；
- 完整传感器和离线命令：[RUNBOOK 第 8 节](RUNBOOK.md#8-lite3-实机传感器采集与上机前清单)。

B 盘仓库位于 `main`，HEAD 为 `d27c103`。本轮未提交修改来源于 A 盘文档 commit
`f9b75de`，并已适配 B 盘精简归档；它们只同步证据、交接入口和搜索规则，不是算法
改动。用户最终审查前不得提交、推送或清理这些修改。

## 4. 新窗口首先执行的任务

不要重复运行已经通过的静止烟测，除非代码、依赖、传感器安装、驱动或录制模式发生
变化。新窗口应先只读核对推荐 Bag 与推荐 smoke 的 manifest/验收报告，然后按以下顺序
推进：

1. 保存机器狗真实 TF 树，核对 `rslidar`、`camera_color_optical_frame`、`base_link`、
   `odom` 的来源和父子关系；
2. 完成 `rslidar -> camera_color_optical_frame` 外参标定，并用多距离、多方位目标做
   重投影验收；
3. 明确唯一 `odom -> base_link` 权威发布者；
4. 只读调查云深处官方 SDK 的运动模式、状态反馈、急停和速度命令接口，设计
   `/cmd_vel_lite3_safe` 的唯一安全桥；
5. 标定、TF 和安全桥设计通过审查后，再设计受控运动 Bag，验证
   `fast_lio_lite3_real.yaml`，仍不直接启动 Nav2 闭环。

任何新任务都必须保留 `projection_calibration_verified=false`、
`goal_bridge_enabled=false` 和无 SDK 执行端的当前失败关闭状态，直到对应硬件验收完成。

## 5. 再次连接机器狗时开几个终端

如果只是重新采集静止 Bag，开三个机器狗 SSH 终端：

1. 终端 1 按 [RUNBOOK 8.2](RUNBOOK.md#82-终端-1mid360-与-imu) 启动并保持
   Mid360/IMU；
2. 终端 2 按 [RUNBOOK 8.3](RUNBOOK.md#83-终端-2d435i) 启动并保持 D435I；
3. 终端 3 执行 `bash ~/lite3_tools/record_lite3_sensors.sh`。

这三个终端只负责传感器和录包，不会让机器狗运动。当前推荐 Bag 已满足静止采集要求，
除非设备安装、驱动配置或传感器模式发生变化，否则没有必要重复采集同一种静止数据。

下一次真正有价值的实机会话，应优先做以下只读或静止工作：

1. 保存完整 TF 树并确定唯一 `odom -> base_link` 权威源；
2. 标定并验收 `rslidar -> camera_color_optical_frame` 外参；
3. 查清厂商运动 SDK、模式切换、急停、状态反馈和速度命令接口；
4. 明确受控运动 Bag 还要记录的底盘里程计、关节/足端状态、机器人模式、实际速度命令、
   急停/看门狗状态和 `/tf`；话题名必须从机器狗实际接口确认，不能凭空假定。

## 6. 运动前不可跳过的阻塞项

以下项目未完成前，不发布 `/cmd_vel`，不启动 Nav2 实机闭环：

1. 完成真实 LiDAR--相机外参标定和多距离、多方位投影验收；
2. 统一 `/Odometry`、Nav2 odometry 和 `odom -> base_link`，保证只有一个权威发布者；
3. 实现 `/cmd_vel_lite3_safe -> 云深处官方 SDK` 的唯一安全桥；
4. 安全桥必须具有限速、模式/姿态检查、100--300 ms 命令超时自动零速、网络断开本地
   停车和独立急停；
5. 用 `fast_lio_lite3_real.yaml` 验证受控运动 Bag，而不是把静止烟测配置用于运动；
6. 修复或验收所有语义消息的源图时间戳和 GA-BSVM 历史 TF 查询；
7. 最后才按“架空测试 -> 低速空场 -> 障碍环境”逐级解锁。

当前仓库把实机速度链停在 `/cmd_vel_lite3_safe`，且没有下游执行桥；同时
`projection_calibration_verified` 保持 `false`、Goal Bridge 默认关闭。这些是安全门禁，
不能为了让狗先动起来而关闭。

## 7. 可直接粘贴给新 Codex 窗口的开场消息

```text
请先完整阅读：
/home/yk/ws/src/semantic_mapping/README.md
/home/yk/ws/src/semantic_mapping/AGENTS.md
/home/yk/ws/src/semantic_mapping/docs/LITE3_REAL_HANDOFF.md
/home/yk/ws/src/semantic_mapping/docs/PROJECT_STATUS.md
/home/yk/ws/src/semantic_mapping/docs/RUNBOOK.md 的第 8 节。

当前从 LITE3_REAL_HANDOFF.md 第 4 节继续。先在电脑端用
/home/yk/ws/lite3_bags/lite3_concurrent_20260818_203250_HsW1R7
和
/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN
作为当前静止基线。烟测已经得到 ALGORITHM_STATIC_PASS_NON_GEOMETRIC，不要重复运行，
也不要把它当成 MOTION_READY。下一步只读检查真实 TF 树、LiDAR--相机外参结构和云深处
SDK 接口；不要发布运动命令，不要启动 Nav2 实机闭环，不要修改
projection_calibration_verified=false 或 goal_bridge_enabled=false。
```
