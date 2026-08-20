# Lite3 实机下一阶段交接

更新日期：2026-08-16

本文是开启新 Codex 窗口时的 Lite3 专用交接入口。新窗口应先读本文，再按需查阅
[PROJECT_STATUS.md](PROJECT_STATUS.md) 和 [RUNBOOK.md](RUNBOOK.md)。旧聊天中的临时命令、
旧 Bag 路径和旧脚本哈希不得作为当前依据。

## 1. 当前结论

当前已经完成并确认：

- Lite3 机载环境为 Ubuntu 20.04、ROS 2 Foxy、aarch64；
- Mid360 点云 `/timefix/lidar` 约 10 Hz，Mid360 IMU `/timefix/imu` 约 200 Hz；
- D435I 使用 `424x240@15Hz` 彩色和深度启动模式时，彩色图像和 CameraInfo 可稳定录制；
- 三个独立 recorder、SQLite 检查、频率验收、消息时间戳审计和 USB 分级检查工具已经实现；
- 2026-07-26 最佳静止 Bag 的四路消息频率、计数、共同时间窗和 header 连续性通过；
- 录包接收时间有 0.52--0.69 秒调度间隔，只记为 `PASS_WITH_RECEIPT_WARN`；消息自身
  header 连续，因此不是 0.5 秒传感器断流；
- Image 与 CameraInfo 在 1 ms 内匹配率为 100%，LiDAR 到最近图像的 p99 时间差约
  32.9 ms，所有 LiDAR 帧周围都有 IMU；
- CameraInfo 内参稳定，但 `/tf_static` 中仍没有 `rslidar -> camera_color_optical_frame`
  链路。

当前严格状态为：

```text
SENSOR_HEADER_ALIGNMENT=PASS
CALIBRATION_STRUCTURE=NOT_READY
MOTION_READY=NO
```

因此，现在可以继续做静止数据的电脑离线算法烟测，但不能据此让机器狗行走。

## 2. 当前必须保留的数据

电脑上的正式原始数据归档是：

```text
/home/yk/lite3_robot_captures/lite3_concurrent_20260726_202334_azggiT
```

该目录约 562 MiB，是当前离线实验的唯一推荐输入，不要修改或删除。主要统计为：

| 话题 | 消息数 | 频率 | header 最大间隔 |
| --- | ---: | ---: | ---: |
| `/timefix/imu` | 13558 | 200.000 Hz | 0.0122 s |
| `/timefix/lidar` | 677 | 10.000 Hz | 0.1038 s |
| `/camera/color/image_raw` | 1014 | 14.992 Hz | 0.0668 s |
| `/camera/color/camera_info` | 1014 | 14.990 Hz | 0.0668 s |

四话题共同时间窗约 67.437 秒。机器狗上的原始副本是：

```text
/home/ysc/lite3_bags/lite3_concurrent_20260726_202334_azggiT
```

目录用途不要混淆：

- `/home/yk/lite3_robot_captures`：机器狗原始 Bag 的电脑归档，相当于实验“底片”，必须保留；
- `/home/yk/lite3_offline_runs`：脚本生成的合并 Bag、日志和结果，可按单次运行清理；
- 当前旧离线目录
  `/home/yk/lite3_offline_runs/lite3_clip_smoke_20260726T104318Z_k1GU5P`
  使用的是旧 Bag，结果为 `FAIL`，不能代表上述最佳 Bag。

最佳原始 Bag 目录内原有的 `validation.txt` 是阈值修正前生成的历史报告，其中 IMU
接收间隔 0.521 秒被旧规则误判为失败。用当前
`scripts/verify_lite3_capture.py` 复验的结果是基础 `OVERALL=PASS` 并带四路接收调度
警告；严格 header 审计仍为 `SENSOR_HEADER_ALIGNMENT=PASS`。

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

截至本文写入时，本地 Git 分支为 `agent/carla-motion-v2-results`，HEAD 为 `963aa69`。
本文以及 README、PROJECT_STATUS、RUNBOOK 的交接链接是当前未提交文档修改；除此之外
本次核对未发现其他工作区改动。新窗口开始时仍需重新运行 `git status --short`，不要
假设状态不会变化，也不要清理这些交接文档。

## 4. 新窗口首先执行的任务

下一步不是发布运动命令，而是在电脑端用最佳 Bag 完成一次完整离线烟测。先做
`--prepare-only`：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
cd /home/yk/ws/src/semantic_mapping

bash scripts/run_lite3_offline_smoke.sh \
  --input /home/yk/lite3_robot_captures/lite3_concurrent_20260726_202334_azggiT \
  --prepare-only

source /home/yk/lite3_offline_runs/lite3_offline_current_run.env
printf 'RUN_DIR=%s\n' "$RUN_DIR"
cat "$RUN_DIR/OVERALL"
```

只有输出 `PREPARED` 才继续完整烟测。完整烟测仍直接使用正式原始归档路径；本地输入的
prepare-only 运行不会创建 `$RUN_DIR/raw`，不要再把 `/raw` 当作输入：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
cd /home/yk/ws/src/semantic_mapping

bash scripts/run_lite3_offline_smoke.sh \
  --input /home/yk/lite3_robot_captures/lite3_concurrent_20260726_202334_azggiT

source /home/yk/lite3_offline_runs/lite3_offline_current_run.env
printf 'RUN_DIR=%s\n' "$RUN_DIR"
cat "$RUN_DIR/OVERALL"
```

该回放默认是 0.10 倍速，约需 11 分钟。理想输出为：

```text
ALGORITHM_STATIC_PASS_NON_GEOMETRIC
```

它只证明 FAST-LIO、CLIP 和 GA-BSVM 能处理这份静止数据，不证明投影几何正确，也不
授权运动。

## 5. 再次连接机器狗时开几个终端

如果只是重新采集静止 Bag，开三个机器狗 SSH 终端：

1. 终端 1 按 [RUNBOOK 8.2](RUNBOOK.md#82-终端-1mid360-与-imu) 启动并保持
   Mid360/IMU；
2. 终端 2 按 [RUNBOOK 8.3](RUNBOOK.md#83-终端-2d435i) 启动并保持 D435I；
3. 终端 3 执行 `bash ~/lite3_tools/record_lite3_sensors.sh`。

这三个终端只负责传感器和录包，不会让机器狗运动。当前最佳 Bag 已满足静止采集要求，
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
/home/yk/ws/src/semantic_mapping/docs/LITE3_REAL_HANDOFF.md
/home/yk/ws/src/semantic_mapping/docs/PROJECT_STATUS.md
/home/yk/ws/src/semantic_mapping/docs/RUNBOOK.md 的第 8 节。

当前从 LITE3_REAL_HANDOFF.md 第 4 节继续。先在电脑端用
/home/yk/lite3_robot_captures/lite3_concurrent_20260726_202334_azggiT
运行 prepare-only 和完整离线烟测，分析实际输出。不要发布任何运动命令，不要启动
Nav2 实机闭环，也不要把 ALGORITHM_STATIC_PASS_NON_GEOMETRIC 当成 MOTION_READY。
如果进入机器狗接口调查，先做只读检查并给出每个终端的精确命令。
```
