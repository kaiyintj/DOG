# Lite3 实机下一阶段交接

核对日期：2026-09-29

本文是开启新 Codex 窗口时的 Lite3 专用交接入口。新窗口应先读本文、项目根目录的
[README.md](../README.md) 与 [AGENTS.md](../AGENTS.md)，再按需查阅
[PROJECT_STATUS.md](PROJECT_STATUS.md) 和 [RUNBOOK.md](RUNBOOK.md)。旧聊天中的临时命令、
旧 Bag 路径和旧脚本哈希不得作为当前依据。

## 1. 当前结论

实机仍为 `MOTION_READY=NO`。已有证据按日期区分：

| 日期 | 已验证内容 | 边界 |
| --- | --- | --- |
| 8 月 18/23 日 | 四路静止采集、Header 审计；B 盘 FAST-LIO + CLIP + GA-BSVM 离线烟测 | `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`，不证明运动几何 |
| 9 月 1 日 | 六位姿外参候选、留出验证、多距离/方位叠加与静止融合 | 候选写入 Lite3 YAML；实时 TF 和运动投影未验收 |
| 9 月 2 日 | 隔离里程计桥静止 120 秒输出检查 | 只发布测试帧；运动方向/尺度和正式 TF 权威未验收 |
| 9 月 24/26 日 | 电脑端真实传感器接收、投影及隔离组件输出 | 详见下文；未通过导航闭环 |

机载为 Ubuntu 20.04/Foxy/aarch64，电脑为 Humble。外参约定为
`x_camera = R * x_rslidar + t`。原始 Bag 的 `/tf_static` 缺少雷达到相机光学帧链路，
不能把 YAML 外参视为已有 TF 发布关系。

9 月 2 日厂商 `/leg_odom2` 审计发现空 frame 和稳态时钟 Header；隔离桥只输出
`lite3_test_odom -> lite3_test_base_link` 与 `/diagnostics/lite3/leg_odom_fixed`。
正式 `odom -> base_link` 和 SDK 速度执行桥仍须验收。

### 9 月 24 日新增实机静止测试

电脑经有线 ROS 2 链路收到约 10 Hz LiDAR、200 Hz IMU、15 Hz 图像/内参，并保留四组新 Bag。
第四轮离线 SegFormer/投影报告仍在，三个采样帧 chair 像素为零；上一位置中间帧对照记录为
2498 个 chair 像素、42 个雷达点。此结果不能证明稳定三维目标或外参精度。
前三轮分析目录本次核对时缺失，原始录包仍在本机；第四轮可查
[跨系统结果摘要](results/lite3_20260924_20260926/chair_projection_20260924.md)。
这些新测试未运行完整 FAST-LIO/GA-BSVM/Nav2 链，实时雷达—相机 TF 仍待验收。
若调整相机高度/俯仰，须重新标定；保持所有实机运动门禁。

### 9 月 26 日电脑端隔离排查

真实输入的隔离里程计桥、静止 FAST-LIO 和 Indoor-7 实时语义融合已分别观察到输出，
未执行导航。融合的 60 秒功能检查通过，但用户随后报告曾因低电量趴下，具体时刻未确认；
该窗口不能作为全程静止或运动几何验收。历史 IMU 对齐告警在追加的短时检查中未复现，
原因尚未确认。

已在真实 Livox 订阅边界确认并修正停止时的转换异常：ROS context 关闭后处理已知
`take_message` RuntimeError；通信有效时与其他错误仍抛出。8 项 shutdown 定向测试通过，
实测同一异常再次触发后正常退出 0。这项 Python 修正不解除实机门禁。
完整排查见 [当前融合记录](results/lite3_20260924_20260926/static_fusion_20260926.md)。
旧实验的保留用途与清理候选记录在 Linux 本机 `lite3_offline_runs/README.md`；
普通短时排查复用已有目录，不默认录完整 Bag。

同日 23:44 最后一次连接检查为有线 `DOWN/NO-CARRIER`，未进行数据验收。
这是当时的连接状态；下次会话先确认现场供电、有线地址和传感器输入。

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
- `/home/yk/ws/lite3_offline_runs`：B 盘的历史精简归档和当前完整烟测证据；
- 新运行按 RUNBOOK 显式传 `--work-root /home/yk/ws/lite3_offline_runs`；脚本省略该参数时
  仍默认写入 `~/lite3_offline_runs`，避免误把主目录与工作区中的同名目录混淆；
- 当前推荐离线证据是
  `/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260823T030733Z_wR2TtN`。它约 600 MiB，
  保留完整 `merged/`、`output_bag/`、报告和日志；manifest 绑定 `b86008d`、
  `git_dirty=false` 和 `ALGORITHM_STATIC_PASS_NON_GEOMETRIC`；
- 历史迁移证据
  `/home/yk/ws/lite3_offline_runs/lite3_clip_smoke_20260821T020049Z_xLnEzN` 仍约 1.2 MiB，
  绑定 `d27c103`。它故意不保留 `merged/` 与 `output_bag/`，只能验证迁移来源和历史
  结论，不能直接重放。

manifest 与 `source_path.txt` 中的 `/home/yk/lite3_*` 是 2026-08-21 运行时的 A 盘来源
记录，不是 B 盘当前路径，不得为了迁移而改写。B 盘完整性使用 RUNBOOK 第 8.9 节的
源 Bag、现存制品和实现文件三组哈希验证。

推荐 Bag 的基础验收为 `OVERALL=PASS`，并带四路接收调度警告；严格 header 审计为
`SENSOR_HEADER_ALIGNMENT=PASS`。原始 `/tf_static` 仍缺少 LiDAR 到相机光学帧的链路；
外参来自后续独立六位姿标定，不能把算法参数误当作已有的 TF 发布关系。

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
- B 盘精简归档完整性验证：`scripts/verify_lite3_migrated_archive.py`；
- B 盘 Python/ROS 运行时预检：`scripts/check_b_disk_runtime.py`；
- 静止 FAST-LIO 配置：`config/fast_lio_lite3_offline.yaml`；
- 受控运动数据候选配置：`config/fast_lio_lite3_real.yaml`；
- 实机语义配置：`config/semantic_mapping_lite3_real.yaml`；
- 隔离里程计桥：`semantic_mapping/runtime/lite3_odom_bridge_node.py`；
- 完整传感器和离线命令：[RUNBOOK 第 8 节](RUNBOOK.md#8-lite3-实机传感器采集与上机前清单)。

历史算法与实验基线为 `d27c103`，A 盘交接来源为 `f9b75de`，B 盘完整复验绑定
`b86008d`。这些版本证明当时的静止链路，不代表当前工作区代码已重新通过同样的实机验收。
当前融合、查询及仿真配置的后续修改与验证边界见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
后端无关的 `semantic_query` 是 `/text_query` 正式便捷发布器；`clip_query` 保留为兼容
别名。选择 CLIP 时文本特征仍由正在运行的 `clip_node` 编码。
实时 Git 状态统一按 [当前工作区状态](PROJECT_STATUS.md#查看实际工作区状态) 查询。

## 4. 新窗口首先执行的任务

不要重复运行已经通过的静止烟测，除非代码、依赖、传感器安装、驱动或录制模式发生
变化。新窗口应先只读核对推荐 Bag 与推荐 smoke 的 manifest/验收报告，然后按以下顺序
推进：

1. 保存机器狗真实 TF 树，核对 `rslidar`、`camera_color_optical_frame`、`base_link`、
   `odom` 的来源和父子关系；
2. 核对安装是否仍与 9 月 1 日静止标定候选一致，复用已有多距离/多方位验收；
   安装改变才重新标定，运动条件下的投影仍待验收；
3. 用隔离里程计桥验证厂商 `/leg_odom2` 的时间、方向和尺度；通过前不发布正式
   `odom -> base_link`；
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

## 6. 运动前不可跳过的阻塞项

以下项目未完成前，不发布 `/cmd_vel`，不启动 Nav2 实机闭环：

1. 确认已有静止标定候选在当前安装下仍有效，并补齐受控运动投影验收；
2. 统一 `/Odometry`、Nav2 odometry 和 `odom -> base_link`，保证只有一个权威发布者；
3. 实现 `/cmd_vel_lite3_safe -> 云深处官方 SDK` 的唯一安全桥；
4. 安全桥必须具有限速、模式/姿态检查、100--300 ms 命令超时自动零速、网络断开本地
   停车和独立急停；
5. 用 `fast_lio_lite3_real.yaml` 验证受控运动 Bag，而不是把静止烟测配置用于运动；
6. 在受控运动数据上验收已实现的源图时间戳传递和 GA-BSVM 历史 TF 查询；
7. 最后才按“架空测试 -> 低速空场 -> 障碍环境”逐级解锁。

当前仓库把实机速度链停在 `/cmd_vel_lite3_safe`，且没有下游执行桥；同时
`projection_calibration_verified` 保持 `false`、Goal Bridge 默认关闭。这些是安全门禁，
不能为了让狗先动起来而关闭。

## 7. 资料入口

采集与离线命令见 [RUNBOOK 第 8 节](RUNBOOK.md#8-lite3-实机传感器采集与上机前清单)；
源码后续修改与最新测试见 [当前状态](PROJECT_STATUS.md)。
[电脑/板载计算调研](research/LITE3_HOST_COMPUTE_FEASIBILITY_20260922.md) 是 9 月 22 日方案资料，
不代表新硬件已验收。
