# Lite3 / Livox 在 Jetson 上的开源减载实践调研

日期：2026-09-06

## 结论

开源项目与上游仓库没有给出“Jetson 算力有限，所以 MID-360 出现约 1.7 秒空档属于正常”
的依据。可复用的实践集中在四层：

1. 使用包含 Livox 高 CPU 与分帧优化的驱动版本；
2. 原始雷达保持 `CustomMsg + 10 Hz` 基线，单雷达不拆多 topic；
3. 关闭 FAST-LIO 不需要的派生点云、RViz 和 PCD 累积，并通过点过滤/体素参数逐项减载；
4. 录包只录验收必需话题，按“驱动 → SLAM → 相机/状态”的顺序逐层加负载。

这些措施应作为**静止 A/B 候选**，不能直接永久套用。当前平台是 Xavier NX、Ubuntu
20.04、ROS 2 Foxy、MID-360，必须先记录已安装 `livox_ros_driver2` 的版本/提交，再验证
每一个改动。继续保持 `MOTION_READY=NO`；本调研不授权运动、Nav2 或速度闭环。

## 证据等级

- **上游事实**：Livox 官方驱动、官方 release、HKU-MARS FAST-LIO 源码。
- **作者项目实践**：项目作者在真实四足/Jetson 平台公开的仓库和实际配置。
- **问题报告**：上游 issue 中由使用者报告、但维护者尚未复现或给出修复结论的现象；
  只能用于形成假设，不能当作已确认根因。
- **弱参考**：个人项目仓库。仅作为 A/B 候选，不作为本机性能保证。

## 1. Livox 官方驱动：先确认版本，再谈参数

### 官方默认与格式选择

Livox 官方 ROS 2 MID-360 launch 默认：

- `xfer_format = 1`：Livox `CustomMsg`；
- `multi_topic = 0`：所有 LiDAR 共用一个 topic，单雷达无需拆分；
- `publish_freq = 10.0`；
- `output_type = 0`：发布到 ROS。

来源：

- [Livox ROS Driver 2 官方 README](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/README.md?plain=1)
- [官方 ROS 2 MID-360 launch](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/launch_ROS2/msg_MID360_launch.py)

官方 README 还明确把该驱动定位为调试工具，不建议不加优化直接用于量产。这一声明支持
“应审计驱动版本和运行链”，但不支持放宽传感器连续性门槛。

对 Lite3 的含义：当前 FAST-LIO 已使用 Livox `CustomMsg` 路径时，应保留它作为最小基线；
不要为显示方便在雷达源端额外发布一份 PointCloud2。单 MID-360 保持 `multi_topic=0`。

### 官方 release 中与 CPU/时序直接相关的修复

[Livox 官方 releases](https://github.com/Livox-SDK/livox_ros_driver2/releases)记录：

| 版本 | 官方说明 | 本机意义 |
| --- | --- | --- |
| 1.1.2 | 修复 high CPU；发布频率范围改为 0.5--10 Hz | 早于此版本不宜继续调参 |
| 1.1.3 | 改善 ROS 2 Humble 性能 | 不可直接推断 Foxy 同样获益 |
| 1.2.0 | 重写分帧；PointCloud2 增加逐点时间戳 | 跨版本需检查 PointCloud2 兼容 |
| 1.2.1 | 修复 CustomMsg timestamp 问题 | 对时间线审计相关 |
| 1.2.3 | 优化分帧并降低 CPU | Xavier 候选至少应包含此修复 |
| 1.2.4 | 继续优化 framing performance | 可作为受控升级候选 |
| 1.2.6 | 增加 MID-360s、Ubuntu 24.04/Jazzy 支持 | 不是本机所需，不能默认追最新 |

本机厂商目录的 `package.xml` 声明 `1.0.0`，且目录没有可用于追溯提交的 Git 元数据。
两个厂商 `Livox-SDK2` 源码目录同样没有 Git 元数据。已分别建立并实测隔离候选
`1.1.2`（提交 `99a1c7a`）和 `1.2.3`
（提交 `852c147`）。两者的真实 C++ driver 子进程都持续占用约一个 CPU 核；`1.2.3`
从 10 Hz 改为 5 Hz 后仍约 97--99%，因此本机现象没有被这两版的官方 CPU 修复消除，
也不是由 ROS 点云发布频率直接决定。若继续比较 1.2.4，仍须使用隔离副本，且不能预设
升级必然解决问题。跨越 1.2.0 时必须验证 CustomMsg 时间字段、PointCloud2 字段和
FAST-LIO 订阅兼容性。

机器人 `/usr/local/include/livox_lidar_def.h` 声明 SDK 内部版本 `3.1.1`；官方 SDK2
`v1.2.4` 的同一头文件也声明 `3.1.1`，但由于本机库和源码均无提交元数据，只能记为
“v1.2.4 等价或定制构建”，不能声称精确对应官方 tag。其动态库时间为 2024-08-06。
因此更换 ROS wrapper 候选并没有更换底层 SDK2，后续若要升级 SDK2 必须另建隔离库并在
运行时确认实际加载路径。

`1.2.3` 的线程采样进一步显示，总占用由至少三个 worker 共同构成（约 50.6%、25.1%、
18.2%），主线程和 SDK 接收线程多数在 futex、epoll 或 socket wait。该形状更符合
收包、组帧和发布流水线共同消耗，而不是单一 ROS 主循环空转；准确函数归属仍需 profiler
或带符号栈确认。

### 上游未解决的降频报告

- [Livox issue #114](https://github.com/Livox-SDK/livox_ros_driver2/issues/114)：报告者在
  Jetson AGX、ROS 2 Foxy/Humble 下配置 10 Hz/200 Hz，却只观察到约 5 Hz/50 Hz；ROS 1
  正常。issue 仍开放，没有维护者确认的根因或修复。
- [Livox issue #206](https://github.com/Livox-SDK/livox_ros_driver2/issues/206)：Raspberry Pi 5、
  Humble/Docker 下，单独驱动约 7.5 Hz，加入完整 pipeline 后更低；报告者称已查 CPU、
  温度和 UDP 丢包但未找到原因。issue 仍开放。

这些报告证明“ROS 2 路径实际可能降频，且不一定由平均 CPU 解释”，不证明降频正常，也
不证明本机根因相同。尤其不能据此接受 1.7 秒连续空档。

## 2. 四足机器人作者项目的实际做法

### Spot + Jetson AGX Orin + MID-360

University of Agder 的
[quadruped_navigation_ros2](https://github.com/mil-as/quadruped_navigation_ros2)
是明确的四足实例：Boston Dynamics Spot、Jetson AGX Orin、MID-360、ZED2i、Ubuntu
22.04、ROS 2 Humble/Isaac ROS。作者仓库称系统在实验室和办公室完成导航测试。

其实际
[MID-360 launch](https://github.com/mil-as/quadruped_navigation_ros2/blob/main/src/livox_config_launch/msg_MID360_launch.py)
采用：

- `xfer_format=0`（PointCloud2）；
- `multi_topic=0`；
- `publish_freq=10.0`；
- 默认 launch 不启动 RViz。

可复用结论：即使是同时使用 MID-360 与 ZED2i 的 Jetson 四足项目，雷达仍以 10 Hz 为
目标，并把可视化从驱动 launch 中剥离。不可复用之处：AGX Orin 明显强于 Xavier NX，
项目使用 Humble/Isaac ROS/Zenoh，且仓库没有公开 CPU、rosbag 丢失或最大 gap 指标。
因此它不能替代本机验收，也不能证明 PointCloud2 比 CustomMsg 更适合本机。

### 其他四足实例的限制

[go2_navigation](https://github.com/MattiaGrigoli/go2_navigation)描述 Unitree Go2 EDU、
Jetson Nano、MID-360、ROS 2 Foxy 原型，但没有公开可核对的 CPU、丢帧、最大 gap 或录包
完整度数据。它最多证明这类组合有人尝试运行，不能提供本机验收阈值或可靠减载参数。

另一个 Go2 实机项目
[find_my_human_go2](https://github.com/arpa-byte/find_my_human_go2)明确把 D435i 驱动放在
Jetson/Foxy，把 MID-360 驱动放在笔记本/Humble，并要求两机使用相同 RMW。它是个人项目，
不能证明该划分是唯一根因修复，但提供了与本机资源约束直接相关的工程候选：传感器计算
分机部署，而不是破坏 MID-360 的 10 Hz 基线。

## 3. FAST-LIO 官方可控负载

HKU-MARS 官方
[MID-360 launch](https://github.com/hku-mars/FAST_LIO/blob/main/launch/mapping_mid360.launch)
给出的处理参数包括：

- `point_filter_num=3`；
- `max_iteration=3`；
- `filter_size_surf=0.5`；
- `filter_size_map=0.5`；
- `runtime_pos_log_enable=0`。

官方 [mid360.yaml](https://github.com/hku-mars/FAST_LIO/blob/main/config/mid360.yaml) 暴露：

- `scan_publish_en=false`：关闭全部点云输出；
- `dense_publish_en=false`：减少 global-frame 注册点云点数；
- `scan_bodyframe_pub_en=false`：关闭机体系扫描输出；
- `path_en=false`：关闭 path 输出；
- `pcd_save_en=false`：关闭 PCD 保存。

该 YAML 当前示例的 `pcd_save_en=true`、`interval=-1` 会把全部帧写进一个 PCD；官方注释
明确警告帧数过多可能导致内存崩溃。长期实机连续性测试必须关闭 PCD 保存，除非该测试的
唯一目的就是验收 PCD 输出。

对本机的低风险顺序：

1. 先关 RViz、PCD、path、机体系点云等非验收输出；
2. 若当前测试只验传感器/录包，可完全不启动 FAST-LIO；
3. 若测试需要 FAST-LIO odom，但不需要注册点云消费者，测试
   `scan_publish_en=false`；
4. 若语义链需要注册点云，保留 scan，但测试 `dense_publish_en=false`；
5. 只有前述措施不足且算法精度允许时，才单变量提高 `point_filter_num` 或体素尺寸。

不能把 `point_filter_num=3` 或体素 `0.5 m` 当作本项目最终值：本项目已有自己的
MID-360/FAST-LIO 配置和几何要求，任何变化都要同时复验轨迹连续性与精度。

## 4. 一个与 Xavier NX 高度相关的反例

[HKU-MARS FAST-LIO issue #374](https://github.com/hku-mars/FAST_LIO/issues/374) 报告的硬件正是
Jetson Xavier NX + MID-360。报告者称原始 MID-360 数据正常，但 `cloud_registered` 发布
阻塞并连带造成 odom 3--4 秒中断；当时 CPU 不高、温度约 47°C。issue 后来仅因 stale
关闭，没有维护者确认的根因或修复。

这个 issue 不能证明本机也是相同 bug，却能推翻两个不安全的推断：

- 平均 CPU 不高，不代表 callback/发布路径没有局部阻塞；
- 派生点云或 odom 的 gap，不等于 MID-360 原始源频率异常。

因此后续报告必须分别列出 `/timefix/lidar` 原始消息、FAST-LIO odom、
`/cloud_registered` 和语义消费者的 header/receipt gap，不能用一条 topic 代表整条链。

## 5. 录包与传感器负载的可复用做法

本轮找到的强一手四足仓库没有公开量化 rosbag 调优结果。能可靠迁移的是实验设计，不是
某个“万能缓存值”：

- 不用 `ros2 bag record -a`；只录本阶段判定所需 topic；
- 原始 LiDAR/IMU 与派生大点云分开计量；不需要派生点云时不录；
- 关闭 RViz/Foxglove 等显示消费者，避免额外反序列化、转换和 DDS 复制；
- 保存 recorder 完整 stdout/stderr、topic QoS、系统负载和磁盘延迟；
- 同时比较消息 `header.stamp` gap 与 bag receipt gap；
- 缓存、writer 数量和合并/分拆 recorder 必须一次只改一个变量。

个人仓库
[Livox-mid360-docker](https://github.com/patrick-darbin-orica/Livox-mid360-docker)
面向 Jetson Orin、JetPack 6、Ubuntu 22.04、ROS 2 Humble，建议 FAST-LIO 高 CPU 时关闭
dense publish、调整 blind，并只录必要话题。它不是上游项目，平台也不同，只能作为
候选方向；其中 MAXN、`jetson_clocks` 或具体参数不可直接复制到 Xavier NX 正式配置。

## 6. 用户提供的语雀入口

浏览器实查
[语雀入口](https://www.yuque.com/lixupeng-rquex/nwvaxd)后可确认：

- 页面标题为“绝影lite四足机器人”，发布者显示为“AI机器人李工”；
- 目录包含激光版产品手册、20.04 版本感知/运动/通讯手册、SDK、课程和示例程序；
- “20.04版本感知开发手册”页面挂载《绝影Lite3感知开发手册(beta) V2.2.2.pdf》；
- “产品手册--激光版”页面挂载《绝影Lite3激光版产品手册V1.0.7.pdf》；
- “示例程序”当前只显示一个 `move_lite3.py` 附件；
- 可见目录与上述页面没有 GitHub/Gitee 项目仓库链接，也没有 Livox 驱动 CPU、发布频率、
  rosbag gap 或丢消息的量化结论。

该语雀空间不是云深处官方域名；仅凭页面作者名无法确认它由厂商维护，也无法从入口追到
一个开源感知/导航仓库。因此可把其中手册作为待核对资料入口，但本次不能把它当作开源
实现或一手性能证据。若后续使用附件中的接口事实，应先核验 PDF 内的厂商标识、版本和
官方发布渠道，并只引用相应具体页面。

## 7. 建议用于 Lite3 的静止分层测试

以下各阶段都保持机器狗趴下或稳定站立、`MOTION_READY=NO`、Nav2 关闭、速度 topic 无
publisher。每次记录驱动版本、功耗模式、CPU/内存/温度、磁盘延迟、UDP/内核错误、实时
计数、Bag 计数、header gap 和 receipt gap。

### A0：确认软件基线

- 记录 `livox_ros_driver2` 版本、提交和启动参数；
- 确认是 `CustomMsg`、`multi_topic=0`、`publish_freq=10.0`；
- 确认未同时启动 PointCloud2 转换、RViz、PCD 保存或重复雷达消费者。

### A1：驱动最小负载

- 只启动 MID-360 driver；
- LiDAR 与 IMU 使用两个独立 recorder；本机已实测单个无缓存 recorder 同录两者时，
  IMU 只记录约 102.6 Hz，而拆分后恢复约 200.0 Hz；
- 目标仍是点云约 10 Hz、IMU 约 200 Hz；
- 若这里已出现 header gap，优先查驱动、UDP、网口和设备，不查 FAST-LIO。

### A2：加入桥与底盘状态

- 加入里程计桥和 state recorder；
- 对比实时 raw/fixed 与 Bag raw/fixed；
- 若 fixed/raw 只在录包内下降，查 recorder；若实时也下降，查桥 callback/QoS/executor。

### B1：加入 FAST-LIO 最小输出

- `pcd_save_en=false`、`path_en=false`、`scan_bodyframe_pub_en=false`；
- 无下游注册点云需求时测试 `scan_publish_en=false`；
- 有需求时保留 scan，先测试 `dense_publish_en=false`；
- 分开检查原始 LiDAR、odom、registered cloud，防止 issue #374 形状的问题被平均 CPU
  掩盖。

### B2：逐层加入相机与语义链

- 先加 D435I 采集与必要话题录制，再加 SegFormer，再加 GA-BSVM；
- 每一层都与上一层比较，而不是直接重跑原满载动作；
- 若仅加入四个 SQLite writer 后失败，再比较非零缓存或合并 recorder；不要同时修改
  Livox publish rate、桥 QoS 和 FAST-LIO 点过滤。

### 何时才测试 5 Hz

只有在 10 Hz 驱动最小负载通过、逐层加负载明确显示处理能力不足，且业务验证证明 5 Hz
仍满足动态定位/融合时，才把 `publish_freq=5.0` 作为候选 A/B。5 Hz 的期望周期是
`0.2 s`；即使选择 5 Hz，约 `1.7 s` 仍相当于 8--9 个周期，不能自动变成正常。

## 8. 当前可以做与不能做

可以立即做：

- 离线检查现有 Bag 的 header/receipt 时间线；
- 在机器人端只读确认驱动版本和实际 launch 参数；
- 设计并运行无动作、逐层加负载的静止 A/B；
- 在隔离副本中准备驱动版本或 FAST-LIO 输出开关候选。

尚不能确认：

- 当前机器人的 driver 是否已经包含 1.2.3/1.2.4 CPU 修复；
- 1.7 秒空档究竟来自源端、DDS、writer、USB、存储还是派生发布阻塞；
- 降到 5 Hz 是否仍满足动态 FAST-LIO 与语义融合；
- 语雀资料是否由云深处官方维护，或是否存在未公开的配套源码仓库。

不能做：

- 仅因 Xavier NX 算力有限而放宽秒级 gap；
- 同时改驱动频率、FAST-LIO 参数、QoS 和 rosbag 布局后用一次成功宣称根因已定位；
- 用 Orin/Humble 项目的运行结果替代 Xavier NX/Foxy 的实测；
- 在静止满载门未通过前重启受控运动或自主导航验收。
