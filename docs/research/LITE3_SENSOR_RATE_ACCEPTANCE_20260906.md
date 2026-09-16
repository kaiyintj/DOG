# Lite3 满载采集频率与连续性验收调研

日期：2026-09-06

## 结论

当前四路传感器的**全程平均频率基本正常**，但本次满载受控运动 Bag 仍不能通过：

- MID-360 IMU `200.009 Hz` 对照官方 `200 Hz`，正常；
- MID-360 点云 `9.922 Hz` 对照官方典型 `10 Hz`，平均值正常；
- D435I 彩色图像 `14.451 Hz`、CameraInfo `14.516 Hz` 对照配置的
  `424x240@15 FPS`，分别达到目标的约 `96.34%` 和 `96.77%`，作为约 234 秒运行的
  平均吞吐可以接受；
- 但是 LiDAR、图像和 CameraInfo 的 rosbag 接收时间最大间隔分别约
  `1.714 s`、`1.722 s`、`1.722 s`。平均频率不能证明这些连续空档无害；官方资料也没有
  给出“嵌入式算力有限时，约 1.7 秒连续空档属于正常”的依据；
- `fixed/raw = 0.7872` 表示修复里程计相对已录到的原始里程计少约 `21.28%`。桥的职责是
  对每条 raw 生成 fixed；有限算力只能作为待验证的根因假设，不能把这种比例改判正常。

因此，不应仅放宽现有 `1.0 s` 最大 gap 或 `0.95--1.05` fixed/raw 门槛。下一步应按现有
交接执行**机器狗静止、无动作**的满载 A/B，先区分传感器源、DDS/调度、桥节点和 rosbag
writer 的损失位置，再只修改被证据指向的一层。

## 一手来源与事实

### 1. Livox MID-360

Livox 官方《Mid-360 User Manual v1.2》规格表给出点云 `Frame Rate 10 Hz
(typical value)`；手册同时说明内置 IMU 默认以 `200 Hz` 推送数据。

来源：

- [Livox MID-360 官方下载页](https://www.livoxtech.com/mid-360/downloads)
- [Livox Mid-360 User Manual v1.2（官方 PDF）](https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/Livox/Livox_Mid-360_User_Manual_EN.pdf)
  （IMU：第 3、16 页；点云 Frame Rate：第 21 页）

推论：`200.009 Hz` 与 `9.922 Hz` 均与官方频率相符；后者只比典型值低约 `0.78%`。
然而，`9.922 Hz` 是整段平均值。若消息均匀到达，10 Hz 的期望间隔约为 `0.1 s`；
`1.714 s` 接收间隔是约 17 个期望周期的跨度，必须通过 header 时间戳与接收时间戳对照
判断是源端停发、传输/调度积压，还是 recorder 侧丢失，不能由平均值抵消。

### 2. Intel RealSense D435I

Intel RealSense 官方 D400 系列数据表的 USB 3.1 Gen 1 图像格式表明确列出：
D435/D435I RGB 相机的 YUY2 彩色流支持 `424x240` 的 `6/15/30/60 FPS`。数据表也提醒，
共享 USB hub 时必须考虑带宽需求；这说明资源竞争是合理的诊断方向，但没有把长时间停流
定义为合格行为。

来源：

- [Intel RealSense D400 Series Product Family Datasheet（官方 PDF）](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf)
  （表 4-2，第 80 页；同时流说明，第 83 页）

推论：当前 `424x240@15` 是厂商正式支持模式。`14.451 Hz` 和 `14.516 Hz` 的长期平均值
接近目标，且图像与 CameraInfo 彼此接近，但约 `1.722 s` 的共同接收空档仍需归因。
按 15 FPS 估算，该跨度覆盖约 26 个帧周期；即使最终证明是 rosbag 批量写入或调度造成
接收时间聚簇，而不是相机 header 断流，也应明确标成 recorder/调度连续性问题，而不是
宣称传感器端完全正常。

### 3. Jetson Xavier NX 只提供资源背景，不提供豁免

NVIDIA 官方规格列出 Xavier NX 为 6 核 Carmel ARM CPU、384 核 Volta GPU、48 个 Tensor
Core，8 GB/16 GB LPDDR4x（59.7 GB/s），最高 21 TOPS，并有 10--20 W 功耗模式；模块存储
列为 16 GB eMMC 5.1。实际可用 CPU、内存、GPU 和存储写入能力还取决于功耗模式、同时
负载、载板和存储介质。

来源：

- [NVIDIA Jetson Xavier 系列官方规格](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/)
- [NVIDIA Xavier NX 官方技术介绍](https://developer.nvidia.com/blog/jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/)

推论：Xavier NX 确实是资源有限的嵌入式平台，四个 SQLite writer、相机、LiDAR、桥接和
推理同时运行时出现 CPU/内存/存储竞争是可信假设。但硬件规格不能证明这次损失来自算力，
更不能证明 `1.7 s` 空档或 `0.787` 比例满足定位、TF 和安全验收。应记录实际
`tegrastats`、CPU、内存和磁盘写入延迟来验证假设。

### 4. ROS 2 rosbag2：缓存、SQLite、QoS、丢消息与时间戳

ROS 2 官方资料支持以下判断：

1. 传感器数据 QoS 通常偏向及时性：Sensor Data profile 使用 best effort 和较小队列，
   可以丢失个别旧样本来优先取得最新样本。但 QoS 不兼容时可能完全不建立通信；
   “best effort 允许丢样本”不等于任意长度的连续停流可接受。
2. rosbag2 官方 README 明确指出，高率录放在低网络带宽、高 CPU、慢存储或序列化开销下
   可能出现额外丢包，并提供 per-topic QoS override 和消息丢失统计。这意味着应测量损失
   来源，而不是从 `.db3` 的一个 gap 反推传感器一定停发。
3. recorder 的缓存用于把订阅接收与存储写入解耦；`--max-cache-size 0` 表示不使用这层
   大小缓存、直接进入存储写路径。缓存满或写入跟不上仍可能丢消息，所以缓存是可做 A/B 的
   变量，不是无损保证。具体默认值和选项随 ROS 2/rosbag2 版本变化，Foxy 0.3.11 应以机器
   上 `ros2 bag record --help` 和实际启动参数为准，不能照搬 Rolling 默认值。
4. SQLite3 是 rosbag2 的官方存储插件；其默认写入设置偏性能，崩溃恢复能力存在权衡。
   SQLite 完整性为 `ok` 只证明数据库结构可读，不证明所有发布消息都已被记录。
5. 现代 rosbag2 Writer API 明确区分 `recv_timestamp` 与 `send_timestamp`，默认消息顺序
   也可区分 received/sent。当前 Foxy SQLite Bag 的 `messages.timestamp` 应保守解释为
   recorder/中间件接收侧时间；它不等同于传感器消息内的 `header.stamp`。必须解析消息
   header 对照，才能判断 `1.7 s` 是源时间断流，还是接收/写入侧的排队或丢失。

来源：

- [ROS 2 官方 QoS 概念文档](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
- [ros2/rosbag2 官方仓库与 README](https://github.com/ros2/rosbag2)
- [rosbag2 SQLite3 官方插件说明](https://github.com/ros2/rosbag2/blob/rolling/rosbag2_storage_sqlite3/README.md)
- [rosbag2 Writer 官方 API 源码](https://github.com/ros2/rosbag2/blob/rolling/rosbag2_cpp/include/rosbag2_cpp/writer.hpp)
- [rosbag2 丢失消息统计官方设计](https://github.com/ros2/rosbag2/blob/rolling/docs/design/statistics_about_lost_and_recorded_messages.md)

版本限制：以上 rosbag2 链接有部分指向当前 Rolling，用于说明官方设计语义和可诊断机制；
不能据此声称这些选项在 Foxy 0.3.11 全部存在。针对本机的改动必须先查 Foxy 的
`--help`、启动日志和已安装源码/版本。

## 当前数据逐项判定

| 指标 | 官方/配置参照 | 当前值 | 判定 |
| --- | ---: | ---: | --- |
| MID-360 IMU 平均频率 | 200 Hz | 200.009 Hz | 正常 |
| MID-360 点云平均频率 | 10 Hz（typical） | 9.922 Hz | 正常 |
| D435I image 平均频率 | 15 FPS | 14.451 Hz | 平均吞吐可接受 |
| D435I CameraInfo 平均频率 | 随 15 FPS 彩色流 | 14.516 Hz | 平均吞吐可接受 |
| LiDAR bag-receipt 最大 gap | 项目基础门 ≤1.0 s | 1.714 s | 不通过，需归因 |
| image/info bag-receipt 最大 gap | 项目基础门 ≤1.0 s | 1.722 s | 不通过，需归因 |
| fixed/raw | 桥应逐条转换；项目门 0.95--1.05 | 0.7872 | 不通过 |

这里没有把相机平均频率强行要求为精确 `15.000 Hz`：ROS 调度、起止边界和少量 best-effort
丢样本会让长期平均值略低。也没有把 `1.7 s` 空档改成平均频率合格：定位、动态 TF 和
传感器融合依赖时间连续性，平均值无法表达最坏连续空窗。

## 下一步：静止、无动作 A/B

继续保持 `MOTION_READY=NO`，机器狗趴下或稳定站立，不启动 Nav2，不发布 `/cmd_vel` 或
`/cmd_vel_lite3_safe`。

### 先做离线时间线审计

对现有四个 SQLite Bag 同时输出：

- 每个关键话题的 bag receipt gap 与消息 `header.stamp` gap；
- 1 秒 bin 的 raw/fixed/TF/LiDAR/image/CameraInfo 条数；
- 最大 gap 附近是否在多个 recorder 同时发生、随后是否出现消息突发；
- raw 到 fixed、fixed 到 TF 的局部比例，而不只看整段平均；
- marker/碰撞阶段与 gap 是否重合。

### A/B 设计

- A：桥接 + 状态录制，120 秒；记录实时 raw/fixed 计数、Bag 内计数和系统负载。
- B：桥接 + 与失败运行相同的四个 recorder，120 秒；其余条件保持相同。
- 若 A 通过、B 失败：再做单变量 recorder 对照，优先比较现有四个
  `--max-cache-size 0` writer 与合理非零缓存/合并 recorder；不要同时改桥 QoS。
- 若 A 也失败且实时 fixed/raw 偏低：才检查桥的订阅 QoS depth、callback 和 executor。
- 若实时 fixed/raw 接近 1，但 Bag 内比例偏低：不要改桥，定位 recorder QoS/cache/layout。
- 若 header gap 连续而 receipt gap 大：倾向 DDS/调度/recorder；若 header gap 也约
  `1.7 s`：倾向驱动/传感器/USB 或源端调度。

每轮保存：实际录包命令和 `--help` 对应选项、topic QoS、消息数、平均频率、gap 的
p50/p95/p99/max、header/receipt 对照、丢消息日志、`tegrastats`、内存和磁盘写入延迟、
USB/UVC/xHCI 摘要。

### 验收保持不变

- 每路传感器最大内部 gap `≤ 1.0 s`；正常状态应显著低于此值；
- fixed/raw 与 TF/fixed 均在 `0.95--1.05`；
- 无 USB disconnect、UVC failure、xHCI death；
- `/cmd_vel` 发布者为 0；正式配置与厂商源码不变。

只有静止满载连续通过后，才按交接要求在空旷场地、人工持急停/掌机保护下重跑更小幅度的
受控动作。以上结论不授权自主导航或真机速度闭环。
