# Lite3 上位机与机载计算选型

> 本文保留原调研日期的实验事实与建议；2026-09-29 已复核全文和引用入口。
> 网页受限及无法复算的内容见文内说明，历史建议不表示已实施。
> 现行能力与待办见 [PROJECT_STATUS](../PROJECT_STATUS.md)，操作见 [RUNBOOK](../RUNBOOK.md)。

日期：2026-09-22。9 月 29 日复核 NVIDIA、Canonical、Pi5、SDK 和 UCF 引用入口；
ROS Humble 页面被访问验证拦截，语雀页面本轮无法抓取。正文规格与选型仍为工程参考，
未对候选板卡做新验收；9 月 26 日分机组件检查见 [Lite3 交接](../LITE3_REAL_HANDOFF.md)。

## 结论

现阶段优先使用现有开发电脑承载完整语义算法，保留机器狗原有运动控制器和本地安全执行职责；最终独立机载部署优先评估 Orin NX 16GB，预算方案为 Orin Nano Super 8GB。这是基于算法结构和硬件规格的选型判断，不是两块板已通过本项目实测的结论。

项目当前主线为 FAST-LIO、SegFormer 完整 posterior、GA-BSVM、目标查询与 Nav2；CLIP 是独立对比后端。现有实机证据覆盖传感器、静止标定候选、静止语义融合与隔离里程计桥，仍缺 TF 权威、受控运动同步、厂商速度接口安全桥及导航闭环验收。参见 [当前状态](../PROJECT_STATUS.md)、[实机交接](../LITE3_REAL_HANDOFF.md) 和 [RUNBOOK 第 8 节](../RUNBOOK.md#8-lite3-实机传感器采集与上机前清单)。

## 资料与可行性依据

- 用户的[语雀入口](https://www.yuque.com/lixupeng-rquex/nwvaxd/iahtp0ydqz31yx74)可见《绝影Lite3激光版产品手册V1.0.7.pdf》附件。本次点击附件要求登录，未读取其正文；页面作者身份不等于厂商身份。
- 厂商署名的[感知开发手册 V2.1.1-0（经销商托管）](https://static.generation-robots.com/media/deep-robotics-lite3-perception-development-manual.pdf)第 5、14–15 页说明 Pro/LiDAR 感知主机使用 Xavier NX，并提供 ROS 与 UDP 转换及 Twist 速度指令通道。这证明厂商感知与运动分层的接口路线存在；旧手册不证明当前 Foxy 机型可以照搬其中服务名或 ROS 接口。
- [官方 Lite3_MotionSDK](https://github.com/DeepRoboticsLab/Lite3_MotionSDK)支持开发主机通过 UDP 通信，提供 x86/ARM 编译方式，但其示例主要操作关节。不能将它直接视为 Nav2 的底盘速度桥；本项目应核对匹配固件的高层速度接口，保留厂商步态控制。
- [Wilhelm, UCF 2025 学位论文](https://stars.library.ucf.edu/etd2024/389/)研究 Go2 的 ROS2 SLAM、导航、三维重建和无线操作，指出并发建图与可视化的计算开销、无线链路不稳定的限制。它是相近系统的可行性参考，不能作为 Lite3 语义导航已通过的证据。[仓库已有详细对照](LITE3_SEMANTIC_NAVIGATION_ALTERNATIVES_20260906.md)。

## 部署建议

1. 桌面/远程操作：电脑显示 RViz、下发目标，算法可继续在机载主机运行。仅增加操作界面通常不需要购买开发板。
2. 实验室外部计算：机载采集，电脑运行 FAST-LIO、SegFormer、GA-BSVM 和 Nav2，经机载安全桥下发速度。先验证有线数据链，再测无线延迟、断链及源时间戳。运行位置是候选设计，尚未完成在线分机验收。
3. 独立机载运行：新增计算单元承担感知、建图和导航；原运动控制器负责步态和关节闭环；电脑只用于监控和目标操作。本地安全桥必须处理超时、模式状态和断链停车。

机载已确认 Ubuntu 20.04/Foxy，电脑当前使用 Humble。不能仅设置相同 ROS_DOMAIN_ID 就认为跨发行版已兼容；应核对 RMW、QoS、自定义消息结构和时间基准，或使用显式数据桥。保留源图/点云时间戳，不以接收时间覆盖它们。依据是现有 [RUNBOOK](../RUNBOOK.md) 的双环境与传感器契约；跨机互通仍须实测。

采购时核对整套载板的千兆网络、USB 3、NVMe、输入电压、供电裕量、散热和固定方式。Orin NX 模块不是可直接上电工作的完整开发板；不要未经载板/BSP核验就替换现有 Xavier NX。对当前完整 posterior/体素地图路线，16GB 共享内存的容量余量比单看 TOPS 更有意义；CPU 融合瓶颈仍需测量。

## 验证顺序

先用同一实机 Bag 以原速重放，比对 FAST-LIO 输出连续性、推理延迟、融合积压、内存峰值和热降频；离线通过后做在线静止链路，再按实机交接完成受控运动与闭环。机载语义推理可从 1–3 Hz 的候选频率起步，但该频率不是实测保证；不降低几何定位和避障的及时性要求。桌面 Gazebo 的低 RTF 不能直接换算为机载板上的运行速度。

本次没有改变安全配置，没有连接或驱动真实底盘。继续保持 `projection_calibration_verified=false`、`goal_bridge_enabled=false`、`MOTION_READY=NO`，直到交接文件中的实机验收完成。

## 开发板规格核查

| 候选 | 官方规格要点 | 对本项目的工程判断 |
| --- | --- | --- |
| Jetson Orin NX 16GB | 8 核 CPU、16GB 共享内存，Super 总算力最高 157 sparse INT8 TOPS | 完整机载路线优先，需配套载板、电源与散热 |
| Jetson Orin Nano Super 8GB | 6 核 CPU、8GB 共享内存、67 INT8 TOPS | 预算原型候选，先测语义降频和地图内存峰值 |
| Raspberry Pi 5 | 4 核 Cortex-A76、VideoCore VII GPU、千兆网、USB3 | 采集和通信节点候选，不适合原样迁移 CUDA 语义栈 |

规格来源：[NVIDIA Super 模式对照](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)、[Nano Super 产品页](https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)、[Pi5 产品页](https://www.raspberrypi.com/products/raspberry-pi-5/)。

NX 的 157 TOPS 包含 DLA，不能用 157/67 推断 SegFormer 或 CLIP 加速比例。Nano 的 25W、NX Super 的 40W 为模块功耗模式口径，不是含载板、SSD、相机、雷达与稳压损耗的整机用电；配电须核对实际整机规格并测量。依据为上述 NVIDIA 对照。

现有 Ubuntu22.04/Humble 路线可以优先核验 JetPack6.2.2 对应的载板 BSP、PyTorch/CUDA/TensorRT 与传感器驱动组合；该版提供 Ubuntu22.04。这里是匹配项目版本的建议，不声称它是最新发行版，也不承诺现有 Python 依赖原样兼容。来源：[JetPack6.2.2](https://developer.nvidia.com/embedded/jetpack-sdk-622)、[Humble 平台支持](https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html)。

Pi5 的官方 Ubuntu 支持矩阵从24.04起，没有22.04原生支持项，因此还存在当前 Humble 环境适配成本。来源：[Canonical 支持矩阵](https://ubuntu.com/hardware/docs/boards/how-to/ubuntu_supported/raspberry-pi/)。
