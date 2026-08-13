# Semantic Mapping 系统级审查报告

- 审查日期：2026-08-11
- 审查范围：semantic_mapping 核心节点、配置、启动文件、Nav2 接口，以及 FAST-LIO、CHAMP、Livox 等相关依赖
- 审查方式：静态代码审查、跨文件配置核对、ROS 2 隔离启动检查、自动测试
- 修改说明：本报告仅记录审查结论，未修改项目代码

## 1. 总体结论

当前项目已经具备完整的“语义感知—三维融合—语义代价地图—Nav2 规划—主动感知限速—底盘控制”架构。仿真配置下，Nav2 速度链路和 ROS 2 端点关系基本正确；但 Lite3 实机配置仍处于安全闭锁的联调状态，不建议直接进行无保护自主导航。

主要结论如下：

- Nav2 当前速度链路已实测确认：controller_server → /cmd_vel_nav → velocity_smoother → /cmd_vel → active_perception。
- 仿真最终输出 /cmd_vel_champ 能与 CHAMP 对接。
- Lite3 最终输出 /cmd_vel_lite3_safe 在仓库内没有下游订阅者，实机不会形成运动闭环。
- 默认实机配置与 use_sim_time=true 的组合不合理，是关键启动风险。
- 实机 projection_verified=false，程序会主动禁止语义目标生成。这是正确的安全设计，但说明系统尚未完成标定验收。
- 体素地图具有 TTL、半径和数量上限，不会无限增长；但动态证据衰减、自由空间清除和长期权重累计仍需改进。
- SegFormer 全后验链路的时间戳一致性较好；CLIP 链路存在跨帧缓存和模型空间不一致风险。
- 自动测试结果为 151 passed、1 skipped、2 failed；两个失败均来自 flake8/pep257 风格检查，功能测试通过，但缺少实机端到端测试。

综合判断：仿真架构处于“可集成验证”阶段，Lite3 实机处于“标定与安全执行桥待完成”阶段，尚不满足无人值守自主导航条件。

## 2. 系统架构与数据流

    Camera ──> SegFormer / CLIP ───────────────┐
                                               │
    LiDAR point cloud ─────────────────────────┼─> GA-BSVM
    IMU + TF(odom→base_link) ──────────────────┘       │
                                                       v
                                          Evidential voxel map
                                          ├─ semantic cost map ─> Nav2 global costmap
                                          ├─ entropy cloud ─────> Active perception
                                          └─ query/approach goal ─> NavigateToPose

    Nav2 controller
        └─ /cmd_vel_nav
             └─ velocity_smoother
                  └─ /cmd_vel
                       └─ active_perception
                            ├─ /cmd_vel_champ       仿真
                            └─ /cmd_vel_lite3_safe  实机，当前无执行桥

核心组件：

| 组件 | 主要职责 | 审查结论 |
|---|---|---|
| [ga_bsvm_node.py](../semantic_mapping/ga_bsvm_node.py) | 点云投影、语义证据融合、目标检索、安全接近点、代价地图 | 主体逻辑完整，但时间一致性、动态清除和接近点安全模型仍有限 |
| [voxel_map.py](../semantic_mapping/voxel_map.py) | Dirichlet 证据体素、特征与颜色融合、地图裁剪 | 地图有界；证据与特征的长期更新机制需要改进 |
| [segformer_node.py](../semantic_mapping/segformer_node.py) | 像素级类别后验和不确定性 | 全后验输出和 Header 传递设计较好 |
| [clip_node.py](../semantic_mapping/clip_node.py) | 图像块 CLIP 特征与类别 logits | 输出没有 Header，无法严格关联原始图像帧 |
| [active_perception_node.py](../semantic_mapping/active_perception_node.py) | 根据路径语义风险、运动模糊和曲率调节速度 | 控制律合理，但缺少输入新鲜度约束和实时隔离 |
| [nav_goal_bridge_node.py](../semantic_mapping/nav_goal_bridge_node.py) | 将 Goal 转为 Nav2 Action | 缺少完整的抢占、取消、重试和超时状态机 |
| [nav_with_remap.launch.py](../launch/nav_with_remap.launch.py) | Nav2、主动感知和 Goal Bridge 启动 | 端点总体正确，默认时间配置需修正 |
| [nav2_params.yaml](../config/nav2_params.yaml) | Nav2 控制、局部/全局代价地图和规划器参数 | odom/base_link 体系一致，语义层已接入全局代价地图 |

隔离 ROS_DOMAIN_ID 的运行检查确认：

- /cmd_vel_nav 由 controller_server 发布、velocity_smoother 订阅。
- /cmd_vel 由 velocity_smoother 发布、active_perception_node 订阅。
- active_perception_node 发布最终安全速度。
- nav_goal_bridge_node 订阅 /goal_pose 并连接 /navigate_to_pose。
- 审查环境没有真实 odom→base_link，因此 Nav2 生命周期没有进入 active；这是预期现象，不是启动配置崩溃。

## 3. 阻塞性和高风险问题

### 3.1 P0：默认实机配置与仿真时钟冲突

[启动文件](../launch/nav_with_remap.launch.py)默认选择 Lite3 实机语义配置，但 use_sim_time 默认值为 true。实机没有 /clock 时，ROS Timer、TTL、推理节流和 TF 时间查询可能停滞，Nav2 也可能无法正常激活。

建议将仿真和实机拆分为独立启动入口；实机默认 use_sim_time=false，并在启用仿真时间却收不到 /clock 时主动报错。

### 3.2 P0：实机语义投影尚未验收

[Lite3 配置](../config/semantic_mapping_lite3_real.yaml)要求 CameraInfo 和零畸变输入，并将 projection_verified 设为 false；外参仍需要实机确认。[GA-BSVM](../semantic_mapping/ga_bsvm_node.py)会在投影未验证时禁止目标和接近点输出。

应完成以下验收后再打开开关：

- 使用真实 CameraInfo，或完整处理 K、P、ROI 和 binning。
- 使用标定得到的 LiDAR—相机静态 TF，避免手工维护两套外参。
- 在近、中、远距离分别验证重投影误差。
- 将标定文件哈希、传感器序列号和验收结果绑定保存。

仿真外参与 Go2 URDF 的计算结果一致，但不能替代 Lite3 实机标定。

### 3.3 P0：实机安全速度没有执行端

Lite3 配置将输出设为 /cmd_vel_lite3_safe，但仓库范围内没有发现该话题的订阅者。当前闭环结束在 ROS 话题，尚未到达 Lite3 SDK 或底盘控制器。

建议实现唯一的实机执行桥，并确保：

- /cmd_vel_lite3_safe 恰好有一个执行订阅者。
- 执行桥具有 100–300 ms 指令超时自动停车。
- 节点退出、网络断开和急停时发送零速度。
- 禁止其他节点绕过安全节点直接控制底盘。

### 3.4 P1：主动感知可能使用过期风险数据

[active_perception_node.py](../semantic_mapping/active_perception_node.py)缓存路径和熵点云，但没有最大年龄限制；熵回调还会在单线程执行器中进行 Python 遍历和 KD-tree 构建。

可能表现为旧路径或旧语义风险继续影响当前速度，大点云回调阻塞 /cmd_vel 回调，以及队列中的旧 Twist 在延迟后才被输出。

应为路径、熵点云、IMU 和速度指令增加时间戳与新鲜度门控；控制回调与点云处理使用不同 callback group 或执行线程，控制输出增加独立 watchdog。

### 3.5 P1：IMU 缺失时可靠性错误地趋向完全可信

运动可靠性计算在没有 IMU 或没有对齐样本时返回可靠度 1。同时 IMU 使用普通 reliable QoS，而许多传感器驱动采用 best-effort。

应改用与发布端匹配的 SensorDataQoS；缺失或过期 IMU 时使用保守可靠度或暂停融合，并发布 IMU 年龄、匹配率和可靠度诊断。

### 3.6 P1：Goal Bridge 缺少事务语义

[nav_goal_bridge_node.py](../semantic_mapping/nav_goal_bridge_node.py)存在以下风险：

- 发送前移除 pending goal，发送异常时会丢失目标。
- 没有保存和取消当前 Action GoalHandle。
- 新目标不会显式抢占旧目标。
- 多个结果回调可能交错。
- latest_dispatched_sequence 没有真正阻止旧结果影响状态。

建议改为“一个活动目标 + 一个最新待发送目标”的状态机：新目标到达时取消旧目标，等待取消确认后发送最新目标，并增加 goal ID、超时、重试、反馈和结果发布。

### 3.7 P1：动态语义残影仍可能长期偏置

[voxel_map.py](../semantic_mapping/voxel_map.py)只在体素再次被击中时衰减旧证据，衰减强度受传感器帧率影响；没有基于射线的自由空间清除。特征和颜色权重也会长期累加。

当前 TTL 会移除长期未观测的动态体素，因此不是永久幽灵；但反复经过的动态物体可能持续刷新 last_observed 并维持错误语义。

建议使用类别相关的连续时间衰减，结合自由空间射线、视角新颖度和 capped EMA 更新颜色与特征。

### 3.8 P1：安全接近点与 Nav2 碰撞模型不一致

安全接近点主要使用 XY KD-tree、目标同侧判断和直线路段净空，没有使用 Nav2 实际 inflation cost、机器人 footprint、地面高度、坡度、台阶和局部实时障碍物。

建议直接调用 Nav2 costmap/footprint collision checker，或统一构建 traversability layer，再对候选姿态执行规划可达性验证。

### 3.9 P1：部署依赖未完整声明

[setup.py](../setup.py)只声明 setuptools，实际运行还依赖 NumPy、SciPy、PyTorch、OpenCLIP、Transformers 和 Pillow。当前环境还出现 SciPy 1.8.0 要求 NumPy 小于 1.25、但实际 NumPy 为 1.26.4 的警告。

应提供锁定版本的 CPU/GPU 环境文件，并在启动时输出模型、CUDA、依赖和 checkpoint 哈希。

## 4. 跨文件配置一致性

| 项目 | 当前情况 | 判定 |
|---|---|---|
| Nav2 速度链 | /cmd_vel_nav → velocity_smoother → /cmd_vel → active_perception | 正确，已运行验证 |
| 仿真最终速度 | Active 输出 /cmd_vel_champ，CHAMP 订阅 | 正确 |
| Lite3 最终速度 | Active 输出 /cmd_vel_lite3_safe | 阻塞：无执行订阅者 |
| M2DGR 速度配置 | /cmd_vel_nav → /cmd_vel | 只适用于不启动 Nav2 的离线流程 |
| Nav2 坐标系 | global_frame=odom、robot_base_frame=base_link | 正确 |
| FAST-LIO | 发布 /Odometry、odom→base_link 和 body-frame point cloud | 与当前 Nav2 体系一致 |
| CHAMP TF | 默认 publish_odom_tf=false | 默认不会与 FAST-LIO 重复 |
| 语义代价地图 | /semantic_cost_map，transient-local；Nav2 global StaticLayer 同名订阅 | 正确 |
| 旧 /map 发布 | legacy map 默认关闭 | 当前不存在已确认冲突 |
| 点云高度过滤 | 相对机器人 base 高度过滤 | 已实现 |
| 道路类别 | 按类别名称解析 | 已修复，不再假定 class 0 是 road |
| Lite3 点云帧 | 配置留空，优先使用消息 Header | 合理，但需验证驱动 Header |
| 仿真外参 | 与 Go2 URDF 计算结果一致 | 正确 |
| 实机时间 | 实机 profile 配合默认 use_sim_time=true | 错误默认值 |
| AMCL 参数 | 参数文件中存在，但当前 launch 不启动 AMCL | 不影响当前链路，但容易误导维护者 |

## 5. 三十项算法与工程核查

| # | 核查项 | 结论 |
|---:|---|---|
| 1 | Nav2 速度链 | 通过；M2DGR profile 不得与 Nav2 launch 混用 |
| 2 | Active 节点是否启动 | 通过，launch 默认启动 |
| 3 | map/odom 路径转换 | 通过；缺 TF 时不能工作 |
| 4 | /map 重复发布 | 通过；legacy map 默认关闭 |
| 5 | frame 配置 | 通过；Nav2、FAST-LIO 使用 odom/base_link 一致 |
| 6 | 高度过滤 | 通过；按机器人相对高度过滤 |
| 7 | 动态物体幽灵 | 部分通过；有 TTL，但无自由空间清除 |
| 8 | 证据衰减 | 未通过；只在再次击中时衰减且与帧率相关 |
| 9 | 地图无限增长 | 通过；有 TTL、半径和最大体素数限制 |
| 10 | road=class 0 假设 | 通过；已按类别名查找 |
| 11 | CLIP 时间一致性 | 未通过；输出无 Header，可能跨帧 |
| 12 | SegFormer 后验同步 | 全后验模式通过；legacy 模式仅近似同步 |
| 13 | 相机模型和外参 | 部分通过；实机外参未验收 |
| 14 | projection_verified | 安全门控正确；实机当前不会生成目标 |
| 15 | 点云 frame override | 有风险；错误配置可能覆盖正确 Header |
| 16 | 历史 TF 队列 | 有界，但延迟帧可能晚于新帧融合 |
| 17 | 可靠性融合 | 数值有界但未经校准，IMU 缺失时 fail-open |
| 18 | 单帧/总证据上限 | 通过；缺少时间和视角新颖度建模 |
| 19 | posterior/evidence 使用 | 通过；查询和不确定性使用逻辑合理 |
| 20 | entropy 语义 | 命名不精确；实际是融合不确定性的等效熵 |
| 21 | 路径风险 | 使用均值和高分位数；仍缺距离、z 和新鲜度权重 |
| 22 | scale 变化率 | 通过；按实际经过时间限制 |
| 23 | 曲率限速 | 基本通过；纯旋转和执行器死区仍需处理 |
| 24 | Action 抢占/取消 | 未通过；状态机不完整 |
| 25 | 查询和目标聚类 | 部分通过；仍是一次性查询和 XY 链式聚类 |
| 26 | 安全接近点 | 未通过；未与 Nav2 footprint/costmap 统一 |
| 27 | Python 性能 | 高风险；点循环、KD-tree 和全图序列化是瓶颈 |
| 28 | QoS | 大部分匹配；IMU 可能不匹配 |
| 29 | 端到端入口 | 内部话题一致；实机闭环尚未完成 |
| 30 | 运行依赖 | 未通过；Python/GPU 依赖未完整声明和锁定 |

## 6. 建议增强的研究创新

当前“语义体素地图 + 不确定性限速 + 安全接近点”具有较好的系统组合创新，但部分模块仍偏启发式。建议重点发展以下方向。

### 6.1 时间—视角—可观测性联合的证据体素融合

将类别相关连续时间衰减、射线自由空间、观察视角新颖度和动静态分类统一到 Dirichlet 证据更新中。相比简单 TTL，这更容易形成独立的方法学贡献。

### 6.2 基于风险分布的导航控制

沿规划路径计算 CVaR、置信上界或碰撞概率约束，使速度控制直接关注高风险尾部，而不是只使用平均熵乘比例系数。

### 6.3 信息增益驱动的主动感知

让机器人选择能够减少目标类别后验不确定性的观察位姿，并综合旅行代价和碰撞风险。这样比被动降速更符合主动感知的理论定义。

### 6.4 语义目标与可通行性联合接近位姿优化

同时优化类别置信度、颜色匹配、可见性、地面坡度、机器人 footprint 净空、导航代价和最终朝向，替代当前同侧加直线净空启发式。

### 6.5 可校准的跨模态不确定性

分别建模 SegFormer、CLIP、投影误差、运动模糊和距离误差，再通过温度标定或概率校准融合，并使用 ECE、NLL 和 Brier Score 验证。

[训练代码](../semantic_mapping/segformer_training.py)目前使用同一验证集完成训练选择和最终验收，没有独立 calibration/test 集；验收报告也没有与 checkpoint 哈希绑定。正式论文实验前应修正这一点。

## 7. 性能与长期运行风险

| 风险 | 当前估计 | 建议 |
|---|---|---|
| 体素内存 | 无特征约 1.37 KB/体素，25 万体素约 327 MiB；带 512 维 CLIP 特征约 842 MiB，未计字典开销 | 降低实机上限，使用紧凑数组或哈希结构及 FP16 特征 |
| DDS 带宽 | 25 万点、约 16 B/点时单条点云约 4 MB；两条 10 Hz 流量理论上约 80 MB/s | 发布局部 ROI、降采样、合并字段或使用进程内通信 |
| CPU | Python 逐点投影、聚类、KD-tree 和全地图发布 | NumPy 向量化，重计算移到工作线程或 C++ |
| 控制实时性 | Active 使用单线程执行器，点云处理可能阻塞速度回调 | 控制与感知分离 callback group，测量 p95/p99 延迟 |
| 长期语义漂移 | 动态证据无自由空间清除，颜色和特征权重长期累计 | 连续时间衰减、射线清除、EMA 和权重封顶 |
| GPU与模型 | 当前环境 CUDA 不可用；默认 checkpoint 对细粒度类别能力有限 | 明确 CPU/GPU profile，验证类别集合并锁定 checkpoint |
| 科研复现 | 依赖未锁定、验证集复用 | 提供容器或 lockfile、随机种子、独立测试集及模型哈希 |

上述内存和带宽数字是工程估算，正式报告中应使用 Xavier 或实机上的 RSS、GPU 显存、DDS 吞吐和控制延迟实测结果。

## 8. 实施优先级

### P0：完成实机闭环前必须解决

1. 将 Lite3 默认设置为 use_sim_time=false，仿真和实机使用独立 launch。
2. 完成 LiDAR—相机标定、重投影验收和单一 TF 来源配置。
3. 实现 /cmd_vel_lite3_safe → Lite3 SDK 安全执行桥及超时停车。
4. 为 Active 增加路径、熵点云、IMU 和速度指令的新鲜度门控。
5. 建立传感器中断、TF 中断、节点崩溃和网络中断时自动停车的端到端测试。

### P1：影响算法正确性和论文结论

6. 将 CLIP 输出改为带 Header、模型标识和 query ID 的原子消息。
7. 重构 Goal Bridge，完成取消旧目标、发送最新目标、异常重试和结果去重。
8. 使用 Nav2 footprint/costmap 和地面可通行性验证安全接近位姿。
9. 实现连续时间证据衰减、自由空间清除、动态类别模型和特征权重封顶。
10. 建立独立 train/validation/calibration/test 划分，并绑定评估报告和 checkpoint 哈希。

### P2：性能和研究增强

- 向量化投影与聚类，限制语义点云发布区域。
- 在目标路径上引入 CVaR 或置信上界风险控制。
- 实现基于期望信息增益的主动观察位姿规划。
- 增加 Xavier 上连续运行、内存、温度、控制延迟和 DDS 吞吐测试。

## 9. 最终判断

项目不存在需要推翻的架构性问题。下一阶段的重点不应只是继续增加语义模块，而应转向：

1. 完成实机时间、标定和速度执行闭环。
2. 保证跨传感器时间一致性与控制实时安全。
3. 统一语义接近点和 Nav2 的碰撞、可通行性模型。
4. 将启发式不确定性处理升级为可校准、可消融验证的方法。
5. 建立可复现的独立测试集、依赖环境和端到端实验体系。

## 10. 审阅后整改记录（2026-08-11）

本节记录本报告形成后的实际代码整改。第 1–9 节保留审阅当时的原始结论，不能把其中
“未通过”直接解释为当前代码状态，也不能把本节的软件修复解释成实机运动授权。

| 原问题 | 当前状态 | 已实施或仍缺少的内容 |
|---|---|---|
| 3.1 实机 profile + 仿真时钟 | 已修复 | 通用入口默认墙钟；新增 `nav_sim.launch.py` 和 `nav_lite3_real.launch.py`，分别给出仿真/实机安全默认值 |
| 3.2 实机投影未验收 | 仍阻塞 | `projection_calibration_verified` 保持 `false`；仍需真实 CameraInfo、外参、序列号和多距离重投影验收，未伪造解锁 |
| 3.3 Lite3 安全速度无执行端 | 仍阻塞 | 仓库未知云深处 SDK 命令接口；仍需唯一 `/cmd_vel_lite3_safe` 执行桥、100–300 ms 本地 watchdog、退出/断网/急停零速 |
| 3.4 Active 使用过期输入 | 已修复 | Path、熵点云、IMU 均按源时间戳检查最大年龄；异常未来时间/时钟回跳失败关闭；独立 callback group、四线程执行器、深度 1 命令队列和 250 ms 稳态时钟 watchdog |
| 3.5 IMU fail-open/QoS | 已修复 | Active 与 GA-BSVM 均改用 SensorDataQoS；GA-BSVM 无 IMU/无对齐样本时返回保守可靠度 0.2，并记录恢复/缺失诊断 |
| 3.6 Goal Bridge 事务 | 已修复 | 严格一个 active + 一个 latest pending；保存 GoalHandle/UUID；显式取消旧目标，等待终态后再发送最新目标；异常/拒绝重试、超时、反馈节流、结果去重和 JSON 状态话题 |
| 3.7 动态残影/长期权重 | 部分修复 | 证据改为连续时间衰减，周期 prune 会老化未命中体素且不刷新 TTL；颜色、特征和总权重有界并持续适应。因当前融合接口没有射线可见性输入，自由空间清除仍未实现 |
| 3.8 接近点与 Nav2 模型不一致 | 仍待实现 | 现有 footprint、inflation、同侧 road 和直线净空保护保留；仍缺对每个候选调用 Nav2 costmap/footprint 或规划可达性验证 |
| 3.9 依赖未声明/冲突 | 代码侧已修，环境待切换 | `setup.py` 声明 NumPy/SciPy 与模型 extras；新增 SegFormer/CLIP requirements，公共锁定组合为 NumPy 1.26.4 + SciPy 1.11.4。没有擅自修改当前全局环境，所以现有 SciPy 1.8 警告仍会出现 |
| 训练集复用与指标溯源 | 已修复 | 支持独立 calibration/test；val 只选模，test 才能形成正式验收；无 test 明确标为非正式；报告写入 checkpoint 逐文件及整体 SHA-256，检查时重新计算 |
| CLIP 输出无 Header | 仍待实现 | 当前主路线 SegFormer 的 Header 全后验不受影响；CLIP 对比路线仍需自定义原子消息，不能用另一个松散 Header 话题冒充原子同步 |

新增的安全行为需要特别注意：Active 的 `/perception_mode` 若为 `STALE[path]`、
`STALE[entropy_cloud]` 或 `STALE[imu]`，速度会立即降至最低倍率；若 250 ms 没有新速度
命令则输出零速。这是输入失效的明确诊断，不应通过关闭新鲜度门控来“让机器人动”。

整改后的自动验证结果：

- 等价完整测试集合：`184 passed, 1 skipped`；
- flake8 与 pep257：全部通过；
- `colcon build --base-paths ~/ws/src/semantic_mapping --symlink-install --packages-select semantic_mapping`：通过；
- `nav_sim.launch.py` 与 `nav_lite3_real.launch.py`：均可从安装空间解析。

尚不能通过自动测试关闭的 P0 是实机 LiDAR—相机标定和 Lite3 SDK 安全执行桥。二者需要
真实硬件参数、厂商控制接口和现场验收；在完成前，实机仍应保持失败关闭。
