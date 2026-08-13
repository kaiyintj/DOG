# 创新方法审查与 SegFormer-first 提升路线

日期：2026-08-10

审查对象：`THESIS_PROPOSAL.md` 及当前 `semantic_mapping` 代码、配置、测试和运行文档。

## 0. 结论先行

当前方案具备较完整的 ROS 2 工程链，但若按论文创新性评价，五个创新点仍偏分散。其中一部分是可靠的工程门禁，一部分是尚未校准的启发式规则。最值得收缩和加强的论文主线不是“CLIP + SegFormer + Dirichlet + Nav2 的模块组合”，而是：

> 在受限算力下，保真传递并校准 SegFormer 的任务空间类别后验，利用观测新颖性和跨帧冲突约束三维证据累积，再以同一概率表示执行目标发布、继续观测或安全拒绝。

建议将论文贡献压缩为“一个核心方法、两个支撑机制”：

1. **核心方法：概率保真、可校准、冲突感知的 SegFormer 三维证据语义地图。**
2. **支撑机制一：面向目标检索与安全通行的细类/父类双层决策。** 细类服务于 `bicycle/electric_bicycle/motorcycle` 等目标区分，父类服务于障碍与可通行安全。
3. **支撑机制二：风险选择性导航。** 系统不再只有“找到/没找到”，而是输出 `ACCEPT`、`OBSERVE`、`REJECT` 三种决策；若实现信息增益视点选择，才把该部分称为主动感知。

CLIP 不应成为论文核心方法成立或 Lite3 安全闭环成立的前提。已知类别查询、道路判断、障碍判断和目标发布均应由 SegFormer、LiDAR 几何和校准后的三维证据完成。开放词汇能力可作为查询触发的异步旁路，且只生成候选，不直接生成可执行目标。

| 优先级 | 建议 | 对论文的作用 | 对 Lite3 的影响 |
|---|---|---|---|
| P0 | 将“先 argmax 再映射”改为“完整概率先聚合再决策” | 修复不确定性链的基础信息损失 | 几乎不增加主干网络计算 |
| P0 | 增加独立校准集和项目类后验校准 | 使 Dirichlet 证据、拒绝阈值和路径风险有共同量纲 | 仅增加离线校准和少量运行时缩放 |
| P0 | 用带 Header 的原子概率包替代硬标签加最大置信度 | 保证时序正确并保留类别竞争关系 | 增加有限通信量，可用原生低分辨率/FP16 控制 |
| P1 | 增加视点新颖性折扣与冲突状态 | 防止重复帧造成过度自信，区分噪声与场景变化 | 只增加轻量地图运算 |
| P1 | 用风险后验选择目标与接近点 | 把“语义不确定性”真正接到安全决策 | 不增加视觉主干计算 |
| P1 | 增加查询状态机和单步信息增益视点选择 | 使“主动感知”名副其实 | 增加规划调用，不增加持续视觉模型 |
| P2 | 查询触发的候选区域 MobileCLIP | 保留有限开放词汇能力 | 仅在闭集无法解析时异步运行 |
| 不建议 | Lite3 上持续运行 3×4 网格 ViT-B/32 CLIP | 创新增益弱，且可能抢占 SegFormer/SLAM 资源 | 当前配置每 0.2 s 编码 12 个图像块，部署风险高 |

### 0.1 2026-08-11 P0 实施状态

本审查提出的“完整概率先聚合、带 Header 后验接口、回归测试”已经实现：

- SegFormer 在原生 decoder 分辨率对温度缩放后的原始类别执行 softmax，再按项目类别
  求和；项目后验形成后才允许 `argmax`；
- `/segformer/project_posterior` 使用一个带源图 Header 的 `sensor_msgs/Image`，编码为
  `16FC13`，以原生分辨率和 FP16 传输完整后验；源图尺寸由同 Header 的
  `/segformer/source_image` 提供；
- GA-BSVM 默认在 LiDAR 投影坐标处双线性采样完整后验，并直接送入 Dirichlet 更新；
  `segformer_use_full_posterior: false` 保留原硬标签+最大置信度基线用于消融；
- 单图工具和 CARLA 评测已改用相同的“先聚合再判决”逻辑；
- 概率守恒、多对一反例、13 类一对一映射、FP16 编解码、Header、双线性采样、概率
  往返和配置一致性均已有回归测试。

这只完成了 P0 的概率保真传输部分。`posterior_temperature` 当前仍为 `1.0`，尚未建立
独立 calibration/test 数据集，也没有得到 NLL、Brier 或 ECE 改善结论。因此 M1 的
“信息损失”已修复，但“后验已经校准”仍不得作为当前结论。

## 1. 当前实现的证据基础

### 1.1 SegFormer 主链已经具备的基础

- `segformer_node.py:168-181` 使用 SegFormer-B0、设备选择、FP16、推理周期、置信度阈值和电动自行车训练验收门槛。
- `segformer_node.py:281-313` 根据 checkpoint 的 `id2label` 判断实际可支持类别，并对自定义 `electric_bicycle` 权重再次核验训练报告。
- `segformer_node.py:410-450` 保留源图 Header，输出类别掩码、原始掩码、最大置信度、源 RGB 和可视化图。
- `segformer_training.py:453-615` 已具备 13 类微调、可复现随机种子、验证集 mIoU/分类 IoU、最佳 checkpoint 和运行时验收报告。
- `semantic_mapping_lite3_real.yaml:20-38` 已将 Lite3 主前端设为 CUDA FP16 SegFormer-B0，目标周期为 0.2 s；`ga_bsvm_node.semantic_backend` 已固定为 `segformer`。
- `ga_bsvm_node.py:1699-1746` 对 SegFormer 掩码、置信度、源图和点云进行带 Header 的近似同步，时序结构明显优于当前 CLIP 缓存接口。

这些基础应保留。它们说明 SegFormer-first 不是重新起步，而是在现有主链上修正概率接口和决策逻辑。

### 1.2 当前概率链的关键信息损失

现实现按以下顺序处理单个像素：

1. 在 checkpoint 原始类别上计算 softmax；
2. 取原始类别 `argmax` 和最大 softmax；
3. 将该硬标签映射到 13 类项目 schema；
4. 低于固定阈值的像素改为 `unknown background`；
5. GA-BSVM 根据“硬标签 + 最大置信度”重新构造一个类别分布，其余概率均匀分给另外 12 类。

代码证据为：

- `segformer_node.py:419-427` 先执行 `probabilities.max(dim=1)`，再执行 `project_lookup[raw_mask]`；
- `semantic_schema.py:79-98` 存在明显的多对一映射，例如 `road/sidewalk -> road`、`building/wall/fence -> building`、`person/rider -> person`；
- `ga_bsvm_node.py:73-118` 只使用硬类别和最大置信度，假设其余类别等概率；
- `test/test_segformer_mapping.py:191-204` 当前测试只验证这一个人工重构分布内部归一化，未验证它是否等于 SegFormer 原始项目空间后验。

这会产生两个可直接构造的反例：

- 若原始模型给 `road=0.32`、`sidewalk=0.29`、`car=0.39`，旧方法输出 `car`，但项目空间中 `road` 的总概率应为 `0.61`；
- 若原始模型的第二候选是 `bicycle`，当前方法会把剩余概率近似均匀分给所有类别，导致 `bicycle/electric_bicycle/motorcycle` 的真实竞争关系消失。

因此，当前三维 Dirichlet 图虽然形式上保存了类别分布，但 SegFormer 路线输入的是人工重构分布。若不修复这一点，“不确定性校准”“证据冲突”和“风险概率”都缺少可靠基础。

### 1.3 训练和验收仍缺少的证据

- 当前数据结构只有 `train/val`，同一验证集同时参与逐 epoch 最佳模型选择和最终门禁；缺少独立 `calibration/test`。
- checkpoint 门禁只检查 road 和四类目标的 IoU，未检查 NLL、Brier、ECE、拒绝风险、跨域性能和时序稳定性。
- 训练报告没有记录逐序列/逐场景划分，因此仅凭“图片和掩码不同名”以外的结构检查不能排除相邻帧泄漏。
- `electric_bicycle` 是稀有细类，当前默认交叉熵和总体像素统计容易被 road/background 主导；需要报告实例数、像素数和困难负样本覆盖，而不是只报告总 mIoU。
- CPU 冒烟测试不能推断 Xavier NX 上的 FP16/TensorRT 性能；当前文档已经正确指出这一边界。

### 1.4 CLIP 为什么不适合留在关键路径

Lite3 配置中的 CLIP 为 CPU `ViT-B-32`，按 3×4 网格工作，每 0.2 s 将 12 个图像块组成 batch 并执行图像编码，见 `semantic_mapping_lite3_real.yaml:1-18` 和 `clip_node.py:173-219`。即使文本特征预先缓存，持续图像编码仍与 FAST-LIO、SegFormer、Nav2 和录包共享资源；而 CLIP 网格边界比 SegFormer 粗，输出消息当前又没有源图 Header。

这不证明 CLIP 在 Xavier NX 上一定不可运行，但证明在得到实机延迟、显存、功耗和控制频率数据之前，不应把持续 CLIP 推理放进论文的必要链路。

## 2. 三个审查视角

### 2.1 方法与可证伪性审查

| ID | 严重度 | 阻断核心主张 | 受影响主张 | 代码/文本证据 | 通过条件 |
|---|---|---|---|---|---|
| M1 | Major | 是 | SegFormer 不确定性能够可靠进入 Dirichlet 融合 | `segformer_node.py:419-427`；`ga_bsvm_node.py:73-118` | 保存项目空间完整后验，并在独立校准集上报告 NLL/Brier/ECE |
| M2 | Major | 是 | 可靠性乘权能够改善校准而非仅改变阈值 | `ga_bsvm_node.py:1796-1836` 五因子直接相乘 | 在留出数据上校准权重，报告单因子、组合和等权基线及置信区间 |
| M3 | Major | 是 | 有界 Dirichlet 能避免过度自信 | `voxel_map.py:198-269` 只限制同帧总证据和总上限 | 增加跨帧视点相关折扣，并在重复静止帧/相邻帧实验中验证校准不持续恶化 |
| M4 | Major | 否 | 电动自行车权重具备可部署证据 | `segformer_training.py:548-615` 仅以同一 val 选择和验收 | 建立 train/val/calibration/test，按序列或场景隔离，测试集仅最终使用一次 |
| M5 | Major | 否 | 主动感知提升安全 | `active_perception_node.py:381-492` 只缩放已有速度 | 实现会改变观测位姿/方向的策略，或将论文术语改为“不确定性感知速度门控” |
| M6 | Minor | 否 | unknown 表示可用于统一安全决策 | `semantic_schema.py:23` 和 costmap 中的单一 unknown | 区分未观测、语义未知、证据冲突和动态过期，至少在内部状态中分离 |

### 2.2 领域新颖性审查

现有文献已经覆盖了下列单项能力：Dirichlet/贝叶斯语义体素、显式不确定性语义图、证据理论融合、开放词汇三维图、主动视点选择、语义地图重规划和共形安全规划。因此，下列说法不宜作为独立创新：

- “首次在体素中保存 Dirichlet 分布”；
- “首次将语义不确定性用于规划”；
- “首次将 CLIP 特征融合到三维地图”；
- “首次区分目标点和导航点”；
- “使用 SegFormer-B0 实现实机器人语义分割”。

仍有机会形成清晰增量的组合边界是：

1. **任务空间概率保真。** 研究原始分割标签向导航本体映射时，先聚合完整概率再校准，而不是先硬判决后映射。
2. **计算约束下的统一证据链。** 不增加大型集成网络，使用同一校准后验驱动三维融合、目标拒绝、道路安全和查询状态。
3. **相关与冲突分离。** 将重复视点的证据相关性、跨帧语义冲突和真实场景变化区分开，而不是只做证据上限和 TTL。
4. **细类检索与父类安全解耦。** 细粒度类别可拒绝，父级障碍类别仍保持保守，从而使电动自行车识别失败不等于障碍安全失败。

这四项必须作为一个闭环来验证。只实现其中某一个工具函数，创新性仍可能被审稿人判断为工程修补。

### 2.3 反方压力测试

最强反方意见是：

> 当前系统可以被描述为 SegFormer/CLIP、启发式可靠性乘积、普通 Dirichlet 累加、阈值式目标筛选和 Nav2 速度缩放的工程拼接。地图不确定性不是经独立数据校准的风险概率，主动感知没有选择观测动作，实机运动闭环尚未完成。因此，现有结果即使显示某些场景成功，也可能来自阈值调参、几何障碍层或更保守的速度，而不是所称语义方法。

要击破这个反方叙事，论文必须证明以下因果链，而不是只展示系统视频：

1. 概率保真与校准确实改善二维和三维校准；
2. 校准改善能预测目标错误、接近点风险或导航失败；
3. 风险选择性策略在相同几何层、Nav2 参数和速度预算下减少不安全决策；
4. 收益不是单纯由更高拒绝率或更慢速度换来的。

## 3. 推荐的 SegFormer-first 核心方法

### 3.1 模块一：项目类别空间的概率保真聚合

设 SegFormer 原始类别 logits 为 \(z_{u,c}\)，像素为 \(u\)，原始类别到项目类别的映射为 \(g(c)=k\)。先在独立校准集上学习温度 \(T\)，再以 log-sum-exp 聚合到项目空间：

\[
\tilde z_{u,k}=\log\sum_{c:g(c)=k}\exp(z_{u,c}/T),
\qquad
p_{u,k}=\frac{\exp(\tilde z_{u,k})}{\sum_j\exp(\tilde z_{u,j})}.
\]

这一写法等价于先对原始类别校准 softmax，再把映射到同一项目类的概率相加。它直接解决多对一标签映射的信息损失。对于自定义 13 类 checkpoint，映射退化为一对一，不增加额外假设。

首轮实现应只使用一个全局温度或极少量分组温度，避免小数据下的 class-wise scaling 过拟合。只有在 calibration/test 分离后，才比较 vector scaling、selective scaling 或更复杂校准器。

运行接口不应传整幅源分辨率 FP32 后验。建议传递：

- 原生 decoder 分辨率的 13 类项目 logits/后验；
- FP16 或其他明确记录误差的压缩表示；
- 原图尺寸、类别数、模型版本和源图 Header；
- GA-BSVM 只在 LiDAR 投影坐标处双线性采样。

这样既保留类别竞争关系，也把通信量限制在可测范围。若 ROS 2 `Image` 多通道编码兼容性不足，应单独建立带 Header 的 `SemanticPosteriorGrid` 消息，而不是退回无时间戳 `Float32MultiArray`。

### 3.2 模块二：校准后验与选择性未知状态

固定 `confidence_threshold=0.45` 只能控制最大 softmax，不能说明错误风险。建议至少保存以下量：

\[
H_u=-\sum_kp_{u,k}\log p_{u,k},
\qquad
m_u=p_{u,(1)}-p_{u,(2)},
\]

其中 \(H_u\) 是类别歧义，\(m_u\) 是 top-1/top-2 间隔。决策层区分四种状态：

| 状态 | 含义 | 安全处理 |
|---|---|---|
| `UNOBSERVED` | 该空间尚无足够观测 | 不形成肯定语义，按几何未知处理 |
| `SEMANTIC_UNKNOWN` | 观测存在，但项目本体无法可靠归类 | 几何占据不被释放；禁止作为目标 |
| `CONFLICTED` | 多帧高质量观测相互矛盾 | 降低目标置信，触发再观测或动态检查 |
| `STALE_DYNAMIC` | 动态类长期未复现 | 从当前动态层移除，保留变更记录 |

特别要避免把“语义未知”理解为“空间为空”。Nav2 的 LiDAR 障碍层继续独立负责几何占据；语义层只能提高或调节风险，不能覆盖几何障碍为自由空间。

### 3.3 模块三：细类检索与父类安全的双层语义

对于计算受限系统，无需增加第二个视觉主干。可从同一个 13 类后验派生父类概率：

\[
p(\text{two-wheeler})=
p(\text{bicycle})+p(\text{electric-bicycle})+p(\text{motorcycle}),
\]

\[
p(\text{vehicle})=p(\text{car})+p(\text{truck})+p(\text{bus})+
p(\text{two-wheeler}).
\]

- 目标检索使用细类后验，并允许在 `electric_bicycle` 与相邻细类难以区分时拒绝；
- 安全代价使用父类危险概率，因此细类混淆不会把车辆或两轮车变成可通行区域；
- 训练时可将父类一致性损失作为消融项，但不预设它一定改善跨域性能。近期层次分割研究对跨域收益并非一致，因此首先采用“由细类概率求和得到父类”的无额外运行时方案。

针对电动自行车数据，验收还应增加：

- `bicycle -> electric_bicycle`、`motorcycle -> electric_bicycle` 等定向混淆率；
- 每类实例召回率和小目标召回率，而不只是像素 IoU；
- 距离、遮挡、夜间/逆光、运动模糊分层指标；
- 困难负样本，包括普通自行车、踏板摩托、电动摩托和外观相似车辆；
- 父类 two-wheeler 的召回率，作为安全底线。

### 3.4 模块四：观测新颖性与冲突感知证据更新

当前五个可靠性因子直接相乘：

\[
r=r^{motion}r^{density}r^{range}r^{view}r^{semantic}.
\]

这隐含了尺度已校准和条件独立的强假设。建议把因子保留为可解释输入，但通过留出数据学习单调或低容量校准器 \(\hat r=f(\phi)\)，其中 \(\phi\) 包括运动、距离、密度、视野、语义熵、时间偏差和投影残差。

每个体素还应计算：

- **视点新颖性 \(n_{v,t}\)**：由视线方向变化、机器人位移、时间间隔和遮挡变化构成，重复静止帧趋近于 0；
- **证据冲突 \(c_{v,t}\)**：可用当前观测与历史预测的 Jensen-Shannon divergence 或预测集不一致度表示；
- **动态变化候选**：高质量、新颖视点仍持续冲突时，不再简单降低权重，而是进入动态/变更假设。

一个可检验的更新原型是：

\[
e_{v,t}=\min\{E_{frame},\,s\hat r_{v,t}n_{v,t}\},
\]

\[
\boldsymbol\alpha_{v,t}=\boldsymbol\alpha_0+
\lambda(\boldsymbol\alpha_{v,t-1}-\boldsymbol\alpha_0)+
e_{v,t}\mathbf p_{v,t}.
\]

冲突 \(c_{v,t}\) 不应简单乘成零，而应单独进入 `CONFLICTED` 状态和再观测逻辑。否则系统可能把真实移动目标或标定漂移误当作低质量噪声而悄悄忽略。

### 3.5 模块五：风险选择性的目标与接近点

当前接近点把体素后验先取 `argmax`，再判断 road/obstacle。改进后可直接计算：

\[
P_v^{hazard}=\sum_{k\in\mathcal H}p_v(k),
\qquad
P_v^{road}=p_v(\text{road}),
\]

并把目标决策定义为：

| 决策 | 条件示例 | 行为 |
|---|---|---|
| `ACCEPT` | 目标簇后验、父类一致性、道路支持和路径风险均通过 | 发布 `/query_target_pose` 和 `/goal_pose` |
| `OBSERVE` | 存在高潜力目标，但置信区间/预测集仍宽或冲突高 | 保留查询并选择再观测动作 |
| `REJECT` | 类别不支持、标定失败、无安全接近点、风险超过预算或超时 | 给出可解释失败原因，不发布目标 |

若采用共形预测集，可把“最危险的可行标签”用于安全距离选择，但只有在校准样本、交换性假设、场景分布和覆盖率实验满足时，才能表述为统计保证。否则只称为经验风险控制。

### 3.6 模块六：从速度门控升级为真正主动观测

当前 `active_perception_node` 不改变传感器视点或目标，只对 Nav2 已生成速度同比缩放。因此建议二选一：

1. **可行版本：** 将模块和论文术语改为“不确定性感知速度门控”，作为支撑机制而非核心主动感知创新；
2. **增强版本：** 对 `OBSERVE` 状态生成少量可达候选视点，以预期信息增益、路径风险和时间成本选择下一动作：

\[
a^*=\arg\max_a
\left[
\mathbb E\big(H(B_t)-H(B_{t+1})\mid a\big)
-\lambda R(a)-\mu C(a)
\right].
\]

学位论文阶段不必实现完整 POMDP。围绕当前目标簇生成 4 至 8 个同侧、可通行候选视点，利用可见体素数量、当前冲突、预计距离和几何净空构造单步效用，就足以形成可复现的主动闭环。

## 4. CLIP 的降级使用方案

### 4.1 推荐架构

```text
自然语言查询
    |
    +-- 可解析为已支持类别/颜色 --> SegFormer 三维证据图 --> 风险选择性目标
    |
    +-- 不在闭集本体中 ----------> 可选异步开放词汇旁路
                                      |
                                      +-- 只输出候选区域/候选词
                                      +-- LiDAR 几何与安全约束再次验收
                                      +-- 超时或资源不足则 REJECT
```

### 4.2 方案比较

| 方案 | 开放词汇能力 | 持续算力 | 安全角色 | 建议 |
|---|---:|---:|---|---|
| 纯 SegFormer + 词典解析 | 受限于训练类别 | 最低 | 主链 | **默认方案** |
| 扩充 SegFormer 细类并离线蒸馏/微调 | 有限扩展 | 与当前相近 | 主链 | **电动自行车等已知任务优先** |
| SegFormer 区域提议 + 查询触发 MobileCLIP | 查询时开放 | 低频、按需 | 候选生成 | **可选增强** |
| 伴随计算机异步 CLIP | 开放 | 机器人本地较低 | 非安全旁路 | 可做实验，必须处理断网和超时 |
| Lite3 持续 3×4 ViT-B/32 网格 | 开放 | 高 | 容易进入关键路径 | 不建议作为默认部署 |

查询触发方案应先用 SegFormer 连通域、LiDAR 聚类或当前目标假设裁剪 1 至 N 个候选区域，再运行轻量图文编码器。文本向量预先计算，图像编码仅在新查询、冲突或闭集无法解析时触发。MobileCLIP 的论文结果表明轻量图文模型可以获得更好的延迟-精度折中，但其报告平台不是本项目 Xavier NX，因此任何速度数字都必须在 Lite3 上重测。

开放词汇旁路不得直接把相似度最高区域发布为导航目标。它只能提出“可能是什么”，几何占据、同侧道路、净空、标定状态和最终动作仍由主链验收。

## 5. 实验设计重构

### 5.1 数据划分

至少建立四个互斥集合：

| 集合 | 用途 | 禁止事项 |
|---|---|---|
| train | SegFormer 参数学习 | 不用于最终报告 |
| val | epoch 选择和超参数初筛 | 不用于温度拟合和最终验收 |
| calibration | 温度、拒绝阈值、预测集或风险预算拟合 | 不用于模型选择 |
| test | 最终二维、三维和导航报告 | 仅在方案冻结后使用 |

必须按序列、Town、采集时段或物理场景分组切分，不能随机拆相邻帧。若实机标注不足，可把 CARLA/Gazebo 用于开发与校准，把完整 Lite3 场景作为外部分布测试；但不能用仿真校准结果声称实机统计保证。

### 5.2 核心实验矩阵

| 实验 | 对照 | 唯一主要变量 | 主要指标 | 要回答的问题 |
|---|---|---|---|---|
| E1 项目概率聚合 | 先 argmax 再映射；概率先聚合 | 映射顺序 | mIoU、NLL、Brier、classwise ECE、定向混淆 | 信息保真是否同时改善准确率和校准 |
| E2 后验校准 | 未校准；temperature；可选 selective scaling | 校准器 | ECE、ACE/SCE、风险-覆盖曲线、OOD 分层 | 最大 softmax 是否可作为可靠证据 |
| E3 三维证据 | 硬投票；当前伪分布 Dirichlet；完整后验；+新颖性；+冲突 | 证据更新 | 3D mIoU、voxel NLL/Brier/ECE、重复帧过度自信、簇定位误差 | 改进来自后验、相关折扣还是阈值 |
| E4 目标决策 | argmax 接近点；风险后验；风险后验+拒绝 | 决策规则 | 安全接近率、错误目标率、拒绝 precision/recall、最小净空 | 校准是否真正改变安全结果 |
| E5 主动闭环 | 无门控；速度门控；查询重试；单步信息增益视点 | 行为策略 | 成功率、碰撞率、time-to-certainty、路径长度、任务时间 | “减速”和“主动获取信息”各自贡献多少 |
| E6 计算预算 | PyTorch FP16；TensorRT FP16；可选 INT8 | 部署后端 | p50/p95 延迟、GPU/CPU、内存、功耗模式、掉帧、控制频率 | 方法是否在 Lite3 上满足实时预算 |
| E7 开放词汇旁路 | 无 CLIP；持续 ViT-B/32；按需 MobileCLIP | CLIP 调度 | 查询成功、额外延迟、峰值资源、主链抖动 | 开放词汇收益是否值得资源成本 |

### 5.3 校准指标不能只报一个 ECE

全图 ECE 容易被 road/background 大类掩盖。至少同时报告：

- NLL、Brier score、总体 ECE；
- classwise ECE/SCE，尤其是 `electric_bicycle` 和相邻两轮车；
- 边界区域与非边界区域；
- 距离、遮挡、光照、运动模糊分层；
- reliability diagram；
- selective risk-coverage 曲线和在固定风险预算下的覆盖率；
- 二维校准误差与三维目标错误、接近点风险之间的相关性。

### 5.4 公平性和统计要求

- 所有方法使用相同图像、点云、TF、Nav2 参数、最大速度和场景顺序；
- 推理更慢的方法不能通过处理更少困难帧获得隐性优势，应报告每个传感器帧的处理/丢弃状态；
- 每个随机种子和场景均保留配对结果，优先报告配对差值和 bootstrap 置信区间；
- 导航安全收益同时报告任务时间、拒绝率和路径长度，避免“永远拒绝/极慢运行”得到虚假安全提升；
- INT8 或 TensorRT 转换后重新测准确率与校准，不能只验证引擎能启动。

## 6. 建议替换的研究问题与假设

### RQ1：概率保真与校准

在相同 SegFormer-B0 主干和相同推理预算下，项目空间概率聚合与独立校准，是否比“原始 argmax 映射 + 最大置信度重构”产生更准确、校准更好且更适合三维累积的语义后验？

**H1：** 概率先聚合将减少多对一标签映射错误；校准将降低 NLL/Brier/ECE，并在固定错误风险下提高可接受像素和目标簇覆盖率。

### RQ2：相关与冲突感知三维证据

视点新颖性折扣和显式冲突状态，是否能在重复帧、机器人运动、遮挡变化和动态目标条件下减少过度自信，同时保持静态目标证据收敛速度？

**H2：** 相比仅有单帧/总证据上限的方法，新颖性折扣会抑制重复静止帧造成的置信膨胀；冲突状态会提高变化检测和错误拒绝质量，但可能延迟目标确认。

### RQ3：风险选择性目标生成

使用校准后的目标后验、父类危险概率和道路后验进行 `ACCEPT/OBSERVE/REJECT` 决策，是否比 argmax 类别和固定阈值减少错误目标与危险接近，同时保持可接受的任务覆盖率？

**H3：** 风险选择性策略将降低错误目标率、目标另一侧接近和最小净空违规；代价是一定拒绝率和观测时间增加，需通过风险-覆盖曲线而非单一成功率评价。

### RQ4：主动观测（仅在实现视点选择时保留）

单步信息增益视点选择，是否比仅减速或被动等待更快消除目标细类冲突并恢复可执行目标？

**H4：** 在遮挡和两轮车细类混淆场景中，信息增益视点将缩短 time-to-certainty 并提高目标确认率，但可能增加路径长度。

## 7. 创新点重新评级

下表是基于当前代码和近期相邻工作的审查估计，不是正式同行评审结论。

| 当前表述 | 当前创新强度 | 主要问题 | 建议定位 |
|---|---:|---|---|
| 运动质量有界可靠性融合 | 6/10 | 乘权未经校准，跨帧相关未处理 | 并入“校准、相关与冲突感知证据”核心 |
| 开放/闭集统一接口 | 4/10 | 主要是系统接口，且两前端并未联合推理 | 降为工程设计；SegFormer 为主，CLIP 为旁路 |
| 簇级目标到安全接近点 | 7/10 | 当前基于 argmax 和启发式代价 | 用风险后验和选择性决策加强为支撑贡献 |
| 不确定性到四足运动闭环 | 5/10 | 当前只是速度门控，不是主动观测 | 实现视点选择后升级；否则准确命名 |
| sim-to-real 证据门禁 | 5/10 | 很重要但更像可复现工程规范 | 保留为可信实验基础，不作为唯一算法创新 |

若完成第 3 节核心方法及 E1-E6 证据链，建议创新结构为：

1. **概率保真与校准的任务空间 SegFormer 语义接口；**
2. **观测新颖性和冲突分离的有界三维证据融合；**
3. **细类检索、父类安全与风险选择性目标生成。**

主动视点和按需开放词汇均作为增强项。这样即使 CLIP 在 Lite3 上性能不足，论文主线仍完整。

## 8. 实施顺序与停止条件

### P0：先修正确性和证据基础

1. 增加项目空间概率聚合函数，并用多对一反例做单元测试；
2. 为 SegFormer 增加带 Header 的完整项目后验接口，GA-BSVM 删除伪均匀分布路径；
3. 将数据目录扩展为 train/val/calibration/test，并增加序列级重复检查；
4. 增加 temperature scaling、NLL/Brier/ECE 和 reliability diagram 报告；
5. 在开发机与 Lite3 上记录原生 PyTorch FP16 基线，随后再决定 TensorRT/INT8；
6. 完成查询状态机的 pending、retry、timeout、cancel 和 result 状态。

**停止条件：** 若完整后验和校准未改善测试集 NLL/Brier/ECE，先排查数据泄漏、标签映射和域偏移，不进入复杂证据融合；不能通过挑选更有利阈值掩盖失败。

### P1：形成论文核心算法

1. 实现视点新颖性折扣，先验证重复帧不会无限增加有效证据；
2. 实现独立冲突状态和冲突日志；
3. 将接近点、语义代价和目标发布改为后验风险决策；
4. 完成 E1-E4 的冻结协议和消融；
5. 若风险与真实错误显著相关，再实现单步信息增益视点选择。

**停止条件：** 若不确定性不能预测目标/路径错误，则保留概率地图和拒绝能力，但撤回“风险控制提升安全”的强主张。

### P2：可选扩展

1. 使用 SegFormer 连通域 + LiDAR 聚类形成轻量实例候选和持久 ID；
2. 通过负观测清理移动目标，而不只依赖固定 TTL；
3. 评测查询触发 MobileCLIP 或伴随计算机 CLIP；
4. 只有在 E7 显示净收益且不影响主链实时性时，才把开放词汇旁路写入主要贡献。

### 不应改坏的现有设计

- 历史 TF 查询、有限重试和禁止 latest-TF 回退；
- CameraInfo、外参和投影标定失败关闭；
- `/query_target_pose` 与 `/goal_pose` 分离；
- 单一里程计/TF 权威源和单一最终速度写入者；
- 电动自行车 checkpoint 的运行时二次验收；
- Lite3 SDK 安全桥未完成前禁止实机运动闭环。

## 9. 近期原始研究定位

本轮学术检索未发现已挂载的 academic-search MCP，因而按技能的失败回退规则，使用论文官网、会议论文页、PMLR、CVF Open Access 和 arXiv 原文进行多源检索。下表已按 DOI/标题去重；预印本与已同行评审论文分开标注。

| 工作 | 状态/来源 | 与本项目的关系 |
|---|---|---|
| [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) | NeurIPS 2021 | 支持以 B0 作为轻量分割主干；不支持把使用 SegFormer 本身作为创新 |
| [On Calibrating Semantic Segmentation Models: Analyses and An Algorithm](https://arxiv.org/abs/2212.12053) | CVPR 2023 | 说明分割置信度会失准，并提供 scaling 类校准基线 |
| [ConvBKI: Real-Time Probabilistic Semantic Mapping Network with Quantifiable Uncertainty](https://arxiv.org/abs/2310.16020) | IEEE T-RO 2024 | 已覆盖 Dirichlet/贝叶斯式实时语义体素和不确定性，要求本项目明确额外贡献 |
| [Uncertainty-aware Semantic Mapping in Off-road Environments with Dempster-Shafer Theory of Evidence](https://arxiv.org/abs/2405.06265) | 2024 workshop/preprint | 已研究证据理论语义融合，提示本项目应重点处理来源冲突和校准，而非只称“证据融合” |
| [Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding](https://openaccess.thecvf.com/content/WACV2025/html/Kong_Calib3D_Calibrating_Model_Preferences_for_Reliable_3D_Scene_Understanding_WACV_2025_paper.html) | WACV 2025 | 支持把 3D 校准作为安全场景的独立评价轴 |
| [Context-Aware Replanning with Pre-Explored Semantic Map for Object Navigation](https://proceedings.mlr.press/v270/ko25b.html) | CoRL 2024 proceedings, 2025 | 已用置信与多视一致性修正错误语义地图，提示应测“地图错误后的重规划” |
| [Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation](https://arxiv.org/abs/2506.13367) | ECMR 2025 | 主动感知包含不确定性传感模型和探索动作，说明仅速度缩放不足以支撑同名主张 |
| [Understanding while Exploring: Semantics-driven Active Mapping](https://arxiv.org/abs/2506.00225) | NeurIPS 2025 | 以候选视点信息量指导主动语义建图，构成本项目主动视点模块的直接相邻工作 |
| [Safe Planning in Unknown Environments Using Conformalized Semantic Maps](https://arxiv.org/abs/2509.25124) | IEEE RA-L 2026 | 展示校准预测集如何进入语义 reach-avoid；同时提醒形式化保证依赖校准数据和假设 |
| [OVI-MAP: Open-Vocabulary Instance-Semantic Mapping](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.html) | CVPR 2026 | 已实现实时开放词汇实例地图，故实例/开放词汇不能只靠模块接入宣称新颖 |
| [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html) | CVPR 2024 | 为查询触发的轻量开放词汇旁路提供候选，但需在 Xavier NX 上独立测量 |
| [Voxeland: Probabilistic Instance-Aware Semantic Mapping with Evidence-based Uncertainty Quantification](https://arxiv.org/abs/2411.08727) | 预印本 | 已将实例级几何/语义证据和不确定性用于再分类，提示实例层可作为扩展而非主创新 |

## 10. researchwrite 八维 QA

这是对当前开题创新链的审查分数，不是对未来实验结果的评分。

| 维度 | 当前分数 | 依据与主要修正 |
|---|---:|---|
| 研究问题清晰度 | 8 | RQ1-RQ4 明确，但 SegFormer 与 CLIP 权重相近，需改为 SegFormer-first |
| 科学张力 | 8 | “算力、语义灵活性与安全可靠性”张力成立，应从模块罗列提升为概率信息损失问题 |
| 证据匹配 | 7 | 代码证据充分，近期相邻文献需进入正式综述，实机定量证据仍缺 |
| 逻辑链 | 7 | 五个创新点过宽；主动感知名称与当前动作机制不完全匹配 |
| 方法可行性 | 7 | B0、训练脚本和融合链已存在；校准集、完整后验接口和 Xavier 基准尚未完成 |
| 创新性 | 6 | 单项机制与近期工作重叠；按第 3 节重构后有望形成具体、可检验的系统方法增量 |
| 风险边界 | 9 | fail-closed、标定、TF、安全桥和失败假设边界较完整 |
| 语言质量 | 8 | 表述克制，但“主动感知”和“统一开放/闭集”需要进一步降承诺或补实现 |

平均分约为 7.5，属于可给导师讨论的方法草案，但不宜把当前五点直接冻结为最终创新声明。完成 P0 后再做一次创新性审查；完成 E1-E4 后才能确定论文标题和摘要中的强主张。

## 11. 建议论文标题

优先标题：

> 基于概率保真与冲突感知三维证据融合的四足机器人语义目标导航方法

若完成主动视点选择，可改为：

> 面向计算受限四足机器人的校准三维语义建图与风险约束主动目标导航

若最终只完成闭集词典解析，应在摘要中明确“自然语言查询受任务本体约束”，不要使用“任意开放词汇目标导航”。

## 12. 审查边界

本报告使用方法可证伪性、领域新颖性和反方压力测试三个明确区分的审查视角，但它们由同一会话顺序完成，未获得真正相互隔离的盲审环境，因此不称为三位独立审稿人意见。所有改进均是待检验研究设计，不是当前代码已经取得的性能结论。

## 13. 校验记录

原始审查生成时：

- 新文档已通过 `git diff --check`；
- 加载 ROS 2 Humble 和当前工作区环境，并禁用不兼容的用户级 pytest 插件自动加载后，`test_segformer_mapping.py` 与 `test_segformer_training.py` 共 22 项测试通过；
- 测试环境报告 SciPy 期望 NumPy `<1.25.0`，当前为 `1.26.4`。该警告未使本轮测试失败，但正式训练与定量实验前应在隔离环境中固定兼容版本；
- 当时只新增本审查文档，没有修改算法、配置、测试或用户已有工作区文件。

2026-08-11 P0 跟进实现：

- 完整功能测试（排除仓库既有 flake8/pep257 门禁）为
  `151 passed, 1 skipped`；
- `colcon build --packages-select semantic_mapping --symlink-install` 通过；
- 使用缓存的 Cityscapes checkpoint 和真实图片完成离线推理，JSON 明确记录
  `probability_mapping: aggregate_before_argmax`；该受限运行环境没有 CUDA，实际烟测
  回退 CPU，不构成 Lite3 性能结论；
- SciPy/NumPy 版本范围警告仍存在，未影响本轮测试、构建和推理。
## 14. 2026-08-11 更新：P0 概率保真实施与开题报告同步

下列审查建议的 P0 项已实现：

| P0 项 | 审查建议 | 实施情况 |
|---|---|---|
| 概率保真聚合 | 将"先 argmax 再映射"改为"完整概率先聚合" | 新增 `semantic_posterior.py`（`aggregate_project_probabilities`、`posterior_array_to_image`、`sample_posterior_bilinear`、`posterior_probabilities_to_logits`）；`segformer_core.py` 新增 `aggregate_project_probability_tensor`（GPU 端概率求和）；`segformer_node.py` 新增温度缩放 + `/segformer/project_posterior` 发布 |
| 原子消息传输 | 用带 Header 的原子概率包替代硬标签加最大置信度 | 新增 `16FC<13>` FP16 后验图，携带源图 Header；`ga_bsvm_node.py` 新增 `segformer_posterior_sync_callback` 和 `_same_image_header` 校验 |
| 单元测试 | 覆盖概率链正确性 | 新增 `test_semantic_posterior.py`，覆盖概率聚合、FP16 round-trip、双线性采样、Header 校验和 logits 转换共 9 项测试，全部通过 |

旧版 argmax 路径保留为 `segformer_use_full_posterior=false` 回归基线。完整后验路径对自定义 13 类 checkpoint 退化为恒等映射。

开题报告 `THESIS_PROPOSAL.md` 已同步更新：摘要、1.1 项目组成、2.2 开放词汇（新增概率保真讨论）、3.2 总体目标、3.3 RQ1/H1、4.1 研究内容一（SegFormer 概率链详述）、4.2、4.3、4.4（速度门控命名）、5.2 流程图、5.4 消融（新增 A11/A12）、5.5 指标（新增后验传输指标）、6.2 创新点（替换为概率保真）、7.1 代码基础和 7.3 待修正事项。

全量测试 41 项通过（segformer_mapping + segformer_training + semantic_posterior + voxel_map_algorithm）。

仍在待完成的 P0/P1 项：独立 calibration/test 数据集划分、NLL/Brier/ECE 报告、可靠性乘权校准、视点新颖性折扣、查询状态机。
