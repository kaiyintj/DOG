# Indoor-7 Gazebo benchmark

更新日期：2026-09-14

本文保存室内方案、静态数据归属和可公开的当前结论。完整启动与操作命令见
[RUNBOOK.md](RUNBOOK.md)。当前只验收 Gazebo，不涉及真实机器人运动。

## 1. 范围

室内环境按以下顺序接入：

1. AWS RoboMaker Small House：先打通 chair/table 和基本导航。
2. AWS RoboMaker Bookstore：后续验证 table/shelf、多实例和 recovery。
3. AWS Hospital：最后验证 chair/table/shelf/bed 跨场景泛化。

当前代码和证据只覆盖第一项。Bookstore 和 Hospital 尚未接入，不能在结果表中记为支持。

## 2. Indoor-7 ontology

| Project class | Navigation role | Queryable |
| --- | --- | --- |
| floor | traversable | false |
| wall | structural_obstacle | false |
| door | transition | false |
| chair | object_goal | true |
| table | object_goal | true |
| shelf | object_goal_structural | true |
| bed | object_goal | true |
| unknown | unknown | false |

后端按 navigation role 判断通行性，不把 `road` 写死为唯一可通行类：

```text
outdoor13: road  -> traversable
indoor7:  floor -> traversable
```

`road` 与 `floor` 仍是不同语义类别。Profile 还绑定类别顺序、查询别名、posterior 通道数
和 checkpoint 能力；切换 profile 必须重启语义地图。

## 3. 共享架构

室内没有复制一套算法节点。`semantic_sim.launch.py` 根据 profile 选择 world 与参数：

```text
Go2 + RGB/Livox/IMU
  -> startup sensor gate
  -> FAST-LIO
  -> SegFormer full project posterior
  -> GA-BSVM
  -> semantic map/query/safe approach
  -> optional Nav2 + active perception
```

启动门等待 active 控制器、连续稳定 IMU 窗口和新鲜非空 LiDAR，再启动 FAST-LIO，避免
Go2 出生/落地瞬态进入重力初始化。组合入口中 `rviz` 参数的父子 launch 同名覆盖已修复；
`rviz:=true` 现在传给延迟启动的 FAST-LIO，而 Go2 自带的重复 RViz 保持关闭。

`navigation_enabled:=false` 仍运行 SegFormer 与 GA-BSVM，适合键盘观察和建图；
`navigation_enabled:=true` 才同时启动 Nav2、Goal Bridge 和主动感知速度门。
主链与 Nav2 就绪后可用 `ros2 run semantic_mapping semantic_query chair` 直接触发整条
查询到导航链；监听 target/goal topic 只用于诊断。
同类目标超过一个时，GA-BSVM 按候选簇评分排序；首选簇无法生成满足语义接近约束的
goal 时最多再尝试一个候选簇，仍找不到时才保持查询待重试状态。候选确认可行前不会
发布 target pose，避免 benchmark 把失败候选记录成目标。

## 4. 文件归属

### Gazebo 资产

Small House 的 world、models、许可证和来源记录属于机器人仿真 package：

```text
/home/yk/ws/src/unitree-go2-ros2/robots/configs/go2_config/
  worlds/aws_robomaker/small_house/
```

固定来源为 AWS 官方 `ros2` 分支 commit
`ff9631ca6d1db9c1ba656498151464b5ab74aafe`。原始 world 保持不变；
`small_house_no_bed.world` 是用于 world-absent 负例的单差异派生文件。

### Benchmark manifest

静态实验说明属于算法评测 package：

```text
/home/yk/ws/src/semantic_mapping/benchmark/small_house_manifest.yaml
```

它记录目标模型名与 SDF pose、机器人起点、case、随机种子和 checkpoint。通俗地说：
world 是“考场”，manifest 是“试卷和标准答案”；把试卷放在 `semantic_mapping` 中，算法
版本与评测定义才能一起审查，同时不污染 Go2 的通用场景资产。

### 运行结果

一次运行的 `result.json`、ROS logs 和 simulation log 放在：

```text
/home/yk/ws/indoor_benchmark_runs/<world>/<run-id>/<case>/
```

这里是和源码仓库分开的可重建产物目录。源码中的 `docs/` 只保存本文这种精简结论，不保存
日期化原始日志目录、诊断压缩包或大 JSON。

## 5. 模型能力门

| Checkpoint | Indoor-7 原始标签支持 | 用法 |
| --- | --- | --- |
| Cityscapes B0 | 只有 wall 重合 | 室内目标查询应 unsupported/fail closed |
| ADE20K B0 | floor/wall/door/chair/table/shelf/bed 全部存在 | 当前室内默认模型 |

当前 ADE20K revision 为 `489d5cd81a0b59fab9b7ea758d3548ebe99677da`。

“checkpoint 有标签”只表示可以合法聚合相应 posterior，不表示模型能在 Gazebo 画面中稳定
识别该物体。当前 chair 有可用预测，table 证据明显不足；没有用 CLIP 输出替代 SegFormer
正例。若后续更换或微调模型，仍必须从 raw classes 聚合完整 posterior，并重新做能力门验收。

## 6. Ground Truth 与指标

Small House manifest 当前包含 8 个 chair 和 3 个 table。pose 来自 world/SDF 顶层
model pose，属于模型原点，不是网格几何中心或可见表面。因此当前 target error 记为目标
估计到最近同类 SDF 模型原点的距离，并明确保留这一口径。

正式实验计划记录：

- `E_target`：目标估计与 GT 的距离；
- approach distance：goal 到目标真实表面或目标 cluster 的距离；
- Valid Goal Rate：goal 是否位于有效 traversable 区域；
- Collision-free Goal Rate；
- Navigation Success、SPL 和 Collision Rate；
- False Goal Publication Rate。

尚未取得独立证据的指标必须写 `not_evaluated`，不能从截图或算法输出反推 GT。

## 7. 当前 Small House 结果

| Case | 观察 | 结论 |
| --- | --- | --- |
| `validate_kitchen_close` | GT 静止位移 1.883 mm；FAST-LIO 8.648 mm | 起点与延迟初始化通过 |
| `floor_nonqueryable` | 回执 nonqueryable；0 target/goal/bridge event | negative pass |
| `bed_absent_control` | checkpoint accepted；派生 world 无 bed；0 target/goal | negative pass |
| `car_outside_profile` | 回执 unresolved；0 target/goal | negative pass |
| `table_unsupported_checkpoint` | Cityscapes 回执 unsupported；0 target/goal | negative pass |
| `chair_kitchen` | 可形成 target；一次结果距最近 chair 原点 0.647 m；无合法 approach goal | positive navigation failed |
| chair approach 诊断 | target error 0.333 m；goal 发布并被 Nav2 接受；机器人移动；结束距 goal 0.780 m | navigation failed |
| Gazebo 默认 Nav2 复测（2026-09-14） | PointCloud2 FAST-LIO、默认 `nav_sim.launch.py` 和固定起点；`SUCCEEDED`，GT 位移约 1.17 m，`/Odometry` 无跳变 | feasibility pass |
| `table_kitchen` | query accepted；没有 table 簇达到 runtime 门限 | query timeout |

chair 诊断中的 Nav2 controller 两次报告 `Failed to make progress`。现有摘要不足以把原因
归为 watchdog、速度下限或某个净空阈值。另一次带控制摘要的运行在更早的类别主导门被拒：
chair 最高融合证据概率 0.320、raw posterior 0.275，但没有 chair 为主类别的体素。这个
结果说明重复同一 case 或简单放宽控制参数不会产生新证据，应先改善观测或模型表现。

当前可准确表述为：

- Indoor-7 profile、后验传输、GA-BSVM 融合和失败关闭负例可运行；
- Small House 的 chair 目标估计和 goal-to-Nav2 链已出现；
- chair 已在 Gazebo 默认入口完成一次自动到达；重复性和多目标统计尚未通过；
- collision-free、SPL、Valid Goal Rate 和主动减速收益尚未评估。

## 8. 下一步验收

### 首轮输入表示对比

用户已报告手动建图和椅子导航跑通；上文历史失败记录不代表该次手动运行。
自动 runner 使用固定起点和观测窗口，仍需独立复验，不能把手动成功直接计入其结果。

runner 支持 `--fusion-input full_posterior`（默认）与
`--fusion-input hard_mask_confidence`。前者使用完整项目概率，后者使用现有的最高类别与
置信度接口，将剩余概率均分给其他类别。两者沿用同一模型、融合算法和导航速度链；
这是一项端到端输入表示对比，不是可靠度加权消融。两条路径的同步输入数量不同，
严格的同帧融合精度实验还需要固定输入数据。

在通用 ROS 环境准备完成后，运行一对独立仿真：

```bash
cd /home/yk/ws/src/semantic_mapping
export HF_HOME=/home/yk/ws/.cache/huggingface
export ROS_DOMAIN_ID=217
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI=http://127.0.0.1:11358
RESULT_ROOT="/home/yk/ws/indoor_benchmark_runs"
PILOT_ROOT="$RESULT_ROOT/aws_small_house/$(date +%Y%m%d_%H%M%S)_fusion_input"

ros2 run semantic_mapping run_indoor_semantic_benchmark \
  --manifest benchmark/small_house_manifest.yaml --case chair_kitchen \
  --fusion-input full_posterior --output "$PILOT_ROOT/pair_01/full_posterior"

ros2 run semantic_mapping run_indoor_semantic_benchmark \
  --manifest benchmark/small_house_manifest.yaml --case chair_kitchen \
  --fusion-input hard_mask_confidence --output "$PILOT_ROOT/pair_01/hard_mask_confidence"
```

先检查两例是否进入查询阶段，再决定是否扩到三对；这是预实验，不是正式统计结论。
每次运行保留 `result.json` 中的 `fusion_input`、实际启动命令、失败原因和原始观测。
同批次的命令记录、CSV 和简短报告放在 `PILOT_ROOT` 下，不另建顶层结果目录。
汇总应区分启动失败、未生成目标、未生成 goal、导航终止和到达，保留所有尝试。
`observed_sim_sec / observed_wall_sec` 可记录为查询期间平均仿真实时因子；当前超时按
墙钟计时，因此不同负载下的结果不可直接归因于算法。未独立测量的碰撞率、有效 goal
比例、SPL 和地图精度继续标记 `not_evaluated`。

2026-09-09 首对诊断（批次 `20260909_fp_vs_hardpilot`）已实际运行：

| 输入表示 | 次数 | 终止 | 最近同类模型原点 XY 距离 | 最终机器人到 goal 的 GT XY 距离 | 查询期间平均实时因子 |
| --- | --- | --- | --- | --- | --- |
| full_posterior | 1 | 120 秒墙钟超时 | 4.492 m | 5.407 m | 0.387 |
| hard_mask_confidence | 1 | 120 秒墙钟超时 | 0.314 m | 0.733 m | 0.420 |

两例的启动、查询接收和 Nav2 goal 接收均有记录，但均未验收到达。前者还超出当前
1.5 m 目标原点容差。查询分别在仿真 17.255 s 和 29.190 s 发出，说明相同 seed 和
warmup 下仍未获得相同的观测窗口；同一 seed 不保证语义融合接收同一组帧。
因此这张表只用于发现评测缺口，不能据此断言 hard-mask 路径优于完整概率，也不是
可靠度加权的消融证据。下一轮先固定输入或可复现的建图观测流程，核查目标与 GT 对齐，
并预先规定仿真时间预算及独立的墙钟停滞上限，再扩样本。原始结果、命令和汇总保存在
`/home/yk/ws/indoor_benchmark_runs/aws_small_house/20260909_fp_vs_hardpilot/`。

### 后续实验

1. 用已验证起点检查相机画面、chair 主导体素和 traversable 支持，获得一个可重复 chair
   到达案例。
2. 对 table 分开记录 raw posterior、项目 posterior、主导体素和聚类门，确定失败层级。
3. 对成功 case 保存小型机器可读摘要，补齐 goal validity、碰撞与 approach distance。
4. 用完全相同的 world、start、query、seed 和障碍条件比较主动减速 OFF/ON。
5. Small House 通过后再增加 Bookstore manifest；Hospital 最后处理。

当前不为了得到“成功”而关闭 checkpoint 能力门、类别主导门或 traversable 要求。Gazebo
专用 Nav2 配置保留点云和语义障碍层，但关闭 RPP 的预测碰撞 veto；真实机继续使用带碰撞
检查的 `nav2_params.yaml`。

### 公共接近点流程（2026-09-12）

按用户要求，公共语义接近点流程不再执行候选点净空、直线路径净空或回退点净空检查；仿真、M2DGR 和 Lite3 配置移除了对应参数。路径规划与碰撞检查交给 Nav2。可通行类别、置信度、物体距离和机器人同侧筛选仍生效；实机标定及运动授权门槛保持不变。Gazebo 的 `nav2_sim_params.yaml` 保留障碍层但关闭 RPP 预测碰撞 veto，已在固定起点完成一次默认入口验收；旧批次配置仅用于历史溯源，不应覆盖。
