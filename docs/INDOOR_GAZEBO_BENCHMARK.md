# Indoor-7 Gazebo benchmark

核对日期：2026-09-29

本文保存室内方案、静态数据归属和可公开的当前结论。完整启动与操作命令见
[RUNBOOK.md](RUNBOOK.md)。当前只验收 Gazebo，不涉及真实机器人运动。

## 最近证据与适用版本

9 月 16 日和 18 日 chair 录包已有 `SUCCEEDED`；9 月 18 日导航墙钟 178.305 s、
仿真时间 71.080 s，成功后停止。两次路线和录制内容不同，不是性能对照。
完整出处和当前代码待验收项见 [当前状态](PROJECT_STATUS.md)。
下文早期 pilot 是对应日期的诊断，不能覆盖后续结果，也不能证明最新代码已重跑 Gazebo。
现有 Gazebo 配置关闭 RPP 预测碰撞 veto，无碰撞到达仍需独立统计。

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
/home/yk/ws/src/semantic_mapping/config/benchmark/small_house_manifest.yaml
```

它记录目标模型名与 SDF pose、机器人起点、case、随机种子和 checkpoint。通俗地说：
world 是“考场”，manifest 是“试卷和标准答案”；把试卷放在 `semantic_mapping` 中，算法
版本与评测定义才能一起审查，同时不污染 Go2 的通用场景资产。

### 运行结果

一次运行的 `result.json`、ROS logs 和 simulation log 放在：

```text
/home/yk/ws/indoor_benchmark_runs/<world>/<run-id>/<case>/
```

这里是和源码仓库分开的实验产物目录。可以重新开展实验，但已有录包和批次记录是独立历史证据，不能用新结果覆盖。源码中的 `docs/` 只保存本文这种精简结论，不保存
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

## 7. 已有证据与当前结论

- 历史负例覆盖 floor 不可查询、无 bed 场景、profile 外 car、checkpoint 不支持 table。
- chair 已有 9 月 16/18 日到达与停止记录；出处、适用版本及性能基线见
  [当前状态](PROJECT_STATUS.md#保留的室内实验依据)。
- table 历史正例未形成满足门限的目标簇；不能从 accepted 回执推断检测或导航成功。
- 10 月 3 日当前代码 `chair_kitchen_mapping_pitch` 完成目标、到达与停止的功能验收：
  目标 XY 误差 .068 m，GT 距 goal .254 m，5 sim s 停稳通过。结果与全部失败尝试见
  [本机批次](results/indoor_gazebo_20261003/README.md)。
- 重复性、碰撞率、SPL、有效接近点及主动减速收益统计尚未完成；远处投影错目标仍需处理。

已删除的早期 pilot 不作为当前结论依据。手动建图后导航与 runner 固定起点/观测窗口
是不同实验流程，结果分别记录。

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
  --manifest config/benchmark/small_house_manifest.yaml --case chair_kitchen \
  --fusion-input full_posterior --output "$PILOT_ROOT/pair_01/full_posterior"

ros2 run semantic_mapping run_indoor_semantic_benchmark \
  --manifest config/benchmark/small_house_manifest.yaml --case chair_kitchen \
  --fusion-input hard_mask_confidence --output "$PILOT_ROOT/pair_01/hard_mask_confidence"
```

先检查两例是否进入查询阶段，再决定是否扩到三对；这是预实验，不是正式统计结论。
每次运行保留 `result.json` 中的 `fusion_input`、实际启动命令、失败原因和原始观测。
同批次的命令记录、CSV 和简短报告放在 `PILOT_ROOT` 下，不另建顶层结果目录。
汇总应区分启动失败、未生成目标、未生成 goal、导航终止和到达，保留所有尝试。
`observed_sim_sec / observed_wall_sec` 可记录为查询期间平均仿真实时因子；10 月 3 日起
正例采用 90 秒仿真查询预算、300 秒墙钟上限，避免低实时因子压缩有效运动时间。
成功后继续观察 5 秒仿真时间，记录新的最终控制命令和 GT 水平位移；最终线/角速度
不超过 0.001 m/s、rad/s 且位移不超过 0.05 m 才判停止通过。查询记录的
`terminal_sim_sec` / `terminal_wall_sec` 是到达终态耗时，`observed_*` 还包含停止窗口。
这一功能链通过不替代正式统计。未独立测量的碰撞率、有效 goal
比例、SPL 和地图精度继续标记 `not_evaluated`。

早期两组预实验未获得相同的观测窗口，也均未验收到达；用户已决定删除这批不适合作
当前结论的记录，因此不再列入结果表。下一轮先固定输入或可复现的建图观测流程，
核查目标与 GT 对齐，并预先规定仿真时间预算及独立的墙钟停滞上限，再扩样本。

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

### 公共接近点流程

公共接近点筛选保留可通行类别、置信度、物体距离和机器人同侧约束，不做候选点或直线路径
净空检查。路径交由 Nav2 规划执行；当前仿真关闭 RPP 碰撞 veto，不能将执行当作净空
证明。室内接近距离下限 1.3 m、期望 1.4 m、搜索半径 2.5 m，要求机器人同侧落点；
实机标定与运动门禁保持关闭。

Go2 平地配置还按机器人 odom z 减名义机身高度 0.225 m 估计地面，接近点 z 须处于
上下 0.10 m 范围。floor 语义不能替代高度条件；高处桌腿等点在实际运行中曾被误选。
这项估计要求查询时机身水平，不适用于楼梯或未知坡面；其它 profile 默认不启用。

`chair_kitchen_mapping_pitch` 是独立建图流程：FAST-LIO 水平初始化并预热后，CHAMP
机身在 3 sim s 内低头 +0.30 rad、保持 10 sim s、3 sim s 恢复，至少等待 5 sim s 并
验证实际姿态恢复水平后查询。传感器外参、真实 MID360 CSV、模型和 Nav2 容差不变。
此流程旨在补齐水平静止时约 2.34 m 内的地面盲区，与原静止查询用例分别报告。
