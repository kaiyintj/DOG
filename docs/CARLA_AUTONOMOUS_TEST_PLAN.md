# CARLA 自主测试计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-17
- Verification Status: UNVERIFIED（计划中的实验尚未在本轮执行）
- Version Label: carla_autotest_plan_v1

## 1. 目标与边界

本计划只覆盖当前项目中能够由 CARLA 完成的部分，并供新的 Codex 任务按阶段自主执行。
核心目标是回答四个问题：

1. 当前 SegFormer 在 CARLA 中能否稳定识别项目支持的类别；
2. 五个可靠性因素是否与点级语义错误存在可复现关系；
3. 可靠性加权是否真正改善 VoxelMap，而不只是改变点权重；
4. 结论能否跨随机种子、运动状态、地图和天气保持。

CARLA 当前可验证：

- road、person、car、truck、bus、bicycle、motorcycle 的二维识别；
- RGB 颜色与车辆类别的组合属性；
- 普通 LiDAR 与 Semantic LiDAR 真值对齐；
- semantic、range、density、view、motion 可靠性因素；
- RGB/LiDAR 人为时间偏移；
- VoxelMap 证据融合、衰减、证据上限和消融；
- 不同 Town、天气、seed、LiDAR 密度和 ego 运动条件下的鲁棒性。

CARLA **不能**完成或替代：

- 机器狗接近点、Nav2 路径、footprint、避障和步态控制；
- Lite3 SDK、真实 `/cmd_vel` 安全链、watchdog 和急停；
- 真实相机内参、畸变、LiDAR—相机外参和传感器时间同步；
- 真实网络抖动、算力负载、地面摩擦和运动死区；
- `electric_bicycle` 独立识别验收。当前 Cityscapes checkpoint 没有该输出通道，
  CARLA 车辆分类也没有独立电动车真值，禁止把 bicycle 伪装成 electric_bicycle。

## 2. 执行原则

Codex 必须遵守以下规则：

1. 按 P0 → P1 → P2 顺序执行；前置硬门失败即停止后续阶段并报告。
2. 基线测试阶段只读代码和配置，不修改源代码、不修改正式 YAML、不提交、不推送。
3. 参数扫描只能复制 YAML 到本次结果目录后修改副本，不能改
   `config/semantic_mapping_sim_livox.yaml`。
4. 不删除旧数据，不覆盖已有结果。每次运行创建带时间戳的新目录。
5. 不执行 `pkill`、`killall` 或强制清理。端口被占用时先识别进程；无法确认归属则停止。
6. 一个实验崩溃后不自动重试。保存日志、退出码和已生成文件，转入失败报告。
7. 只以 `manifest.json` 的地图、版本、seed 和传感器配置为准，不相信目录名称。
8. Semantic LiDAR 仅作为离线 GT，绝不能进入 SegFormer 输入或可靠性公式。
9. 不因为结果差就临时降低阈值、删样本或改变 GT 映射。
10. Motion V1 只作诊断；Motion V2 仍为 CARLA 实验量，不接入
    `runtime/ga_bsvm_node.py` 或实机权重。

## 3. 统一输出和状态

本次测试创建独立根目录：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
AUTOTEST_ROOT="/home/yk/ws/carla_benchmark_data/autotest_${RUN_ID}"
SEGFORMER_MODEL="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
SEGFORMER_REVISION="21b3847fae21ddee674abd31129307b6a1235bd9"
mkdir -p "$AUTOTEST_ROOT"/{logs,preflight,image,reliability,summary}
```

上述 revision 来自当前已归档报告的 `resolved_revision`。若本机无法加载该 revision，
应标记 `BLOCKED`；不能静默改用模型仓库的新版本。

Codex 持续维护：

```text
AUTOTEST_ROOT/
├── AUTOTEST_STATUS.md       # 每个阶段 PASS/FAIL/BLOCKED 与原因
├── commands.log             # 实际执行命令、开始/结束时间、退出码
├── preflight/               # Git、环境、版本、显卡和端口信息
├── image/                   # 二维识别数据集及评测结果
├── reliability/             # 点级可靠性数据集及评测结果
└── summary/FINAL_REPORT.md  # 最终结论、失败项和实机遗留项
```

状态只能使用：

- `PASS`：硬门和预先声明的验收条件全部满足；
- `FAIL`：实验完成，但质量指标不满足；
- `BLOCKED`：环境、CARLA、模型或资源阻止实验完成；
- `INVALID`：数据格式、GT 对齐、类别支持或实验控制变量不成立；
- `NOT_RUN`：因前置硬门失败而未运行。

代码质量测试默认超时 15 分钟；30 帧采集默认超时 20 分钟；正式采集和模型评测
默认超时 120 分钟。超时后可终止该次子进程，但不能自动重跑。

## 4. 优先级总览

| 优先级 | 阶段 | CARLA server | 目的 | 通过后才能做 |
| --- | --- | --- | --- | --- |
| P0 | A. 环境与代码硬门 | 不需要 | 确认代码、入口、依赖和自动化测试可运行 | 所有实验 |
| P0 | B. 旧数据回归 | 不需要 | 确认当前版本能重现已有结果 | 新数据采集 |
| P0 | C. 30 帧双链路冒烟 | 需要 | 验证 image/reliability 采集与评测端到端 | 正式矩阵 |
| P0 | D. SegFormer 类别能力 | 需要 | 确认闭集类别是否满足导航目标检索要求 | 可靠性结论 |
| P0 | E. 静态点级可靠性基线 | 需要 | 验证 GT、投影、后验和校准指标有效 | 因素消融 |
| P1 | F. Range/Density/View/Semantic 单因素实验 | 需要 | 判定每个因素是否值得保留或调参 | 组合权重 |
| P1 | G. Motion/时间偏移实验 | 需要 | 检查运动和时延条件下的可靠性 | Motion 决策 |
| P1 | H. VoxelMap 六组消融 | 离线评测 | 判断权重是否改善三维融合 | 参数冻结 |
| P1 | I. 多 seed 鲁棒性 | 需要 | 排除单次随机结果 | 阶段性结论 |
| P2 | J. 多 Town/天气扩展 | 需要 | 评估域变化 | 论文正式结果 |
| P2 | K. CLIP 与性能分析 | 视任务而定 | 非当前 SegFormer 主链的补充实验 | 不阻塞实机 |

## 5. P0-A：环境与代码硬门

### A1. 冻结并记录当前状态

```bash
cd /home/yk/ws/src/semantic_mapping
git status --short --branch
git rev-parse HEAD
git diff --stat
git diff | sha256sum
df -h /home/yk/ws
nvidia-smi
python3 -c "import carla; print(carla.__file__)"
```

允许工作树已有修改，但必须记录，且本计划执行过程中不得覆盖或清理这些修改。

### A2. 构建和离线测试

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select semantic_mapping
source install/setup.bash

cd /home/yk/ws/src/semantic_mapping
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test/test_carla_benchmark.py \
  test/test_carla_reliability.py \
  test/test_carla_reliability_capture.py \
  test/test_carla_evaluate_reliability.py \
  test/test_reliability_factors.py \
  test/test_reliability_voxel_ablation.py \
  -q

ament_flake8 semantic_mapping/carla test/test_carla*.py \
  test/test_reliability_factors.py \
  test/test_reliability_voxel_ablation.py
ament_pep257 semantic_mapping/carla
```

### A3. 入口验证

```bash
ros2 run semantic_mapping carla_capture_benchmark --help
ros2 run semantic_mapping carla_evaluate_benchmark --help
ros2 run semantic_mapping carla_capture_reliability --help
ros2 run semantic_mapping carla_evaluate_reliability --help
```

硬门：

- 构建成功；
- 六组定向测试无失败；
- flake8、pep257 无失败；
- 四个入口均能显示帮助；
- CARLA Python client 可导入；
- CUDA 正式评测时，`nvidia-smi` 和 PyTorch CUDA 均可用。

任何一项失败：阶段 A=`FAIL/BLOCKED`，停止，不自动改代码。

## 6. P0-B：已有数据回归

优先复用以下已存在数据，只验证当前 evaluator 是否仍可重现，不重新解释为新证据：

| 条件 | 数据目录 |
| --- | --- |
| stationary | `/home/yk/ws/carla_benchmark_data/reliability_baseline_fixed_20260813_155014` |
| constant velocity | `/home/yk/ws/carla_benchmark_data/motion_cv_constantmode_20260813_211718` |
| turning | `/home/yk/ws/carla_benchmark_data/motion_turning_clean60_20260813_214753` |

注意：这些 manifest 实际记录 `Carla/Maps/Town10HD_Opt`，不能因为目录历史命名而称为
Town05 数据。

对三个数据集分别执行 point-only：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

ros2 run semantic_mapping carla_evaluate_reliability \
  --dataset DATASET \
  --output "$AUTOTEST_ROOT/reliability/regression_PROFILE" \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  --segformer-model "$SEGFORMER_MODEL" \
  --segformer-revision "$SEGFORMER_REVISION" \
  --time-offset-ms 0 20 50 100 150 \
  --frame-tolerance-ms 6 \
  --device cuda \
  --fp16 \
  --max-points-per-frame 1000 \
  --posterior-storage none \
  --no-voxel-eval
```

turning 数据再执行一次 `--voxel-eval`。与归档的
`docs/results/carla_motion_v2_20260813/` 比较：

- 样本流、accepted frame 数、paired common points 应完全一致；
- 确定性指标允许绝对差 `<= 1e-6`；
- 推理耗时不作为数值复现硬门；
- Motion V2 仍应得到“机制响应正确、Voxel 主验收失败”的结论。

如果当前模型 revision 与缓存变化导致结果不一致，标记 `INVALID`，记录模型 revision，
不能把新结果覆盖旧归档。

## 7. P0-C：CARLA server 与 30 帧双链路冒烟

### C1. server 前置检查

```bash
pgrep -af 'CarlaUE4-Linux-Shipping'
ss -ltnp | grep ':2000'
```

- 已有 CARLA 且 client 可连接：复用；
- 端口空闲：可以启动一个 headless CARLA；
- 端口被非 CARLA 进程占用：`BLOCKED`，不杀进程；
- 禁止同时启动第二个 CARLA server。

启动命令：

```bash
cd /home/yk/ws/third_party/CARLA_0.9.16
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  -carla-port=2000
```

连接硬门：client/server 都为 0.9.16，并且能列出地图：

```bash
python3 -c "
import carla
c=carla.Client('127.0.0.1', 2000)
c.set_timeout(10)
print('client:', c.get_client_version())
print('server:', c.get_server_version())
print('current:', c.get_world().get_map().name)
print('maps:', c.get_available_maps())
"
```

### C2. image smoke

采集 30 帧、仅运行 SegFormer：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

IMAGE_SMOKE="$AUTOTEST_ROOT/image/smoke_seed42"
ros2 run semantic_mapping carla_capture_benchmark \
  --host 127.0.0.1 \
  --port 2000 \
  --map Town05 \
  --output "$IMAGE_SMOKE" \
  --cache-dir /home/yk/ws/.cache/carla \
  --frames 30 \
  --save-every 1 \
  --vehicles 12 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --width 640 \
  --height 480 \
  --grid-rows 4 \
  --grid-cols 6 \
  --seed 42 \
  --stationary-ego

ros2 run semantic_mapping carla_evaluate_benchmark \
  --dataset "$IMAGE_SMOKE" \
  --output "$IMAGE_SMOKE/results/segformer_smoke" \
  --backend segformer \
  --segformer-model "$SEGFORMER_MODEL" \
  --segformer-revision "$SEGFORMER_REVISION" \
  --device cuda \
  --fp16 \
  --save-overlay-every 1
```

硬门：

- `manifest.json` 存在且 `format_version=2`；
- `frames` 非空；
- 实际地图与请求地图一致；
- 评测产生 `report.json`、`per_frame.csv` 和 overlay；
- JSON 中没有 NaN/Infinity；
- 退出码为 0。若数据完整但进程非零，标记 `FAIL`，不得自动视为成功。

### C3. reliability smoke

采集 30 帧 stationary、静态目标，并执行点级和体素评测：

```bash
RELIABILITY_SMOKE="$AUTOTEST_ROOT/reliability/smoke_seed42"
ros2 run semantic_mapping carla_capture_reliability \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-manager-port 8000 \
  --town Town05 \
  --weather ClearNoon \
  --seed 42 \
  --num-frames 30 \
  --warmup-frames 10 \
  --output-dir "$RELIABILITY_SMOKE" \
  --cache-dir /home/yk/ws/.cache/carla \
  --fixed-delta 0.01 \
  --camera-tick 0.01 \
  --lidar-tick 0.05 \
  --width 640 \
  --height 480 \
  --fov 90 \
  --lidar-channels 64 \
  --lidar-range 50 \
  --lidar-points-per-second 600000 \
  --lidar-rotation-frequency 20 \
  --vehicles 12 \
  --walkers 0 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --stationary-ego \
  --no-moving-targets

ros2 run semantic_mapping carla_evaluate_reliability \
  --dataset "$RELIABILITY_SMOKE" \
  --output "$AUTOTEST_ROOT/reliability/smoke_eval" \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  --segformer-model "$SEGFORMER_MODEL" \
  --segformer-revision "$SEGFORMER_REVISION" \
  --time-offset-ms 0 20 50 100 150 \
  --device cuda \
  --fp16 \
  --voxel-eval
```

硬门：

- manifest 为 `carla_reliability_v1`；
- normal/Semantic LiDAR、RGB、IMU、metadata 数量匹配；
- `alignment.aggregate_matched_fraction >= 0.98`；
- `alignment.rejected_frame_count = 0`；
- supported points > 0；
- ECE、NLL、Brier、accuracy 为有限值；
- `per_point.csv`、`factor_summary.csv`、`report.json` 存在。

## 8. P0-D：SegFormer 类别能力

### 假设

默认 checkpoint 能较稳定地识别 car/truck/bus/bicycle/motorcycle，但小型两轮目标和
CARLA 域偏移可能导致 motorcycle/bicycle 明显退化。

### 设计

- Town：先选择运行时确实可用的 Town05；manifest 必须确认实际载入 Town05；
- seed：11、42、73；
- 每个 seed：400 world frames，`save-every=5`；
- 车辆：40，五类尽量各 8；行人：25；
- ego：stationary；
- 分辨率：640×480；
- 使用同一数据分别评测 SegFormer，CLIP 留到 P2。

当前采集器没有调用 CARLA 的 `world.set_pedestrians_seed()`，因此带 walker 的场景不能
声称逐 actor 位姿的 bitwise 重现；车辆与模型指标仍按 manifest 和逐实例记录审计。
如果正式论文要求完全重现行人位置，应另开代码修改任务补齐该 seed，再重新采集，不能在
本次只读基线阶段临时改代码。

每批数据必须满足：

- `valid_for_vehicle_benchmark=true`；
- `missing_vehicle_classes=[]`；
- 五个类别都有可见实例；
- 不支持类保留为 unsupported，不计入主指标。

### 第一轮参数扫描

对相同数据只改变：

```text
segformer-confidence = 0.35, 0.45, 0.55
```

不能同时改变模型、图像分辨率或实例最小像素数。比较：

- road/person/vehicle IoU；
- 各车辆类 detection rate、class accuracy；
- 总体和逐类误报/漏报；
- 推理耗时和显存；
- overlay 失败案例。

工程目标（模型准备度，不是数据有效性门）：

- 每个目标类别至少 30 个可用实例观测；
- road IoU `>= 0.70`；
- vehicle IoU `>= 0.25`；
- 每个导航目标类 detection rate 和 class accuracy 均建议 `>= 0.70`；
- 任一类低于目标时，该类=`FAIL`，不能用总体均值掩盖。

若 bicycle 或 motorcycle 未达到 0.70，应结论为“当前 checkpoint 不足”，而不是继续
无限降低阈值。若要改善，另开模型微调任务。

## 9. P0-E：静态点级可靠性基线

### 控制条件

- stationary ego；
- `--no-moving-targets`；
- `--walkers 0`，避免行人生成随机性干扰车辆/道路可靠性基线；
- offset 0 为主；
- fixed delta 10 ms，camera 10 ms，LiDAR 50 ms；
- 64 channels、600k PPS、50 m；
- seed 11、42、73；每个 seed 300 帧。

### 主要指标

- GT alignment coverage 和 rejected frames；
- project posterior accuracy；
- ECE、NLL、Brier；
- `r_semantic` 的 correctness AUROC、AURC 和 reliability gap；
- 按类别、距离、密度、视场位置分层的样本量；
- unsupported 和 dropped 样本流。

硬门仍为 GT 对齐 `>=0.98`、无拒绝帧、指标有限。质量判据：

- `r_semantic` correctness AUROC 应高于 0.5；
- 三个 seed 的方向应一致；
- 若 bootstrap 95% CI 包含 0.5，只能记为“证据不足”，不能宣称有效。

## 10. P1-F：四个单因素控制实验

每次只改变一个自变量，其余保持 P0-E 基线不变。

| 因素 | 控制变量 | 建议条件 | 关注结果 |
| --- | --- | --- | --- |
| Range | 目标距离/可见距离分箱 | 0–5、5–10、10–20、20–30、30–50 m | error、`r_range`、coverage |
| Density | LiDAR channels/PPS | 64/600k、32/300k、16/150k | local density、error、`r_density` |
| View | 相机中到边缘位置 | view radius 五等分 | error、`r_view` |
| Semantic | posterior 熵和温度 | temperature 0.75、1.0、1.25、1.5 | ECE、NLL、Brier、AUROC |

参数扫描使用 YAML 副本并采用 one-factor-at-a-time：

```text
range_scale_m:              10, 20, 30
density_scale:               4,  8, 16
view_edge_penalty:           0, 0.2, 0.4
semantic_confidence_floor: 0.05, 0.15, 0.30
```

选择规则：

1. 先比较点级 correctness AUROC/AURC 和 reliability gap；
2. 再检查 VoxelMap 是否改善；
3. 不能只凭公式单调或 weighted accuracy 选择；
4. 至少 2/3 seed 方向一致，且没有明显降低少数类 coverage，才列为候选；
5. 本阶段只给“CARLA候选值”，不能写入实机配置。

`view_edge_penalty=0` 是当前基线。若非零值不能在 VoxelMap 层改善结果，则继续保持 0。

## 11. P1-G：Motion 与时间偏移

### 设计

使用相同 seed 和静态目标，分别采集：

- stationary；
- `constant_velocity --ego-speed 5.0`；
- `turning --ego-throttle 0.45 --ego-steer 0.35`。

每组评测 offset：`0, 20, 50, 100, 150 ms`，使用严格 paired common points。

必须分别报告：

- 实际 offset 和 timing error；
- camera 相对转角、相对平移；
- Motion V1、Motion V2；
- 各 offset paired accuracy；
- correctness AUROC/AURC；
- 点级与体素级结论。

验收分两层：

1. 机制验收：stationary 不应随 offset 无故降权；constant/turning 的 Motion V2 应随
   实际相对运动增加而下降。
2. 效果验收：加入 motion 后，错误体素相对正确体素的 uncertainty gap 不得下降，
   correctness AUROC 不得退化，coverage 绝对下降不超过 0.5 个百分点。

当前归档结果只通过机制验收、未通过效果验收。因此自动复现后仍应保持：

```text
Motion V2 = CARLA diagnostic only
不得接入正式 runtime
不得继续只扫 rotation/translation scale 来“调到通过”
```

下一代 motion 研究应另立计划，优先考虑遮挡、深度边界、跨时刻语义一致性和真实重投影
残差。

## 12. P1-H：VoxelMap 六组消融

必须使用同一批点、同一 posterior、同一时间戳、同一 GT support，只改变可靠性权重：

```text
none
semantic
semantic_range
semantic_range_density
semantic_range_density_view
full (+ motion)
```

正式主实验先使用 stationary ego、静态目标，避免动态世界坐标拖影混淆可靠性因素。
每组报告：

- GT voxel、covered、missing 和 coverage；
- covered accuracy、all-GT accuracy；
- correct/wrong mean uncertainty；
- wrong-minus-correct uncertainty gap；
- correctness AUROC或相关性；
- 每类 coverage/accuracy（若报告支持）。

候选组合只有同时满足以下条件才能进入下一阶段：

- all-GT accuracy 不比 `none` 低超过 1 个百分点；
- coverage 不比前一组合低超过 0.5 个百分点；
- uncertainty gap 或 correctness AUROC 至少一个稳定改善，另一个不明显退化；
- 至少 2/3 seed 同方向；
- 没有依靠删除 unsupported、远距离或困难类别制造改善。

若点级指标改善但 VoxelMap 不改善，结论是“诊断相关、不可作为当前融合权重”。

## 13. P1-I：多 seed 决策

P0-D 至 P1-H 的任何参数冻结都至少使用 seed 11、42、73。汇总时：

- 不把每个点当成独立实验重复；
- 以 frame 或 seed 做 cluster/bootstrap；
- 报告均值、标准差和 95% CI；
- 同时保留逐 seed 数值；
- 样本不足或单类全对/全错时，AUROC 写 null，不写 0。

参数候选必须满足 `>=2/3 seed` 方向一致。正式声称稳定性时应要求三个 seed 均无方向
反转。

## 14. P2-J：跨地图和天气

只有 P1-I 完成后再执行。先通过 `client.get_available_maps()` 确认地图存在，再选择：

- Town03；
- Town05；
- Town10HD_Opt。

天气从 CARLA 实际可用 preset 中选择 ClearNoon、CloudyNoon、WetNoon。不要假设不存在的
preset。正式矩阵为：

```text
3 Town × 3 weather × 3 seed
```

先做 30 帧 smoke，再扩到 300 帧。每格固定模型 revision、传感器参数和候选可靠性
参数。输出逐格结果，不能只报 27 格总体平均。

通过标准：P1 选择的参数在大多数格子保持同方向；若只在一个 Town 或天气有效，应明确
标为场景特定，不能作为通用参数。

## 15. P2-K：非阻塞补充实验

以下项目不阻塞 SegFormer 主链：

- 在同一 image 数据上运行 CLIP，比较开放词汇类别、颜色和网格定位；
- CPU/CUDA、FP32/FP16 推理耗时和显存；
- overlay、错误实例图库和论文图；
- `posterior_temperature` 校准曲线；
- 长序列 VoxelMap 内存、TTL 和 evidence decay 压力测试。

CLIP 结果不能替代 SegFormer 像素分割；性能实验不能与 CARLA server 同时占用 GPU 后
直接比较推理耗时。

## 16. 最终交付

Codex 完成可执行阶段后生成 `$AUTOTEST_ROOT/summary/FINAL_REPORT.md`，至少包括：

1. Git commit、dirty 状态和 diff hash；
2. CARLA client/server、地图、天气、seed、GPU、模型 revision；
3. 每个阶段的 PASS/FAIL/BLOCKED/INVALID；
4. 每条实际命令、退出码、耗时和产物路径；
5. 数据完整性、GT alignment 和样本流；
6. SegFormer 总体及逐类结果；
7. 四个可靠性因素和 motion 的点级结论；
8. 六组 VoxelMap 消融结果；
9. 跨 seed/Town/天气结论；
10. 仍需 Gazebo 或实物完成的项目；
11. 明确区分“数据有效”“模型通过”“参数候选”和“可接入实机”。

不得自动修改正式参数。报告最后只能给候选值和证据，等待用户决定是否另开实现任务。

## 17. 给新 Codex 任务的启动指令

可以把下面内容直接发送给新 Codex：

```text
请阅读并严格执行：
/home/yk/ws/src/semantic_mapping/docs/CARLA_AUTONOMOUS_TEST_PLAN.md

从 P0-A 开始顺序执行，只进行 CARLA 与离线评测，不运行 Gazebo 或实机节点。
先记录当前 Git dirty 状态，保护所有已有修改。基线期间不得修改源代码或正式 YAML，
不得提交或推送，不得删除旧数据，不得使用 pkill/killall，不得自动重试失败实验。

每个阶段维护 AUTOTEST_STATUS.md、commands.log 和最终 FINAL_REPORT.md。遇到硬门失败、
端口归属不明、CARLA版本不匹配、CUDA不可用或数据无效时停止后续阶段并报告，不要为了
获得好结果降低阈值或改变真值。先完成 P0；P0 全部通过后继续 P1。P2 的 27 组正式
矩阵先估算时间和空间，在报告中列出后等待我确认再执行。
```

## 18. 参考实现与既有证据

- [CARLA_IMAGE_BENCHMARK.md](CARLA_IMAGE_BENCHMARK.md)
- [CARLA_RELIABILITY_BENCHMARK.md](CARLA_RELIABILITY_BENCHMARK.md)
- [RUNBOOK.md](RUNBOOK.md)
- [CARLA Motion V2 实验记录](results/carla_motion_v2_20260813/README.md)
- `semantic_mapping/carla/carla_capture_benchmark.py`
- `semantic_mapping/carla/carla_evaluate_benchmark.py`
- `semantic_mapping/carla/carla_capture_reliability.py`
- `semantic_mapping/carla/carla_evaluate_reliability.py`
- `semantic_mapping/carla/reliability_voxel_ablation.py`
