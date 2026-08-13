# CARLA 三维语义可靠性基准

## 1. 目的、范围与旧基准的边界

本基准独立评估 GA-BSVM 中一次三维语义观测的可信度是否与真实的点级
语义正确性相关。它建立下列关系，而不是重新训练或替换 SegFormer：

```text
普通 LiDAR + RGB -> SegFormer project posterior -> 点级预测
                                                |
Semantic LiDAR（仅 GT） -----------------------+-> 正确/错误
IMU、量程、密度、视场、posterior 熵 -----------> 五个 reliability factor
```

它与 [CARLA_IMAGE_BENCHMARK.md](CARLA_IMAGE_BENCHMARK.md) 的边界如下：

| 项目 | CARLA image benchmark | 本文的 reliability benchmark |
| --- | --- | --- |
| 主要问题 | 二维图像分类、颜色、网格定位与像素 IoU | 三维 LiDAR 点的语义正确性与可靠度关系 |
| 算法输入 | RGB | RGB、**普通** LiDAR、IMU |
| 真值 | semantic / instance camera | Semantic LiDAR 的标签与 actor ID；相机 GT 也保留用于审计 |
| 模型 | CLIP、SegFormer | SegFormer project posterior；不运行 CLIP |
| 输出 | `per_frame.csv`、实例指标、叠加图 | `per_point.csv`、`factor_summary.csv`、`report.json`、可选体素消融 |

旧 image benchmark 保持原有入口 `carla_capture_benchmark` 和
`carla_evaluate_benchmark`，不与本基准共享数据格式或覆盖其结果。新入口为
`carla_capture_reliability` 和 `carla_evaluate_reliability`。

**严格约束**：Semantic LiDAR 的 `object_tag` 和 `object_idx` 只能在离线评测中
产生 GT，绝不进入 SegFormer、可靠度公式、点级预测或 VoxelMap 输入。GT 也绝不
用于改正模型预测。`electric_bicycle`、`chair`、`bench` 等默认 Cityscapes
checkpoint 没有输出通道的类别会被保留为 unsupported，不能被伪装成有效预测。

## 2. 构建与启动 CARLA

安装的 CARLA Python API 必须和 CARLA server 同版本（本项目验证目标为 CARLA
0.9.16）。先构建并加载工作区：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select semantic_mapping
source install/setup.bash

python3 -c "import carla; print(carla.__file__)"
```

在终端 1 启动 CARLA；实际目录不同则只改第一行路径。无显示器或 CUDA/PRIME
机器可保留后面的 NVIDIA 环境变量：

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

另开终端确认 server 真正就绪后再采集：

```bash
python3 -c "
import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10)
print('CARLA:', client.get_server_version())
print('地图:', client.get_world().get_map().name)
"
```

如果 CARLA 输出 `Waiting for master`、2000 端口已占用，或 client 超时，先停止旧的
`gzserver`/CARLA 进程并只保留一个 CARLA server。不要把 Gazebo 端口问题当成
CARLA 数据问题。

## 3. 采集可复现实验数据

采集器固定 CARLA 的 synchronous mode、`fixed_delta_seconds`、天气、随机种子和
Traffic Manager 随机种子。它同时记录 RGB、semantic camera、instance camera、
normal LiDAR、Semantic LiDAR 和 IMU；每个完整样本要求这些传感器在同一 CARLA
frame 和 timestamp 上对齐。

终端 2 的一个完整、可复现实验命令如下：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash

RUN_DIR="/home/yk/ws/carla_benchmark_data/reliability_town05_seed42_$(date +%Y%m%d_%H%M%S)"

ros2 run semantic_mapping carla_capture_reliability \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-manager-port 8000 \
  --town Town05 \
  --weather ClearNoon \
  --seed 42 \
  --num-frames 300 \
  --warmup-frames 40 \
  --output-dir "$RUN_DIR" \
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
  --lidar-upper-fov 10 \
  --lidar-lower-fov -30 \
  --vehicles 24 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --walkers 12 \
  --ego-motion-profile autopilot \
  --moving-targets

printf '%s\n' "$RUN_DIR" \
  > /home/yk/ws/carla_benchmark_data/LATEST_RELIABILITY_DATASET.txt
```

`--camera-tick` 和 `--lidar-tick` 必须是 `--fixed-delta` 的整数倍，且 camera
tick 不得大于 LiDAR tick；否则采集器拒绝开始，以免伪造时间同步。

### 可控条件

- 静态 ego：在上例最后改为 `--stationary-ego`；它覆盖 motion profile，用于
  range、view、密度或基本语义误差条件。
- 恒速和转向：使用 `--ego-motion-profile constant_velocity --ego-speed 5.0`，或
  `--ego-motion-profile turning --ego-throttle 0.45 --ego-steer 0.35`。
- 静态目标：使用 `--no-moving-targets`。这是比较体素融合时较容易解释的条件。
- 环境：改变 `--town`、`--weather` 与 `--seed`，但每次都将命令、CARLA 版本和
  `manifest.json` 一并存档。
- LiDAR 密度：可控制 `--lidar-channels`、`--lidar-points-per-second`、
  `--lidar-rotation-frequency` 与垂直 FOV；不能通过改正式 `density_scale` 伪造
  密度实验。

采集结束后可先检查格式和同步信息：

```bash
python3 -c "
import json
data = json.load(open('$RUN_DIR/manifest.json'))
print('格式:', data['format'], data['format_version'])
print('CARLA:', data['carla_server_version'])
print('Town:', data['town'], 'weather:', data['weather_preset'])
print('帧数:', data['saved_frame_count'], 'RGB:', data['rgb_frame_count'])
print('ego:', data['ego_motion_profile'], 'moving targets:', data['moving_targets'])
print('传感器:', ', '.join(data['sensors']))
"
```

采集完成后可关闭 CARLA 再进行模型离线推理，避免它与 CUDA SegFormer 争抢显存。

## 4. 离线评测与时间偏移实验

评测使用项目的 SegFormer project ontology 和 posterior 聚合逻辑。模型缓存建议放在
工作区：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

RUN_DIR="$(cat /home/yk/ws/carla_benchmark_data/LATEST_RELIABILITY_DATASET.txt)"

ros2 run semantic_mapping carla_evaluate_reliability \
  --dataset "$RUN_DIR" \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  --time-offset-ms 0 20 50 100 150 \
  --device cuda \
  --fp16 \
  --voxel-eval
```

无 CUDA 时改为 `--device cpu` 并去掉 `--fp16`。评测不启动 ROS 图、FAST-LIO、
GA-BSVM 节点或 Nav2；`--params-file` 仅用于读取与正式节点相同的可靠度和 VoxelMap
参数，**不会修改 YAML 或在线运行参数**。

正的 `--time-offset-ms d` 表示对 lidar 时刻 `t_l` 选用目标时刻
`t_l - d` 附近、且绝不晚于 `t_l` 的历史 RGB。投影同时使用这张 RGB 自己记录的
历史相机 pose，而点坐标仍来自 `t_l` 的 normal LiDAR pose。因此它测量的是
受控的 RGB/LiDAR 时延下真实几何错位，而不是只更换图像内容。候选帧以距离目标
timestamp 最近为准，相同距离选较早帧；超出时间容差、没有历史帧或投影/匹配异常时
该样本 fail-closed 并在报告中计数，不会借用未来 RGB。

建议先以 `0, 20, 50, 100, 150 ms` 跑完整序列，再做下列实验矩阵。每个格子至少换
3 个 seed；Town、天气、种子和每格样本数必须报告。

| 目的 | 控制条件 | 应观察的轴 |
| --- | --- | --- |
| 基线 | stationary ego、`--no-moving-targets`、offset 0 | posterior/熵、range、density、view |
| Range | 静态 ego，改变车辆距离或可见段 | range 与 `r_range` |
| View | 保持距离近似，改变相机中心至边缘位置 | view radius 与 `r_view` |
| Density | 改 LiDAR channels/PPS，其他配置固定 | local density 与 `r_density` |
| Motion | constant/turning/autopilot × 五个 offset | IMU 统计、offset、`r_motion` 与 accuracy |
| 鲁棒性 | 多 Town × 多天气 × 多 seed | 上述关系能否跨场景保持 |

单独让 ego 运动并不必然形成投影错误；因此 motion 结论必须同时报告 IMU 统计和
时间偏移，不能把“移动更快”直接解释成“语义更错”。

## 5. 采集数据目录与 schema

数据集根目录的 `manifest.json` 格式为 `carla_reliability_v1`。典型结构为：

```text
reliability_.../
├── manifest.json
├── rgb_index.json
├── rgb/00001234.png
├── semantic_camera/00001234.png
├── instance_camera/00001234.png
├── lidar/00001234.npz
├── semantic_lidar/00001234.npz
├── imu/imu.csv
└── metadata/00001234.json
```

`manifest.json` 固化 CARLA server/client 版本、Town、天气、seed、同步步长、ego
motion profile、目标是否运动、发现到的 `CityObjectLabel` ID、实际生成 actor 的
fine class、全部 sensor blueprint/属性以及帧 metadata 文件列表。每个
`metadata/*.json` 包含各 sensor 的 frame、simulation timestamp、local-to-world
变换、文件路径、IMU 值、ego snapshot 和同帧同步检查。

`lidar/*.npz` 保存 `xyz`、`intensity`、`channel` 与每 channel 点数；
`semantic_lidar/*.npz` 保存同样的 `xyz`、`cos_inc_angle`、`object_idx`、
`object_tag`、`channel`。坐标约定为 CARLA sensor-local/world 的 x 前、y 右、z 上；
变换矩阵均为 local-to-world。

Semantic camera 和 instance camera 不替代 Semantic LiDAR GT，但被保留为数据审计
和可视化依据；normal LiDAR 和 Semantic LiDAR 也不是数组下标天然对应的同一条射线。

## 6. GT 对齐、投影与类别边界

### 6.1 双 LiDAR 的 fail-closed 对齐

CARLA normal ray-cast LiDAR 是算法点云；Semantic LiDAR 是独立 ray-cast sensor，
两者可能因为 ray-cast 实现而产生数量或顺序差异。评测器不会按 index 拼接，而是：

1. 两个云分别移除非有限点；
2. 在相同 sensor transform 和 LiDAR 参数下，对 normal → semantic 与
   semantic → normal 都做最近邻；
3. 仅保留**互为最近邻**且欧氏距离不超过容差的配对；
4. 记录 pair 数、未匹配数、平均/P95/最大距离和 match fraction；
5. 若帧的 match fraction 低于评测阈值，整帧拒绝（fail-closed），不把错误配对
   当成 GT。

每个通过的 normal 点才从配对的 semantic return 读取 `object_tag` 和
`object_idx`。若 actor ID 出现在 `manifest.actor_id_map`，其实际 spawned
`benchmark_class` 用于车辆细类；否则按 manifest 中发现的 CARLA tag 映射。道路和
sidewalk 合并为 project `road`，building/wall/fence 合并为 `building`，vegetation
合并为 `tree`，rider 合并为 `person`。仅有 generic vehicle tag 而没有 actor
metadata 时不猜测 car/truck/bus/bicycle/motorcycle 细类。

### 6.2 与正式 2D → 3D 链路一致的部分

对每个通过匹配并落入历史 RGB 视锥的 normal LiDAR 点，流程是：

1. 以 lidar 时刻的 local-to-world 变换把点转换到世界坐标；
2. 用历史 RGB 的 world-to-camera、CARLA 坐标到相机 optical 坐标转换和共用的
   pinhole projection 得到整数像素 `(u, v)`；
3. 对 SegFormer native-resolution 的 **project posterior** 做双线性采样；
4. 使用 `posterior_probabilities_to_logits` 转为与运行节点相同比例的 logits；
5. 以最大 posterior 得到 `pred_class`，与匹配 Semantic LiDAR GT 比较。

类别后验负责“是什么”；五个 reliability 只负责“这次语义观测应信多少”。
不支持的 GT 类别仍写入 `per_point.csv`，但从 primary accuracy、ECE、NLL、Brier
和融合 GT 统计中排除。`report.json` 会同时给出保留数、排除数和原因。

## 7. 五个正式 reliability factor

公式由 `reliability_factors.py` 统一实现，GA-BSVM 节点和离线评测调用同一套函数；
参数由所传 `--params-file` 的 `ga_bsvm_node.ros__parameters` 读取。下面是当前
`semantic_mapping_sim_livox.yaml` 的默认值，报告会写入实际使用值及其来源文件。

| 因子 | 运行时定义 | 当前默认参数 |
| --- | --- | --- |
| motion | 在 `t_l` 前后 `imu_window_sec` 内取 \(\|\omega\|\)、\(\|a\|\)，计算 \(\omega_{rms}\) 与 \(a_{dev,rms}=\sqrt{mean(\|a\|-g)^2}\)，\(r_m=clip(exp[-0.5((\omega_{rms}/s_\omega)^2+(a_{dev,rms}/s_a)^2)],r_{min},1)\) | window=0.15 s, \(s_\omega=2.0\), \(s_a=3.0\), \(g=9.81\), \(r_{min}=0.2\)；无对齐 IMU 时为 0.2 |
| density | 仅在本帧、成功投影的 normal LiDAR 点中，以 0.3 m 3D 邻域计数（包含自身）\(d\)，\(r_d=1-e^{-d/s_d}\) | `density_scale=8.0` |
| range | normal LiDAR sensor-local 三维量程 \(q=\|p\|_2\)，\(r_q=e^{-(q/s_q)^2}\) | `range_scale_m=20.0` |
| view | \(\rho=clip(\sqrt{((u-W/2)/(W/2))^2+((v-H/2)/(H/2))^2}/\sqrt2,0,1)\)，\(r_v=clip(1-\lambda_v\rho^2,0.05,1)\) | `view_edge_penalty=0.4` |
| semantic | posterior softmax 为 \(p\)，\(H=-\sum p\log p\)，\(r_s=f+(1-f)(1-H/\log K)\) | `semantic_confidence_floor=0.15`, \(K=13\) |

最终正式组合为：

\[
r_{combined}=r_{motion}r_{density}r_{range}r_{view}r_{semantic}.
\]

这不是已校准的“预测正确概率”。因此 factor summary 同时报告分箱 accuracy、
error rate、平均 confidence、平均 entropy、平均 reliability 和 accuracy–reliability
gap；而 ECE 仅对 SegFormer top-label posterior confidence 计算。NLL 和 Brier 是
支持 GT 点上的多类 posterior 指标；`posterior_temperature=1.0` 不等于完成概率校准，
基准不会自动改变 temperature。

## 8. 评测输出和字段含义

评测结果写入数据集下独立的时间戳结果目录，不覆盖原始采集文件。三个核心文件为：

### `per_point.csv`

每行是一个通过 LiDAR mutual-nearest 对齐、历史 RGB 匹配、几何投影和采样策略后的
点。完整字段顺序由 `semantic_mapping.carla.carla_reliability.PER_POINT_COLUMNS` 固定：

- 追溯和对齐：`schema_version`、`dataset_id`、`frame_index`、`lidar_frame`、
  `lidar_timestamp`、`rgb_frame`、`rgb_timestamp`、`requested_offset_ms`、
  `actual_offset_ms`、`timing_error_ms`、`point_index`、`semantic_match_index`、
  `match_distance_m`；
- 点与投影：`x_lidar/y_lidar/z_lidar`、`intensity`、`channel`、
  `x_world/y_world/z_world`、`u/v`；
- GT 与预测：`actor_id`、`carla_tag`、`gt_project_id`、`gt_class`、
  `gt_supported`、`gt_source`、`pred_project_id`、`pred_class`、`correct`、
  `pred_confidence`、`gt_probability`、`semantic_entropy`；
- 原始因子量：`range_m`、`local_density`、`view_radius`、`angular_rms`、
  `accel_deviation`、`imu_status`；
- 可靠度：`r_motion`、`r_density`、`r_range`、`r_view`、`r_semantic`、
  `r_combined`；
- 消融权重：`w_none`、`w_semantic`、`w_semantic_range`、
  `w_semantic_range_density`、`w_semantic_range_density_view`、`w_full`；
- posterior 索引：`posterior_file`、`posterior_row`。完整 float16 project posterior
  存在该 NPZ 文件的对应行，避免把 13 个浮点数重复塞进巨大的 CSV。

为控制规模，评测可采用确定性点采样/每帧最大点数；采样 seed、规则、保留数及丢弃数
写进 `report.json`，不能把采样后的样本量误报为全量点数。

### `factor_summary.csv`

每行是一种 `factor`、`axis`、`stratum` 和数值 bin。字段为
`factor, axis, stratum, bin_index, bin_left, bin_right, right_closed,
sample_count, unsupported_count, correct_count, accuracy, error_rate,
mean_raw_value, mean_reliability, mean_pred_confidence, mean_entropy, nll,
brier, reliability_gap`。

其中 range 至少使用 0–5、5–10、10–15、15–20、20–30、30–40、>40 m；可靠度使用
0.0–0.1 到 0.9–1.0。无样本 bin 的指标写空值而不是伪造为零。

### `report.json`

严格 JSON（不允许 NaN/Inf），包含：数据集/结果 schema、模型 ID 与 checkpoint
标签映射、project 支持类别、posterior temperature、Town、天气、seed、CARLA 版本、
传感器 blueprint 和 LiDAR 参数、同步和双 LiDAR 匹配统计、时间偏移接受/拒绝统计、
参数文件与五因素参数、总体及按类别 accuracy、ECE/NLL/Brier、各因子关联指标、
消融、可选 voxel 结果、样本/unsupported/失败关闭计数、warning 与 Git/代码版本。

## 9. 基础消融与可选短序列 VoxelMap

同一组点 posterior 不会因只改变 reliability weight 而改变单帧 argmax。因此
`w_none` 到 `w_full` 的单帧结果应解释为“权重是否把更可信点排在前面”，而不能声称
它提高了 SegFormer 的单帧分类准确率。消融序列是：

```text
none = 1
semantic = r_semantic
semantic_range = r_semantic * r_range
semantic_range_density = 上式 * r_density
semantic_range_density_view = 上式 * r_view
full = 上式 * r_motion = r_combined
```

传入 `--voxel-eval` 时，评测进一步将相同的 world 点、logits、timestamp 分别送入
现有 `VoxelMap`，比较 unweighted 与上述五种权重的短序列融合结果。GT voxel 标签由
支持类 Semantic LiDAR 点在相同 voxel size 下的多数类得到。它才是“reliability 是否
改善语义融合”的实际检验；需同时报告 voxel coverage、已覆盖 voxel accuracy、
uncertainty 和动态场景警告。若 ego 或目标运动，动态物体拖影本身会影响 voxel GT，
应优先以 stationary ego、`--no-moving-targets` 条件报告主结果，再把动态结果列为
压力测试。

## 10. 复现、验收与限制

### 复现实验最低要求

1. 固定 `Town × weather × seed × LiDAR configuration × ego motion profile`，
   保存完整 `manifest.json` 和命令行；
2. 先跑 offset=0 的静态基线，确认双 LiDAR match fraction、投影可视点数、各 GT
   类支持度和 unsupported 数；
3. 在同一个捕获数据集上只改变 `--time-offset-ms`；不能重新采集后把场景差异误作
   offset 效果；
4. 多 Town、至少 3 个 seed、晴天/非晴天均单独报告，然后再报告合并统计；
5. 不以一张截图或一个总 accuracy 得出“公式已最优”的结论。检查 reliability bin
   是否随真实 accuracy 呈预期关系，并报告失败案例和置信区间/样本数。

### 当前限制（必须在论文中如实说明）

- normal LiDAR 和 Semantic LiDAR 是两套独立 CARLA ray-cast；mutual-nearest +
  容差和 fail-closed 降低错误 GT 风险，但不保证物理上完全同射线。
- CARLA LiDAR measurement 是 frame 级输出；它不能完整复现真实旋转 LiDAR 每点的
  扫描内时序畸变。`lidar_tick`、rotation frequency 和 point counts 必须锁定并报告。
- CARLA IMU 的 accelerometer 是否在给定版本/传感器设置下以与正式 `imu_gravity`
  假设相同的重力语义输出，必须用静止实验核验。未核验前，motion 因子只应解释为
  仿真传感器条件下的量，不应直接外推至实机。
- `motion_angular_scale`、`motion_accel_scale`、density/range/view/entropy 参数是
  当前正式运行的经验配置；基准的作用是评估和为后续标定提供证据，**不应在本阶段
  调参后再宣称验证成功**。
- SegFormer 默认 checkpoint 是 Cityscapes 闭集模型，CARLA 的材质、光照和目标域会
  产生域偏移；unsupported 类不参与 primary metrics，不能以重新映射 GT 掩盖。
- ECE/NLL/Brier 只评价 posterior 的分类校准；可靠度不是概率，
  `reliability_gap` 只是诊断关联而非严格概率校准证明。
- CARLA 对真实机器人外参误差、图像畸变、网络延迟、地面反射、IMU 噪声和复杂动态
  遮挡的代表性有限。最终仍需 Gazebo 和实机独立验证。

版本锁定建议写入论文附录：CARLA server/client 版本、ROS 2 Humble、Python、
NumPy/SciPy、PyTorch/Transformers、SegFormer model revision、GPU/driver、当前 Git
commit、参数 YAML 的 SHA256，以及本文件所述完整命令。
