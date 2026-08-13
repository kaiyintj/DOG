# CARLA 二维图像识别基准

## 1. 目的与边界

这个基准只回答二维视觉问题：

- 图像里是否有 road、person、car、truck、bus、bicycle、motorcycle；
- CLIP 能否从目标裁剪图中区分四轮或两轮交通参与者；
- CLIP 能否区分 `blue truck`、`red car` 等颜色属性；
- CLIP 的文本查询是否把最高分网格落到目标所在网格；
- SegFormer 能否正确分割 road、person、vehicle，并在车辆实例内预测
  car/truck/bus/bicycle/motorcycle；
- SegFormer 类别加 RGB 颜色模块能否得到 `blue truck` 等组合结果。

这里不启动 FAST-LIO、GA-BSVM、Nav2，也不评价三维坐标或导航成功率。CARLA
负责生成 RGB、语义真值和实例真值；同一份离线数据分别送入 CLIP 和 SegFormer，
从而避免场景和时间差造成不公平比较。

SegFormer 是闭集像素分类器，本项目默认映射 13 类。CLIP 是开放词汇模型，但当前
自动评分覆盖 CARLA 能提供真值的 road、person、car、truck、bus、bicycle、
motorcycle 和车辆颜色。

## 2. 安装与显卡检查

建议使用 CARLA 0.9.16 的 Ubuntu 预编译包。官方文档建议独立显卡至少约 8 GB
显存，并需要端口 2000、2001。先检查：

```bash
nvidia-smi
python3 -m pip -V
```

`nvidia-smi` 必须能正常显示显卡和驱动。若显示无法连接 NVIDIA driver，先修复驱动，
否则 CARLA 图形服务器无法用于本测试。

从 CARLA 0.9.16 release 页面下载并解压 `CARLA_0.9.16.tar.gz`。安装与服务器完全
匹配的 Python API：

```bash
python3 -m pip install --user carla==0.9.16
python3 -c "import carla; print(carla.__file__)"
```

若 PyPI 包与服务器不匹配，则安装解压目录中 `PythonAPI/carla/dist/` 下与 Python
3.10 对应的 wheel。不要混用不同 CARLA 版本的 server 和 Python client。

构建本项目：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select semantic_mapping
source install/setup.bash
```

## 3. 采集一次共享数据集

终端 1 启动 CARLA。`CARLA_ROOT` 改成实际解压目录：

```bash
export CARLA_ROOT=$HOME/CARLA_0.9.16
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-port=2000
```

终端 2 采集同步 RGB、语义分割、实例分割。脚本会循环设置红、蓝、黄、绿、白、黑、
灰车辆，并尽量平衡生成 car/truck/bus/bicycle/motorcycle：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash

RUN_DIR="/home/yk/ws/carla_benchmark_data/town05_v2_$(date +%Y%m%d_%H%M%S)"

ros2 run semantic_mapping carla_capture_benchmark \
  --host 127.0.0.1 \
  --port 2000 \
  --map Town05 \
  --output "$RUN_DIR" \
  --cache-dir /home/yk/ws/.cache/carla \
  --frames 400 \
  --save-every 5 \
  --vehicles 40 \
  --vehicle-classes car truck bus bicycle motorcycle \
  --walkers 25 \
  --walker-min-distance 6 \
  --walker-max-distance 35 \
  --width 640 \
  --height 480 \
  --grid-rows 4 \
  --grid-cols 6 \
  --seed 42 \
  --stationary-ego \
  && printf '%s\n' "$RUN_DIR" \
    > /home/yk/ws/carla_benchmark_data/LATEST_DATASET.txt
```

输出目录包含：

```text
manifest.json
rgb/*.png
semantic/*.png
instance/*.png
metadata/*.json
```

`--cache-dir` 显式保存 CARLA 客户端下载的 OpenDrive、Traffic Manager 和导航地图
缓存，避免默认的相对路径 `carlaCache/` 随终端当前目录落到 `$HOME`。即使省略该参数，
采集器也会将缓存放到输出目录父目录的 `.carla_cache/`，不会写入当前工作目录。

`manifest.json` 中的 `vehicle_blueprint_classes` 会列出本 CARLA 安装实际可用的
五类交通工具蓝图。采集器按类别轮询生成，40 辆的目标是每类 8 辆；实际生成数和
可见观测数分别记录在 `spawned_vehicle_class_counts` 和
`visible_vehicle_observations_by_class`。如果某类列表为空或没有可见观测，
`valid_for_vehicle_benchmark` 会是 `false`，不能用该批数据评价该细类。

格式版本 2 使用 CARLA 官方实例编码 `actor_id = G + (B << 8)`，并直接用
`actor.id` 建立实例真值。旧格式数据评测时会自动交换字节供诊断，但旧元数据中已经
发生的遮挡串号无法完整恢复，不应用于正式论文结果。

采集结束后先检查数据支持度：

```bash
RUN_DIR="$(cat /home/yk/ws/carla_benchmark_data/LATEST_DATASET.txt)"

python3 -c "
import json
d = json.load(open('$RUN_DIR/manifest.json'))
print('格式:', d.get('format_version'), d.get('instance_id_encoding'))
print('地图:', d.get('map'))
print('生成车辆:', d.get('spawned_vehicle_class_counts'))
print('车辆观测:', d.get('visible_vehicle_observations_by_class'))
print('行人观测:', d.get('visible_people_observations'))
print('缺失类别:', d.get('missing_vehicle_classes'))
print('车辆基准有效:', d.get('valid_for_vehicle_benchmark'))
"
```

只有 `format_version=2`、请求的交通工具类别均有观测、
`missing_vehicle_classes=[]` 且 `valid_for_vehicle_benchmark=True`，才把这批数据
用于正式细类对比。
`LATEST_DATASET.txt` 只保存数据目录路径；不要把整个采集命令的标准输出重定向到它。

采集完成后关闭 CARLA 服务器再做模型推理，可以避免 CARLA 与模型争抢显存。

## 4. 同时测试 CLIP 与 SegFormer

模型缓存写到工作区，避免主目录缓存权限或空间问题：

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch
RUN_DIR="$(cat /home/yk/ws/carla_benchmark_data/LATEST_DATASET.txt)"

ros2 run semantic_mapping carla_evaluate_benchmark \
  --dataset "$RUN_DIR" \
  --backend both \
  --device cuda \
  --fp16 \
  --save-overlay-every 5
```

评测器默认读取 `manifest.json` 的 `grid_recommendation`，因此它与采集时记录的
网格一致。只有做明确的网格消融实验时才手动传 `--grid-rows/--grid-cols`；不一致
会写入评测警告。Gazebo/M2DGR 当前为 `4 x 6`，Lite3 预留配置为 `3 x 4`。

没有可用 CUDA 时可以改为 `--device cpu` 并去掉 `--fp16`，但 CLIP 与 SegFormer
都会明显变慢。也可以分别运行，方便测单模型速度：

```bash
ros2 run semantic_mapping carla_evaluate_benchmark \
  --dataset "$RUN_DIR" \
  --backend clip --device cuda

ros2 run semantic_mapping carla_evaluate_benchmark \
  --dataset "$RUN_DIR" \
  --backend segformer --device cuda --fp16
```

## 5. 看哪些结果

每次评测会创建带时间戳的 `results/` 子目录：

- `report.json`：汇总指标；
- `per_frame.csv`：逐帧推理耗时和可见目标数；
- `per_vehicle_instance.csv`：每一辆可见车辆的真值与预测；
- `detections/*.png`：同图显示 GT、CLIP、SegFormer 结果；
- `overlays/*.png`：SegFormer 彩色语义叠加图。

`report.json` 中重点看：

- `clip_class_accuracy`：目标框内交通工具细类分类；
- `clip_color_accuracy`：目标框内颜色分类；
- `clip_attribute_accuracy`：类别与颜色同时正确；
- `clip_object_localization_accuracy`：`truck` 查询最高分网格是否覆盖目标；
- `clip_attribute_localization_accuracy`：`blue truck` 查询最高分网格是否覆盖目标；
- `segformer_detection_rate`：车辆实例中至少 10% 像素被识别为车辆类；
- `segformer_class_accuracy`：检测成功后细分类是否正确；
- `segformer_attribute_accuracy`：SegFormer 类别和 RGB 颜色同时正确；
- `semantic_metrics.road/person/vehicle.iou`：像素级 IoU。

先看 `dataset_support` 和 `warnings`。`by_class`、`by_color` 只对 `count > 0`
的分组有意义；没有 truck 或 bus 样本时，不能把零值或总体均值当成该类性能。

`color_accuracy` 是项目 RGB/HSV 颜色模块在 CARLA 实例真值掩码内的独立结果，不是
SegFormer 自己预测颜色。SegFormer 不具备开放词汇颜色理解；组合属性来自
“SegFormer 类别 + RGB 颜色”。CLIP 的颜色结果则来自文本图像相似度。

## 6. 论文实验建议

至少使用 3 个 Town、3 个随机种子，并分别报告晴天、阴天、夜间或不同光照。每组保留
完全相同的 RGB 和真值数据，再比较 CLIP 与 SegFormer。不要只展示成功截图，应报告
类别、颜色、组合属性、定位准确率、IoU 和每帧耗时，并给出失败案例。

CARLA 只能证明仿真域二维识别性能，不能替代 Go2 Gazebo 三维融合实验，也不能替代
绝影 Lite3 实机外参、时延、网络和导航安全验证。

## 7. 常见问题

`No module named carla`：Python API 未安装，或 client/server 版本不一致。

`CUDA unavailable`：先检查 `nvidia-smi`；不能靠改 ROS 参数修复显卡驱动。

没有某个交通工具样本：查看 `manifest.json` 的
`vehicle_blueprint_classes`、`spawned_vehicle_class_counts` 和
`visible_vehicle_observations_by_class`。增加帧数或车辆数；蓝图列表为空时才需要
增加 CARLA assets。

SegFormer 的 person 为 0：先检查行人实例的 `visible_pixels`。目标太远、像素太少
时增加 `--walkers`，并保持 `--walker-max-distance 35` 或进一步缩短距离；若近距离
大目标仍为 0，才属于 Cityscapes 模型到 CARLA 的域偏移，应增加专用行人检测器或
做目标域微调，不应降低 GA-BSVM 阈值。

CLIP 颜色正确但定位错误：说明目标裁剪识别能力存在，但 `4 x 6` 网格太粗或背景干扰
过大。此时应增加实例检测/分割前端，而不是降低 GA-BSVM 阈值。

SegFormer 把车辆分成 building/unknown：这是模型域偏移或类别前端错误；检查
`overlays/` 和实例 CSV 中的 `segformer_vehicle_fraction`，不要在三维查询阶段伪造
成功。
