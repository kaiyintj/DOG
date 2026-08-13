# SegFormer 电动自行车微调

默认 Cityscapes 权重可输出 `car`、`bicycle` 和 `motorcycle`，但没有
`electric_bicycle`。要真正区分这四类，需要使用带像素标注的电动
自行车数据微调 SegFormer。

## 1. 数据集

```bash
cd /home/yk/ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run semantic_mapping segformer_dataset \
  --dataset /home/yk/ws/segformer_ebike_dataset \
  --init
```

目录结构：

```text
segformer_ebike_dataset/
├── labels.json
├── train/images/
├── train/masks/
├── val/images/
├── val/masks/
├── calibration/images/    # 可选，只用于概率/阈值校准
├── calibration/masks/     # 可选
├── test/images/           # 正式指标必须提供的独立最终测试集
└── test/masks/            # 正式指标必须提供
```

图像和掩码必须同名。掩码是单通道 PNG，像素值如下：

| ID | 类别 |
| ---: | --- |
| 0 | road |
| 1 | building |
| 2 | tree |
| 3 | person |
| 4 | car |
| 5 | truck |
| 6 | bus |
| 7 | bicycle |
| 8 | electric_bicycle |
| 9 | motorcycle |
| 10 | chair |
| 11 | bench |
| 12 | unknown background |
| 255 | 忽略，不计算损失 |

每个实际提供的 split 都必须含有 `road`、`car`、`bicycle`、
`electric_bicycle` 和 `motorcycle` 标注像素。`train` 用于优化，`val` 只用于最佳模型
选择，`calibration` 只能用于将来的温度/阈值校准，`test` 只在选模完成后计算一次最终
验收指标。校验命令：

```bash
ros2 run semantic_mapping segformer_dataset \
  --dataset /home/yk/ws/segformer_ebike_dataset
```

CARLA 可以补充 road/car/bicycle/motorcycle，但自带蓝图没有真正的电动自行车。
不得把 Vespa 或普通摩托车真值直接改成 `electric_bicycle`。

## 2. 微调

```bash
export HF_HOME=/home/yk/ws/.cache/huggingface
export TORCH_HOME=/home/yk/ws/.cache/torch

ros2 run semantic_mapping segformer_finetune \
  --dataset /home/yk/ws/segformer_ebike_dataset \
  --output /home/yk/ws/models/segformer_b0_ebike \
  --base-model nvidia/segformer-b0-finetuned-cityscapes-1024-1024 \
  --device cuda \
  --fp16 \
  --epochs 20 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --image-size 512
```

脚本会将 Cityscapes 中可映射的分类器通道复制到新 13 类分类头。
`electric_bicycle` 通道以 bicycle 和 motorcycle 通道均值初始化，然后由
真实标注学习。

最佳权重和报告保存在：

```text
/home/yk/ws/models/segformer_b0_ebike/best/
/home/yk/ws/models/segformer_b0_ebike/training_report.json
```

默认要求最终验收 split 的 road IoU 不低于 `0.50`，
car/bicycle/electric_bicycle/motorcycle 每类 IoU 不低于 `0.30`，且
`electric_bicycle` IoU 不低于 `0.35`。存在 `test` 时只用 test 做最终门禁；没有 test
时为了兼容旧数据会计算 val 门禁，但报告写入
`evaluation_mode=legacy_val_compatibility` 和
`valid_for_formal_evaluation=false`，这些数值不能写成论文测试集结果。报告同时保存
checkpoint 逐文件和整体 SHA-256；任一门禁未通过时，命令保留权重和报告但以失败状态
退出。

## 3. 验收与运行

```bash
ros2 run semantic_mapping segformer_checkpoint \
  --model /home/yk/ws/models/segformer_b0_ebike/best \
  --local-files-only
```

必须看到：

```text
"missing_target_classes": []
"valid_for_requested_target_distinction": true
"checkpoint_hash_verified": true
"valid_for_formal_evaluation": true
```

最后一项只有独立 test、阈值通过且当前 checkpoint 哈希与报告一致时才为 `true`。
仅有正确 `id2label` 但缺少通过验收的 `training_report.json` 时，目标区分结果仍为
`false`。`--schema-only` 只用于调试权重结构，不代表可用于导航或正式评估。
运行时 `segformer_node` 会独立按 YAML 中的 `electric_bicycle_min_*_iou` 再检查
一次实际指标，不会只相信报告中的 `accepted` 布尔值。

使用微调权重启动：

```bash
ros2 run semantic_mapping segformer_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p model_id:=/home/yk/ws/models/segformer_b0_ebike/best

ros2 run semantic_mapping ga_bsvm_node --ros-args \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  -p semantic_backend:=segformer
```

不启动 CLIP。分别发布 `汽车`、`自行车`、`电动车` 和 `摩托车` 查询进行验收。
