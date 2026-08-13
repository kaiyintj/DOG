# CARLA Motion V2 实验记录（2026-08-13）

本目录记录基于已有 CARLA 数据集完成的 Motion V2 实现与验证。它是
实验分支的可审计结果快照，不代表 Motion V2 已冻结或已接入真实机器狗。

## 结论

Motion V2 使用 RGB 参考时刻与历史 RGB 时刻的相机相对位姿：

```text
相对转角 Δθ + 相对平移 Δd
                ↓
r_motion_v2 = exp(-0.5 * ((Δθ / 0.05)^2 + (Δd / 2.0)^2))
```

机制响应通过：静止时不随 offset 错误降权，直行和转弯时可靠性随
offset 增大而下降。IMU Motion V1 仅保留为诊断量 `r_motion_v1`。

但是，turning 体素消融没有通过主要验收条件。Motion V2 仍是整帧统一
标量，增加正确体素不确定性的幅度大于错误体素，导致
wrong-minus-correct uncertainty gap 下降。因此：

```text
Motion V2 保留为 CARLA 实验/诊断接口；
参数未冻结；
不接入 ga_bsvm_node.py；
不用于当前实机 reliability 乘法融合。
```

## 数据集与结果目录

| Profile | 数据集 | 本目录结果 |
|---|---|---|
| stationary | `reliability_baseline_fixed_20260813_155014` | `stationary_point/` |
| constant velocity | `motion_cv_constantmode_20260813_211718` | `constant_point/` |
| turning | `motion_turning_clean60_20260813_214753` | `turning_point/`、`turning_voxel/` |

每个结果目录提交：

- `report.json`：完整配置、软件、样本流、指标与告警；
- `factor_summary.csv`：分因素/分区间汇总。

## Motion V2 paired common-point 结果

| Profile | Common points | 0 ms | 20 ms | 50 ms | 100 ms | 150 ms |
|---|---:|---:|---:|---:|---:|---:|
| stationary | 24,459 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| constant velocity | 23,472 | 1.000000 | 0.998818 | 0.992648 | 0.970926 | 0.935734 |
| turning | 13,820 | 1.000000 | 0.998559 | 0.991050 | 0.964889 | 0.923452 |

turning 中，V1 在五个 offset 的严格 paired 点上固定为约 `0.929626`，
而 V2 随实际相对位姿变化，证明新旧定义和输出已经分离。

## Turning point-level diagnostic

比较 `semantic_range_density` 与 `full(+Motion V2)` 的 weighted accuracy：

| Offset | Accuracy | Motion V2 gain (percentage points) |
|---:|---:|---:|
| 0 ms | 0.996020 | +0.000000 |
| 20 ms | 0.995731 | +0.000024 |
| 50 ms | 0.995948 | +0.000165 |
| 100 ms | 0.994645 | +0.000893 |
| 150 ms | 0.994718 | +0.002289 |

这些增益很小，只作为排序诊断，不视为分类准确率改进证据。

## Turning voxel-level primary result

| Offset | SRD gap | Full gap | Gap delta | SRD AUROC | Full AUROC | AUROC delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 0.086784 | 0.086784 | 0.000000 | 0.767109 | 0.767109 | 0.000000 |
| 20 ms | 0.087594 | 0.087503 | -0.000092 | 0.769123 | 0.769123 | -0.0000005 |
| 50 ms | 0.086530 | 0.085973 | -0.000557 | 0.757865 | 0.757925 | +0.000060 |
| 100 ms | 0.088612 | 0.086426 | -0.002185 | 0.716753 | 0.717055 | +0.000302 |
| 150 ms | 0.080669 | 0.076189 | -0.004480 | 0.695192 | 0.696268 | +0.001076 |

150 ms 时：

```text
正确体素 mean uncertainty 增量 = +0.006805
错误体素 mean uncertainty 增量 = +0.002325
coverage delta                  = -0.000091
```

这没有满足“错误体素 uncertainty 增量不小于正确体素”的验收条件。

## 复现实验命令

Point-only：

```bash
source /opt/ros/humble/setup.bash
source /home/yk/ws/install/setup.bash
export HF_HOME=/home/yk/ws/.cache/huggingface

python3 /home/yk/ws/src/semantic_mapping/semantic_mapping/carla/carla_evaluate_reliability.py \
  --dataset DATASET \
  --params-file /home/yk/ws/src/semantic_mapping/config/semantic_mapping_sim_livox.yaml \
  --device cuda \
  --time-offset-ms 0 20 50 100 150 \
  --frame-tolerance-ms 6 \
  --max-points-per-frame 1000 \
  --posterior-storage none \
  --no-voxel-eval
```

Turning voxel：将最后一项替换为 `--voxel-eval`。

## 未提交的大型逐点产物

`per_point.csv` 总计约 295 MB，未提交到 Git。它们仍保存在本机原结果
目录，下面记录字节数与 SHA-256：

| Result | Bytes | SHA-256 |
|---|---:|---|
| turning point | 58,739,729 | `9e83239a2a0a71e6e29cf634be533315f04ae391ad69ed41aae99e7aa509b06d` |
| turning voxel | 58,739,728 | `35dc0b3af0922899ae4391917c49287c0a7fe8a4406d4ba3efd2d1905b5d6dd5` |
| constant point | 97,955,079 | `aea2db6dc158461e461545c6e82308e08d7c3d7bba4a1aa30fb34e15713d2841` |
| stationary point | 90,307,384 | `b15a7e3607b4c906a3128c6c7d573e41abc2f262642850ebc9a0041e60d127b8` |

## 验证

```text
pytest: 240 passed, 1 skipped
ament flake8: passed
ament pep257: passed
colcon build --packages-select semantic_mapping --symlink-install: passed
```

环境仍有已知的 SciPy 1.8.0 / NumPy 1.26.4 版本告警，本次没有修改依赖。

## 后续方向

若继续 Motion 设计，不应只调整 `rotation_scale`、`translation_scale`，也
不应直接把像素位移改为点权重。现有 turning 数据显示像素位移本身的
correctness AUROC 在 100/150 ms 仅约 `0.21/0.20`。下一步应优先研究：

- 遮挡和深度不连续边界；
- 跨时刻语义一致性；
- 动态目标与背景的差异化权重；
- 基于实际重投影残差而非纯运动幅值的风险模型。
