# SegFormer-B0 smoke test

Date: 2026-07-11

Model: `nvidia/segformer-b0-finetuned-ade-512-512`

Input: one `640 x 480` frame extracted from the M2DGR `gate_03_ros2` bag.

## Result

The model downloaded and ran successfully on CPU. Three offline inference
passes took `1.460 s`, `0.659 s`, and `0.424 s`; the mean pixel confidence was
`0.847`. The six-class mapping covered the image as follows:

| Project class | Pixel fraction |
| --- | ---: |
| road | 34.18% |
| building | 3.66% |
| tree | 31.18% |
| person | 0.12% |
| car | 1.79% |
| unknown background | 29.07% |

The ROS 2 side-by-side node also processed the bag successfully. It published
a `640 x 480` `mono8` class mask with the original
`camera_color_optical_frame` header and timestamp. Continuous CPU inference
varied from approximately `0.6 s` to `1.8 s` per frame on the development PC.

## Interpretation

SegFormer produces substantially finer boundaries than the existing CLIP grid
classifier. The sampled frame separated road, vegetation, sky and structures,
and found a small person region. Some bicycle/motorcycle pixels were mapped to
the project's broad `car` navigation class, so the ADE20K mapping still needs
task-specific validation or fine-tuning.

This test establishes functional correctness only. The development PC had no
working NVIDIA driver, so these CPU measurements do not predict Lite3 Xavier
NX performance. Real deployment should benchmark an FP16 TensorRT engine on
the robot.
