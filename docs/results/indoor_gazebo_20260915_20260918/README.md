# Small House 室内 Gazebo 历史结果

本目录保存可在 Windows 上直接阅读的小型结果摘要，来源是 Linux 工作区
`indoor_benchmark_runs/aws_small_house/` 的 2026-09-15、16、18 日批次。
原始 ROS Bag、进程日志和完整参数快照未上传；这些摘要不能代替原始数据复算。

| 日期 | 仓库内资料 | 结论 |
| --- | --- | --- |
| 9 月 15 日 | [性能基线](performance_20260915.md)、[参数快照](protocol/semantic_mapping_sim_indoor.yaml)、[Nav2 参数](protocol/nav2_sim_params.yaml)、[阶段标记](phases.tsv) | 建图/导航阶段平均 RTF 约 0.469/0.398；没有导航终态 |
| 9 月 16 日 | [chair 导航与停止](chair_navigation_20260916.md)、[机器汇总](analysis_summary.json) | 一次 `SUCCEEDED`，记录到停止后位移 |
| 9 月 18 日 | [cloud stride 3 复测](cloud_stride3_20260918.md) | 一次 `SUCCEEDED`，无性能改善的因果证据 |

两次成功运行路线和录制内容不同，不能作为性能对照；当前代码尚未完成新的 Gazebo 闭环复测。
状态与验收缺口见 [项目状态](../../PROJECT_STATUS.md)。
