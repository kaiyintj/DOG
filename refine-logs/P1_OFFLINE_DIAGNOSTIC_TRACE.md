# P1-OFFLINE-DIAGNOSTIC rescue planning trace

- Time: 2026-08-17 13:54 +0800
- Role: ARIS experiment-bridge rescue planning
- Planning model: gpt-5.6-sol
- Action type: plan-only; no experiment/test/source/YAML mutation

## Direct evidence read

1. `refine-logs/EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md`.
2. `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/summary/initial_results_latest.md`.
3. Three dataset manifests:
   - stationary: 100 frames, Town10HD_Opt, ClearNoon, seed42, static targets;
   - constant velocity: 100 frames, same Town/weather/seed, static targets;
   - turning: 60 frames, same Town/weather/seed, static targets.

## Facts carried forward

- P0-A build/tests/entries/environment and P0-B four regressions are PASS.
- Town05 live image/reliability capture and Town10HD_Opt diagnostic image capture produced 0 valid frames; CARLA server Signal 11 is a structural block.
- No P0-C live result can be rescued by relabeling old data. P0-D/E remain NOT_RUN.
- P0-B already covers default range20/density8/view0/semantic-floor0.15/temperature1.0 point metrics at five offsets for all three profiles, plus turning/default voxel at five offsets.

## Rescue decision

- Stop live CARLA attempts; continue only a new, explicitly historical offline branch.
- Use stationary for OFAT to avoid motion confounding.
- Test two non-default point values for each of range, density, view, semantic floor, and temperature; do not repeat defaults.
- Reuse P0-B reports for five-offset V1/V2 rather than re-infer.
- Add only stationary/constant default voxel at offsets 0/150 and one representative voxel point per OFAT family.
- Cap at 17 evaluator runs, serial execution, <60 min; quality failures never stop independent runs.

## Claim boundary

This branch can measure sensitivity on one historical seed and compare motion profiles. It cannot establish new-capture validity, Town05 performance, physical density causality, multi-seed stability, parameter optimality, real-world readiness, or Motion V2 runtime eligibility.

## Output contracts

- Unique `P1-OD-*` run records and non-empty per-run output directories.
- Point comparison: accuracy, ECE/NLL/Brier, factor AUROC/AURC/gap, diagnostic weighted accuracy.
- Motion summary: actual offset/timing, relative pose, V1/V2, paired points, point metrics, turning voxel metrics.
- Voxel comparison: coverage, covered/all-GT accuracy, correct/wrong uncertainty, gap, error AUROC, deltas.
- Final report explicitly labels all conclusions `single-seed historical offline diagnostic`.
