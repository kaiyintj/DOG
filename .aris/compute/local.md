# ARIS local compute ledger

This ledger records the execution context actually used for the CARLA autonomous
test run rooted at `/home/yk/ws/carla_benchmark_data/autotest_20260817_132126`.

- Verified: 2026-08-17 13:23:26 +0800.
- Host backend: local Linux, ROS 2 Humble, Python 3.10.12, single GPU, W&B disabled.
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU; driver 580.173.02; NVIDIA-SMI CUDA 13.0; 8151 MiB total.
- PyTorch: `2.12.0.dev20260407+cu128`; `torch.cuda.is_available()` was `True` in the real local context.
- CARLA Python client: 0.9.16; server installation: `/home/yk/ws/third_party/CARLA_0.9.16`.
- SegFormer: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, fixed revision
  `21b3847fae21ddee674abd31129307b6a1235bd9`; snapshot is in the local Hugging Face cache.
- Package versions: NumPy 1.26.4, SciPy 1.8.0, Pillow 12.2.0, Transformers 4.46.3,
  huggingface-hub 0.36.0, tokenizers 0.20.3, PyYAML 5.4.1.
- Seeded CUDA witness: seed `20260817`, int64 affine/modulo kernel, sum `513226393`,
  SHA-256 `b46aece0945f5f79328101c3e6c58857beb1dcfad793eecae686a072ff7961e1`, exit 0.
- The restricted sandbox falsely reported no NVIDIA driver; its log is retained. The
  independent local `nvidia-smi` and CUDA witness logs both exit 0 and are the authoritative
  resource checks for this run.
- Workspace commit is `963aa693eabcece9f3b9c07aa4cc326ebc673000`; dirty files were preserved
  and recorded, not cleaned or overwritten. No source, formal YAML, commit, or push was made.

Evidence paths:

- [P0-A-001 local preflight](/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/preflight/P0-A-001-local.log)
- [nvidia-smi witness](/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/preflight/P0-A-001-nvidia-smi.log)
- [seeded CUDA witness](/home/yk/ws/carla_benchmark_data/autotest_20260817_132126/preflight/P0-A-001-cuda-witness.log)
