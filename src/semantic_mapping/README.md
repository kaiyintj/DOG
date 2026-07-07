# semantic_mapping

ROS 2 nodes for CLIP-based semantic mapping, GA-BSVM voxel fusion, semantic costmap publishing, and active perception speed modulation.

## Python runtime dependencies

Install these Python packages in the ROS environment used to run the nodes:

```bash
pip install numpy scipy torch open_clip_torch pillow
```

The ROS package dependencies are declared in `package.xml`. The three runtime parameter presets are:

- `config/semantic_mapping_m2dgr.yaml`
- `config/semantic_mapping_sim_livox.yaml`
- `config/semantic_mapping_lite3_real.yaml`

