"""
Semantic mapping package.

Layout:

* ``semantic_mapping.runtime``: modules used by the real robot system
  (GA-BSVM, SegFormer/CLIP nodes, active perception, navigation bridge)
  and the shared core they depend on (VoxelMap, posterior, projection,
  reliability factors, ontology).
* ``semantic_mapping.carla``: CARLA-simulation-only capture/evaluation
  tooling. Nothing under ``runtime`` imports ``carla``.
"""
