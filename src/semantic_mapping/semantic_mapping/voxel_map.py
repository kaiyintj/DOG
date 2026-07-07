#!/usr/bin/env python3
import numpy as np

class VoxelMap:
    def __init__(self, voxel_size=0.1, K=6, max_odds=10.0):
        self.voxel_size = voxel_size
        self.K = K
        self.max_odds = max_odds
        # 核心数据结构：双轨体素字典
        self.voxels = {}

    def get_voxel_indices(self, point):
        """将3D点转换为体素的网格整数索引 (ix, iy, iz)"""
        return (
            int(np.floor(point[0] / self.voxel_size)),
            int(np.floor(point[1] / self.voxel_size)),
            int(np.floor(point[2] / self.voxel_size))
        )

    def softmax(self, logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def get_probabilities(self, key):
        """
        API 1: 从指定体素的 log_odds 中恢复完美的归一化概率分布
        """
        if key not in self.voxels:
            return np.ones(self.K) / self.K
        
        log_odds = self.voxels[key]['log_odds']
        # 贝叶斯对数几率还原：p = exp(L) / (1 + exp(L))
        exp_L = np.exp(log_odds)
        p = exp_L / (1.0 + exp_L)
        
        # 严格归一化
        p_sum = p.sum()
        if p_sum > 1e-6:
            p = p / p_sum
        else:
            p = np.ones(self.K) / self.K
        return p

    def get_confidence(self, key):
        """
        API 2: 基于信息论香农熵计算体素的认知置信度 (Confidence ∈ [0, 1])
        """
        if key not in self.voxels:
            return 0.0
            
        p = self.get_probabilities(key)
        p_safe = np.clip(p, 1e-10, 1.0)
        
        # 计算香农熵 H
        H = -np.sum(p_safe * np.log(p_safe))
        H_max = np.log(self.K) # 最大不确定性（均匀分布）时的熵
        
        # 置信度 = 1 - 归一化熵
        confidence = 1.0 - (H / H_max)
        return float(np.clip(confidence, 0.0, 1.0))

    def update(self, points, S_combined, logits, features=None):
        """
        核心建图算法：逐点 Log-Odds 贝叶斯更新。
        logits: (N, K) 每个点各自的语义 logits，或 (K,) 全局共享（向后兼容）。
        features: (N, 512) 每个点各自的 CLIP 特征，或 (512,) 全局共享，或 None。
        """
        logits = np.asarray(logits, dtype=np.float32)
        per_point_logits = logits.ndim == 2

        if not per_point_logits:
            p_obs = self.softmax(logits)
            p_clamped = np.clip(p_obs, 1e-6, 1.0 - 1e-6)
            shared_delta_L = np.log(p_clamped / (1.0 - p_clamped))

        if features is not None:
            features = np.asarray(features, dtype=np.float32)
        per_point_feats = features is not None and features.ndim == 2

        updated_count = 0
        new_count = 0

        for i, pt in enumerate(points):
            if S_combined[i] < 0.1:
                continue

            if per_point_logits:
                p_obs = self.softmax(logits[i])
                p_clamped = np.clip(p_obs, 1e-6, 1.0 - 1e-6)
                delta_L = np.log(p_clamped / (1.0 - p_clamped))
            else:
                delta_L = shared_delta_L

            ix, iy, iz = self.get_voxel_indices(pt)
            key = (ix, iy, iz)

            if key not in self.voxels:
                voxel_center = (np.array([ix, iy, iz], dtype=np.float32) + 0.5) * self.voxel_size
                self.voxels[key] = {
                    'log_odds': np.zeros(self.K, dtype=np.float32),
                    'feature_512': np.zeros(512, dtype=np.float32) if features is not None else None,
                    'weight_sum': 0.0,
                    'pos': voxel_center,
                }
                new_count += 1
            else:
                updated_count += 1

            voxel = self.voxels[key]

            voxel['log_odds'] += S_combined[i] * delta_L
            voxel['log_odds'] = np.clip(voxel['log_odds'], -self.max_odds, self.max_odds)

            if features is not None:
                feat_i = features[i] if per_point_feats else features
                w_old = voxel['weight_sum']
                w_new = S_combined[i]
                if w_old + w_new > 1e-6:
                    voxel['feature_512'] = (w_old * voxel['feature_512'] + w_new * feat_i) / (w_old + w_new)

            voxel['weight_sum'] += S_combined[i]

        return new_count, updated_count

    def get_visualization_clouds(self):
        """
        调用公共 API 生成供高级渲染的彩色语义点云与不确定性热力图
        """
        color_map = {
            0: [120, 120, 120],  # road
            1: [210, 210, 210],  # building
            2: [0, 160, 0],      # tree
            3: [255, 0, 0],      # person
            4: [0, 90, 255],     # car
            5: [50, 50, 50],     # unknown background
        }
        
        semantic_pts = []
        uncertainty_pts = []

        for key, voxel in self.voxels.items():
            pos = voxel['pos']
            
            # 直接调用刚写好的公共 API，彻底清除冗余代码！
            best_idx = np.argmax(voxel['log_odds'])
            sem_color = color_map.get(best_idx, [255, 255, 255])
            semantic_pts.append([pos[0], pos[1], pos[2], sem_color[0], sem_color[1], sem_color[2]])

            # 获取该体素的置信度
            confidence = self.get_confidence(key)
            
            # 认知不确定性颜色映射：置信度高 = 蓝色，置信度低(不确定) = 红色
            r = int((1.0 - confidence) * 255)
            b = int(confidence * 255)
            uncertainty_pts.append([pos[0], pos[1], pos[2], r, 0, b])

        return semantic_pts, uncertainty_pts
