#!/usr/bin/env python3
import numpy as np


class VoxelMap:
    """Reliability-weighted categorical semantic voxel map.

    Each voxel stores a Dirichlet distribution instead of independent binary
    log-odds. Observations from one sensor frame are aggregated before they are
    fused, preventing dense LiDAR returns from making one view overconfident.
    """

    def __init__(
        self,
        voxel_size=0.1,
        K=6,
        evidence_prior=1.0,
        evidence_strength=1.0,
        evidence_decay=0.995,
        max_frame_evidence=3.0,
        max_total_evidence=200.0,
        uncertainty_entropy_weight=0.7,
    ):
        self.voxel_size = float(voxel_size)
        self.K = int(K)
        self.evidence_prior = float(evidence_prior)
        self.evidence_strength = float(evidence_strength)
        self.evidence_decay = float(np.clip(evidence_decay, 0.0, 1.0))
        self.max_frame_evidence = float(max_frame_evidence)
        self.max_total_evidence = max(
            float(max_total_evidence), self.K * self.evidence_prior + 1.0)
        self.uncertainty_entropy_weight = float(
            np.clip(uncertainty_entropy_weight, 0.0, 1.0))
        self.voxels = {}

    def get_voxel_indices(self, point):
        """Convert a 3D point to integer voxel coordinates."""
        return tuple(np.floor(np.asarray(point) / self.voxel_size).astype(int))

    @staticmethod
    def softmax(logits):
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        denominator = np.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits / np.maximum(denominator, 1e-12)

    def get_probabilities(self, key):
        """Return the categorical posterior mean for a voxel."""
        voxel = self.voxels.get(key)
        if voxel is None:
            return np.ones(self.K, dtype=np.float64) / self.K
        alpha = voxel['alpha'].astype(np.float64)
        return alpha / np.maximum(np.sum(alpha), 1e-12)

    def get_uncertainty(self, key):
        """Return predictive entropy, epistemic uncertainty and a fused score."""
        voxel = self.voxels.get(key)
        if voxel is None:
            h_max = np.log(self.K)
            return h_max, 1.0, 1.0

        probabilities = self.get_probabilities(key)
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-10, 1.0)))
        h_max = max(np.log(self.K), 1e-12)
        entropy_ratio = float(np.clip(entropy / h_max, 0.0, 1.0))

        # For a Dirichlet distribution K / sum(alpha) is a useful ignorance
        # measure: it is one at the uniform unit prior and decreases with evidence.
        alpha_sum = float(np.sum(voxel['alpha']))
        epistemic = float(np.clip(self.K / max(alpha_sum, self.K), 0.0, 1.0))
        fused = (
            self.uncertainty_entropy_weight * entropy_ratio
            + (1.0 - self.uncertainty_entropy_weight) * epistemic
        )
        return float(entropy), epistemic, float(np.clip(fused, 0.0, 1.0))

    def get_confidence(self, key):
        """Return confidence in [0, 1] from semantic ambiguity and evidence."""
        _, _, uncertainty = self.get_uncertainty(key)
        return 1.0 - uncertainty

    def _new_voxel(self, key, feature_dim):
        center = (np.asarray(key, dtype=np.float32) + 0.5) * self.voxel_size
        feature = None
        if feature_dim is not None:
            feature = np.zeros(feature_dim, dtype=np.float32)
        return {
            'alpha': np.full(self.K, self.evidence_prior, dtype=np.float32),
            'feature_512': feature,
            'weight_sum': 0.0,
            'observation_count': 0,
            'pos': center,
        }

    def update(self, points, reliability, logits, features=None):
        """Fuse one frame of point-wise semantic observations.

        Args:
            points: ``(N, 3)`` points in the map frame.
            reliability: ``(N,)`` observation reliability in ``[0, 1]``.
            logits: ``(N, K)`` point logits, or one shared ``(K,)`` vector.
            features: optional point-wise or shared open-vocabulary features.
        """
        points = np.asarray(points, dtype=np.float32)
        reliability = np.asarray(reliability, dtype=np.float32).reshape(-1)
        logits = np.asarray(logits, dtype=np.float32)
        if len(points) != len(reliability):
            raise ValueError('points and reliability must have equal length')
        if logits.ndim == 2 and len(logits) != len(points):
            raise ValueError('point-wise logits must match points')

        probabilities = self.softmax(logits)
        point_wise_probabilities = probabilities.ndim == 2

        if features is not None:
            features = np.asarray(features, dtype=np.float32)
            if features.ndim == 2 and len(features) != len(points):
                raise ValueError('point-wise features must match points')
        point_wise_features = features is not None and features.ndim == 2
        feature_dim = None if features is None else int(features.shape[-1])

        # Aggregate correlated returns from this frame by voxel. A frame gets a
        # bounded amount of evidence no matter how many LiDAR points hit it.
        frame_updates = {}
        for index, point in enumerate(points):
            weight = float(np.clip(reliability[index], 0.0, 1.0))
            if weight < 1e-3:
                continue

            key = self.get_voxel_indices(point)
            update = frame_updates.setdefault(
                key,
                {
                    'weight': 0.0,
                    'probability_sum': np.zeros(self.K, dtype=np.float64),
                    'feature_sum': (
                        np.zeros(feature_dim, dtype=np.float64)
                        if feature_dim is not None else None
                    ),
                },
            )
            probability = probabilities[index] if point_wise_probabilities else probabilities
            update['weight'] += weight
            update['probability_sum'] += weight * probability
            if features is not None:
                feature = features[index] if point_wise_features else features
                update['feature_sum'] += weight * feature

        new_count = 0
        updated_count = 0
        for key, update in frame_updates.items():
            raw_weight = update['weight']
            if raw_weight <= 1e-9:
                continue
            frame_evidence = min(raw_weight, self.max_frame_evidence)
            mean_probability = update['probability_sum'] / raw_weight

            if key not in self.voxels:
                self.voxels[key] = self._new_voxel(key, feature_dim)
                new_count += 1
            else:
                updated_count += 1
            voxel = self.voxels[key]

            prior = self.evidence_prior
            voxel['alpha'] = prior + self.evidence_decay * (voxel['alpha'] - prior)
            voxel['alpha'] += (
                self.evidence_strength * frame_evidence * mean_probability
            ).astype(np.float32)

            alpha_sum = float(np.sum(voxel['alpha']))
            if alpha_sum > self.max_total_evidence:
                evidence = np.maximum(voxel['alpha'] - prior, 0.0)
                evidence_budget = self.max_total_evidence - self.K * prior
                evidence_sum = float(np.sum(evidence))
                if evidence_sum > 1e-9:
                    voxel['alpha'] = (
                        prior + evidence * (evidence_budget / evidence_sum)
                    ).astype(np.float32)

            if features is not None:
                mean_feature = update['feature_sum'] / raw_weight
                feature_norm = np.linalg.norm(mean_feature)
                if feature_norm > 1e-9:
                    mean_feature = mean_feature / feature_norm
                if voxel['feature_512'] is None:
                    voxel['feature_512'] = mean_feature.astype(np.float32)
                else:
                    old_weight = voxel['weight_sum']
                    denominator = old_weight + frame_evidence
                    voxel['feature_512'] = (
                        (old_weight * voxel['feature_512'] + frame_evidence * mean_feature)
                        / max(denominator, 1e-9)
                    ).astype(np.float32)
                    fused_norm = np.linalg.norm(voxel['feature_512'])
                    if fused_norm > 1e-9:
                        voxel['feature_512'] /= fused_norm

            voxel['weight_sum'] += frame_evidence
            voxel['observation_count'] += 1

        return new_count, updated_count

    def get_visualization_clouds(self):
        """Build semantic and uncertainty-colored point lists for RViz."""
        color_map = {
            0: [120, 120, 120],  # road
            1: [210, 210, 210],  # building
            2: [0, 160, 0],      # tree
            3: [255, 0, 0],      # person
            4: [0, 90, 255],     # car
            5: [50, 50, 50],     # unknown background
        }

        semantic_points = []
        uncertainty_points = []
        for key, voxel in self.voxels.items():
            position = voxel['pos']
            class_index = int(np.argmax(self.get_probabilities(key)))
            color = color_map.get(class_index, [255, 255, 255])
            semantic_points.append([*position, *color])

            _, _, uncertainty = self.get_uncertainty(key)
            red = int(uncertainty * 255)
            blue = int((1.0 - uncertainty) * 255)
            uncertainty_points.append([*position, red, 0, blue])

        return semantic_points, uncertainty_points
