#!/usr/bin/env python3
"""Time-aware Dirichlet semantic voxel map."""

import time
from bisect import bisect_left

import numpy as np

from semantic_mapping.runtime.semantic_schema import DEFAULT_CLASS_COLORS


class VoxelMap:
    """
    Reliability-weighted categorical semantic voxel map.

    Semantic evidence follows a continuous-time exponential decay model.
    ``evidence_decay`` remains backward compatible: it is the retained fraction
    over one ``evidence_decay_reference_sec`` interval.  Explicit observation
    timestamps age history by elapsed time. Evidence injection normalizes only
    intervals up to one reference period; a visibility gap never supplies more
    than the current observation's budget. Callers that omit timestamps retain
    the historical one-step decay behavior.

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
        evidence_decay_reference_sec=0.1,
        max_frame_evidence=3.0,
        max_total_evidence=200.0,
        max_observation_weight=None,
        uncertainty_entropy_weight=0.7,
        class_colors=None,
        source_history_size=32,
    ):
        self.voxel_size = float(voxel_size)
        self.K = int(K)
        self.evidence_prior = float(evidence_prior)
        self.evidence_strength = float(evidence_strength)
        evidence_decay = float(evidence_decay)
        if not np.isfinite(evidence_decay):
            raise ValueError('evidence_decay must be finite')
        self.evidence_decay = float(np.clip(evidence_decay, 0.0, 1.0))
        self.evidence_decay_reference_sec = float(
            evidence_decay_reference_sec)
        if (
            not np.isfinite(self.evidence_decay_reference_sec)
            or self.evidence_decay_reference_sec <= 0.0
        ):
            raise ValueError(
                'evidence_decay_reference_sec must be finite and positive')
        self.max_frame_evidence = float(max_frame_evidence)
        self.source_history_size = int(source_history_size)
        if self.source_history_size <= 0:
            raise ValueError('source_history_size must be positive')
        self.max_total_evidence = max(
            float(max_total_evidence), self.K * self.evidence_prior + 1.0)
        if max_observation_weight is None:
            max_observation_weight = (
                self.max_total_evidence - self.K * self.evidence_prior)
        self.max_observation_weight = float(max_observation_weight)
        if (
            not np.isfinite(self.max_observation_weight)
            or self.max_observation_weight <= 0.0
        ):
            raise ValueError(
                'max_observation_weight must be finite and positive')
        self.uncertainty_entropy_weight = float(
            np.clip(uncertainty_entropy_weight, 0.0, 1.0))
        configured_colors = (
            DEFAULT_CLASS_COLORS if class_colors is None else class_colors)
        configured_colors = np.asarray(configured_colors, dtype=np.uint8)
        if configured_colors.ndim == 1 and configured_colors.size % 3 == 0:
            configured_colors = configured_colors.reshape(-1, 3)
        if configured_colors.ndim != 2 or configured_colors.shape[1] != 3:
            raise ValueError('class_colors must have shape (K, 3)')
        if len(configured_colors) < self.K:
            raise ValueError(
                f'class_colors has {len(configured_colors)} entries, expected at least {self.K}')
        self.class_colors = configured_colors[:self.K]
        self.voxels = {}
        # Monotonic map revision used by query retries.  It changes only after
        # an update or pruning pass has had a chance to mutate map state.
        self.revision = 0
        # Positive injections from the most recent update, before total-budget
        # rescaling. Used by confirmation without treating net growth as new data.
        self.last_effective_updates = {}

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

    def get_evidence_probabilities(self, key):
        """
        Return the class distribution of observations, excluding the prior.

        The posterior mean intentionally includes the symmetric Dirichlet prior
        and is therefore appropriate for uncertainty estimation.  Querying a
        rare object needs a different view: with many classes, that prior can
        push even one strong observation below a fixed probability threshold.
        Removing it here preserves the observed class distribution while the
        caller separately enforces a minimum amount of accumulated evidence.
        """
        voxel = self.voxels.get(key)
        if voxel is None:
            return np.ones(self.K, dtype=np.float64) / self.K
        evidence = np.maximum(
            voxel['alpha'].astype(np.float64) - self.evidence_prior,
            0.0,
        )
        evidence_sum = float(np.sum(evidence))
        if evidence_sum <= 1e-12:
            return np.ones(self.K, dtype=np.float64) / self.K
        return evidence / evidence_sum

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

    @staticmethod
    def _resolve_timestamp(timestamp_sec):
        """Return a finite observation timestamp without changing old callers."""
        if timestamp_sec is not None:
            try:
                timestamp_sec = float(timestamp_sec)
            except (TypeError, ValueError):
                timestamp_sec = None
        if timestamp_sec is None or not np.isfinite(timestamp_sec):
            timestamp_sec = time.monotonic()
        return float(timestamp_sec)

    @staticmethod
    def _has_explicit_timestamp(timestamp_sec):
        """Return whether a caller supplied a usable clock timestamp."""
        try:
            timestamp_sec = float(timestamp_sec)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(timestamp_sec))

    def _decay_factor(self, elapsed_sec):
        """Return continuous evidence retention over ``elapsed_sec``."""
        elapsed_sec = max(float(elapsed_sec), 0.0)
        if elapsed_sec <= 0.0 or self.evidence_decay >= 1.0:
            return 1.0
        if self.evidence_decay <= 0.0:
            return 0.0
        reference_steps = elapsed_sec / self.evidence_decay_reference_sec
        return float(np.exp(np.log(self.evidence_decay) * reference_steps))

    def _evidence_interval_scale(self, elapsed_sec):
        """
        Convert a sampled observation into reference-period evidence.

        Short continuous intervals use the exact exponential recurrence.
        At or beyond the reference interval the scale is one: missing views
        cannot be extrapolated into observed support. Frame-rate equivalence
        applies only to continuous intervals within that reference period.
        """
        elapsed_sec = max(float(elapsed_sec), 0.0)
        if elapsed_sec <= 0.0:
            return 0.0
        if elapsed_sec >= self.evidence_decay_reference_sec:
            return 1.0
        reference_steps = elapsed_sec / self.evidence_decay_reference_sec
        if self.evidence_decay >= 1.0:
            return reference_steps
        if self.evidence_decay <= 0.0:
            return 1.0
        exponent = np.log(self.evidence_decay) * reference_steps
        return float(-np.expm1(exponent) / (1.0 - self.evidence_decay))

    def reset_clock(self, timestamp_sec):
        """
        Re-anchor an explicitly confirmed clock restart, retaining the map.

        The ROS adapter distinguishes a restarted /clock from a delayed frame.
        Source-image stamps from the old clock must not reject new images with
        reused timestamps. Ordinary pruning never clears source admission.
        """
        timestamp_sec = self._resolve_timestamp(timestamp_sec)
        self.last_effective_updates = {}
        for voxel in self.voxels.values():
            voxel['last_observed_at_sec'] = timestamp_sec
            voxel['last_decay_at_sec'] = timestamp_sec
            voxel['last_effective_observed_at_sec'] = None
            voxel.pop('source_history', None)
        if self.voxels:
            self.revision += 1

    @staticmethod
    def _stored_timestamp(voxel, field):
        """Read one finite timestamp from a voxel, or return ``None``."""
        try:
            timestamp_sec = float(voxel.get(field))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(timestamp_sec):
            return None
        return timestamp_sec

    def _decay_voxel_to(self, voxel, timestamp_sec, *, legacy_step=False):
        """Decay semantic and auxiliary evidence to one clock timestamp."""
        previous_sec = self._stored_timestamp(voxel, 'last_decay_at_sec')
        if previous_sec is None:
            previous_sec = self._stored_timestamp(
                voxel, 'last_observed_at_sec')
        if legacy_step:
            factor = self.evidence_decay
        elif previous_sec is None:
            # A legacy voxel without any clock metadata cannot be aged safely.
            return 1.0
        elif timestamp_sec < previous_sec:
            # ROS /clock may rewind when a simulation restarts.  Re-anchor the
            # decay clock without inventing a large or negative time interval.
            voxel['last_decay_at_sec'] = float(timestamp_sec)
            return 1.0
        else:
            factor = self._decay_factor(timestamp_sec - previous_sec)

        prior = self.evidence_prior
        alpha = np.asarray(voxel['alpha'], dtype=np.float64)
        voxel['alpha'] = (
            prior + factor * np.maximum(alpha - prior, 0.0)
        ).astype(np.float32)
        fallback_weight = voxel.get('weight_sum', 0.0)
        fields = {
            'weight_sum': fallback_weight,
            'feature_weight_sum': (
                fallback_weight
                if voxel.get('feature_512') is not None
                else 0.0
            ),
            'color_weight_sum': voxel.get('color_weight_sum', 0.0),
        }
        for field, fallback in fields.items():
            try:
                weight = float(voxel.get(field, fallback))
            except (TypeError, ValueError):
                weight = 0.0
            if not np.isfinite(weight):
                weight = 0.0
            voxel[field] = float(np.clip(
                factor * max(weight, 0.0),
                0.0,
                self.max_observation_weight,
            ))
        voxel['last_decay_at_sec'] = float(timestamp_sec)
        return factor

    def _new_voxel(self, key, feature_dim, timestamp_sec=None):
        center = (np.asarray(key, dtype=np.float32) + 0.5) * self.voxel_size
        feature = None
        if feature_dim is not None:
            feature = np.zeros(feature_dim, dtype=np.float32)
        observation_time = self._resolve_timestamp(timestamp_sec)
        return {
            'alpha': np.full(self.K, self.evidence_prior, dtype=np.float32),
            'feature_512': feature,
            'feature_weight_sum': 0.0,
            'color_rgb': None,
            'color_weight_sum': 0.0,
            'weight_sum': 0.0,
            'observation_count': 0,
            'pos': center,
            'created_at_sec': observation_time,
            'last_observed_at_sec': observation_time,
            'last_effective_observed_at_sec': None,
            'last_decay_at_sec': observation_time,
        }

    def update(
        self,
        points,
        reliability,
        logits,
        features=None,
        colors=None,
        timestamp_sec=None,
        source_id=None,
    ):
        """
        Fuse one frame of point-wise semantic observations.

        ``points`` has shape ``(N, 3)``, ``reliability`` has shape ``(N,)``,
        and ``logits`` is either point-wise ``(N, K)`` or one shared ``(K,)``
        vector.  ``features`` may likewise be point-wise or shared, while
        ``colors`` must have shape ``(N, 3)``.  When ``timestamp_sec`` is a
        finite caller-clock timestamp, history ages continuously, while new
        support covers at most one reference period. Omitting it applies one
        legacy reference-period update. ``source_id`` is an optional pair of
        (camera identity, integer image timestamp in nanoseconds). Each voxel
        accepts each source once within a bounded recent history; new coverage
        remains valid. Older sources beyond the retained window are rejected.
        The source clock sets short-interval normalization, independently of
        the point-cloud timestamp used for aging. Timestamp-free callers with
        no source identity treat every call as a new observation.
        Source-aware callers confirm clock restarts with ``reset_clock``;
        older observations alone never rewind their history. The returned
        counts include only new/existing voxels receiving positive contribution.
        """
        self.last_effective_updates = {}
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

        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.shape != (len(points), 3):
                raise ValueError('point-wise colors must have shape (N, 3)')
            if colors.size and float(np.nanmax(colors)) > 1.0:
                colors = colors / 255.0
            colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
            colors = np.clip(colors, 0.0, 1.0)

        # Aggregate correlated returns from this frame by voxel. A frame gets a
        # bounded amount of evidence no matter how many LiDAR points hit it.
        frame_updates = {}
        for index, point in enumerate(points):
            weight = float(np.clip(reliability[index], 0.0, 1.0))
            if not np.isfinite(weight):
                continue

            key = self.get_voxel_indices(point)
            update = frame_updates.setdefault(
                key,
                {
                    'weight': 0.0,
                    'point_count': 0,
                    'probability_sum': np.zeros(self.K, dtype=np.float64),
                    'feature_sum': (
                        np.zeros(feature_dim, dtype=np.float64)
                        if feature_dim is not None else None
                    ),
                    'color_sum': (
                        np.zeros(3, dtype=np.float64)
                        if colors is not None else None
                    ),
                },
            )
            probability = probabilities[index] if point_wise_probabilities else probabilities
            update['weight'] += weight
            update['point_count'] += 1
            update['probability_sum'] += weight * probability
            if features is not None:
                feature = features[index] if point_wise_features else features
                update['feature_sum'] += weight * feature
            if colors is not None:
                update['color_sum'] += weight * colors[index]

        explicit_timestamp = self._has_explicit_timestamp(timestamp_sec)
        observation_time = self._resolve_timestamp(timestamp_sec)
        new_count = 0
        updated_count = 0
        touched_count = 0
        for key, update in frame_updates.items():
            raw_weight = update['weight']
            if raw_weight <= 1e-9:
                continue
            # Cap geometric support before applying quality. Dense low-quality
            # returns cannot consume the same budget as high-quality returns.
            mean_quality = raw_weight / update['point_count']
            frame_evidence = (
                min(update['point_count'], self.max_frame_evidence)
                * mean_quality)
            mean_probability = update['probability_sum'] / raw_weight

            is_new_voxel = key not in self.voxels
            source_history = None
            source_interval_scale = 1.0
            source_index = 0
            if source_id is not None:
                camera_id, source_ns = source_id
            if source_id is not None and not is_new_voxel:
                source_history = self.voxels[key].get(
                    'source_history', {}).get(camera_id)
                if source_history is not None:
                    stamps = source_history['stamps']
                    source_index = bisect_left(stamps, source_ns)
                    retired_ns = source_history['retired_through_ns']
                    if (
                        (source_index < len(stamps) and stamps[source_index] == source_ns)
                        or (retired_ns is not None and source_ns <= retired_ns)
                    ):
                        # Do not refresh TTL, auxiliary values or revision for
                        # copied/retired sources. Pruning still ages history.
                        continue
                    # A unique out-of-order source is still a real observation.
                    # Use its nearest retained source for bounded sampling
                    # support; source reordering never resets the decay clock.
                    neighbor_gaps = []
                    if source_index:
                        neighbor_gaps.append(source_ns - stamps[source_index - 1])
                    if source_index < len(stamps):
                        neighbor_gaps.append(stamps[source_index] - source_ns)
                    source_interval_scale = self._evidence_interval_scale(
                        min(neighbor_gaps) / 1e9)
            if is_new_voxel:
                self.voxels[key] = self._new_voxel(
                    key,
                    feature_dim,
                    timestamp_sec=observation_time,
                )
            voxel = self.voxels[key]
            touched_count += 1
            # Backfill metadata for maps created before timestamps were added.
            voxel.setdefault('created_at_sec', observation_time)
            previous_observed_sec = self._stored_timestamp(
                voxel, 'last_observed_at_sec')
            decay_time = observation_time
            if is_new_voxel:
                interval_scale = 1.0
            elif not explicit_timestamp:
                interval_scale = 1.0
                self._decay_voxel_to(
                    voxel,
                    observation_time,
                    legacy_step=True,
                )
            else:
                clock_rewound = (
                    source_id is None
                    and previous_observed_sec is not None
                    and observation_time < previous_observed_sec)
                if clock_rewound:
                    voxel['last_effective_observed_at_sec'] = None
                if not clock_rewound:
                    # Pruning can already have aged this voxel past a delayed
                    # observation. Keep that horizon and age only the new
                    # contribution to it; do not decay old evidence twice.
                    last_decay = self._stored_timestamp(voxel, 'last_decay_at_sec')
                    if last_decay is not None:
                        decay_time = max(observation_time, last_decay)
                self._decay_voxel_to(voxel, decay_time)
                if source_id is not None:
                    interval_scale = source_interval_scale
                elif previous_observed_sec is None or clock_rewound:
                    interval_scale = 1.0
                else:
                    interval_scale = self._evidence_interval_scale(
                        observation_time - previous_observed_sec)
                interval_scale *= self._decay_factor(decay_time - observation_time)
            voxel['last_observed_at_sec'] = observation_time
            voxel['last_decay_at_sec'] = decay_time

            prior = self.evidence_prior
            observation_decay = self._decay_factor(decay_time - observation_time)
            weight_limit = self.max_observation_weight * observation_decay
            effective_evidence = frame_evidence * interval_scale
            if source_id is not None:
                if source_history is None:
                    source_history = {'stamps': [], 'retired_through_ns': None}
                    voxel.setdefault('source_history', {})[camera_id] = source_history
                stamps = source_history['stamps']
                stamps.insert(source_index, source_ns)
                if len(stamps) > self.source_history_size:
                    source_history['retired_through_ns'] = stamps.pop(0)
            if effective_evidence <= 0.0:
                # A repeated timestamp cannot initialize an unweighted feature
                # or color that disagrees with the unchanged semantic evidence.
                continue
            previous_effective = self._stored_timestamp(voxel, 'last_effective_observed_at_sec')
            voxel['last_effective_observed_at_sec'] = (
                observation_time if previous_effective is None
                else max(previous_effective, observation_time))
            self.last_effective_updates[key] = (float(effective_evidence), mean_probability.copy())
            if is_new_voxel:
                new_count += 1
            else:
                updated_count += 1
            voxel['alpha'] += (
                self.evidence_strength * effective_evidence * mean_probability
            ).astype(np.float32)

            alpha_sum = float(np.sum(voxel['alpha']))
            evidence_budget = (self.max_total_evidence - self.K * prior) * observation_decay
            if alpha_sum > self.K * prior + evidence_budget:
                evidence = np.maximum(voxel['alpha'] - prior, 0.0)
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
                    old_weight = 0.0
                else:
                    old_weight = float(voxel.get(
                        'feature_weight_sum',
                        min(
                            float(voxel.get('weight_sum', 0.0)),
                            self.max_observation_weight,
                        ),
                    ))
                    denominator = old_weight + effective_evidence
                    voxel['feature_512'] = (
                        (
                            old_weight * voxel['feature_512']
                            + effective_evidence * mean_feature
                        )
                        / max(denominator, 1e-9)
                    ).astype(np.float32)
                    fused_norm = np.linalg.norm(voxel['feature_512'])
                    if fused_norm > 1e-9:
                        voxel['feature_512'] /= fused_norm
                voxel['feature_weight_sum'] = min(
                    old_weight + effective_evidence,
                    weight_limit,
                )

            if colors is not None:
                mean_color = update['color_sum'] / raw_weight
                old_color_weight = float(voxel.get('color_weight_sum', 0.0))
                if voxel.get('color_rgb') is None:
                    voxel['color_rgb'] = mean_color.astype(np.float32)
                else:
                    voxel['color_rgb'] = (
                        (
                            old_color_weight * voxel['color_rgb']
                            + effective_evidence * mean_color
                        )
                        / max(old_color_weight + effective_evidence, 1e-9)
                    ).astype(np.float32)
                voxel['color_weight_sum'] = min(
                    old_color_weight + effective_evidence,
                    weight_limit,
                )

            voxel['weight_sum'] = min(
                float(voxel.get('weight_sum', 0.0)) + effective_evidence,
                weight_limit,
            )
            voxel['observation_count'] += 1

        if touched_count:
            self.revision += 1
        return new_count, updated_count

    def prune(
        self,
        now_sec,
        *,
        dynamic_class_ids=(),
        dynamic_ttl_sec=0.0,
        stale_ttl_sec=0.0,
        center=None,
        max_distance_m=0.0,
        max_count=0,
    ):
        """
        Age evidence and remove expired or out-of-scope voxels.

        All time values must use the same clock domain.  Non-positive TTL,
        distance, and count limits disable their respective rule.  Distance is
        measured horizontally from ``center``.  Legacy voxels without timestamp
        metadata are not deleted by TTL rules, but remain eligible for distance
        and count limits.  Keys are collected before deletion so callers never
        observe mutation during dictionary iteration.
        """
        try:
            now_sec = float(now_sec)
        except (TypeError, ValueError):
            raise ValueError('now_sec must be a finite number') from None
        if not np.isfinite(now_sec):
            raise ValueError('now_sec must be a finite number')

        dynamic_ids = set()
        for class_id in dynamic_class_ids or ():
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                continue
            if 0 <= class_id < self.K:
                dynamic_ids.add(class_id)

        def positive_float(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value) or value <= 0.0:
                return 0.0
            return value

        dynamic_ttl = positive_float(dynamic_ttl_sec)
        stale_ttl = positive_float(stale_ttl_sec)
        max_distance = positive_float(max_distance_m)
        try:
            count_limit = max(0, int(max_count))
        except (TypeError, ValueError):
            count_limit = 0

        center_xy = None
        if center is not None and max_distance > 0.0:
            candidate_center = np.asarray(center, dtype=np.float64).reshape(-1)
            if (
                candidate_center.size >= 2
                and np.all(np.isfinite(candidate_center[:2]))
            ):
                center_xy = candidate_center[:2]

        removals = {}
        snapshot = list(self.voxels.items())
        for key, voxel in snapshot:
            class_index = int(np.argmax(self.get_probabilities(key)))
            # /clock can lag a just-fused sensor stamp without restarting.
            # Explicit reset_clock handles restarts; pruning never rewinds.
            last_decay = self._stored_timestamp(voxel, 'last_decay_at_sec')
            decay_time = now_sec if last_decay is None else max(now_sec, last_decay)
            self._decay_voxel_to(voxel, decay_time)
            observed_at = voxel.get('last_observed_at_sec')
            try:
                observed_at = float(observed_at)
            except (TypeError, ValueError):
                observed_at = None
            if observed_at is not None and not np.isfinite(observed_at):
                observed_at = None
            age = (
                max(0.0, now_sec - observed_at)
                if observed_at is not None
                else None
            )

            if age is not None and dynamic_ttl > 0.0 and dynamic_ids:
                if class_index in dynamic_ids and age > dynamic_ttl:
                    removals[key] = 'dynamic_ttl'
                    continue
            if age is not None and stale_ttl > 0.0 and age > stale_ttl:
                removals[key] = 'stale_ttl'
                continue
            if center_xy is not None:
                position = np.asarray(voxel.get('pos', ()), dtype=np.float64)
                if (
                    position.size >= 2
                    and np.all(np.isfinite(position[:2]))
                    and np.linalg.norm(position[:2] - center_xy) > max_distance
                ):
                    removals[key] = 'distance'

        remaining = [
            (key, voxel)
            for key, voxel in snapshot
            if key not in removals
        ]
        overflow = len(remaining) - count_limit if count_limit > 0 else 0
        if overflow > 0:
            def eviction_rank(item):
                key, voxel = item
                observed_at = voxel.get('last_observed_at_sec')
                try:
                    observed_at = float(observed_at)
                except (TypeError, ValueError):
                    observed_at = now_sec
                if not np.isfinite(observed_at):
                    observed_at = now_sec
                distance = 0.0
                if center_xy is not None:
                    position = np.asarray(voxel.get('pos', ()), dtype=np.float64)
                    if position.size >= 2 and np.all(np.isfinite(position[:2])):
                        distance = float(np.linalg.norm(position[:2] - center_xy))
                # Oldest first; for equal timestamps, remove farther voxels first.
                return (observed_at, -distance, tuple(key))

            for key, _ in sorted(remaining, key=eviction_rank)[:overflow]:
                removals[key] = 'max_count'

        counts = {
            'dynamic_ttl': 0,
            'stale_ttl': 0,
            'distance': 0,
            'max_count': 0,
        }
        for key, reason in removals.items():
            if self.voxels.pop(key, None) is not None:
                counts[reason] += 1
        counts['total'] = int(sum(counts.values()))
        counts['remaining'] = len(self.voxels)
        if snapshot:
            # Pruning also advances continuous-time decay, so a non-empty
            # pruning pass is a map change even when no key is removed.
            self.revision += 1
        return counts

    def get_visualization_clouds(self):
        """Build semantic and uncertainty-colored point lists for RViz."""
        semantic_points = []
        uncertainty_points = []
        for key, voxel in self.voxels.items():
            position = voxel['pos']
            class_index = int(np.argmax(self.get_probabilities(key)))
            color = self.class_colors[class_index].tolist()
            semantic_points.append([*position, *color])

            _, _, uncertainty = self.get_uncertainty(key)
            red = int(uncertainty * 255)
            blue = int((1.0 - uncertainty) * 255)
            uncertainty_points.append([*position, red, 0, blue])

        return semantic_points, uncertainty_points
