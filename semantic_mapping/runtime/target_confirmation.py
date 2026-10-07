"""Bounded confirmation of one static candidate, independent of ROS transport."""

from dataclasses import dataclass

import numpy as np


CONFIRMATION_MODES = ('none', 'always', 'age', 'age_quality')


@dataclass(frozen=True)
class ConfirmationPolicy:
    mode: str = 'none'
    max_age_sec: float = 60.0
    min_class_support: float = .4
    max_uncertainty: float = .7
    min_memory_evidence: float = 3.0
    max_position_shift_m: float = .3
    total_sim_sec: float = 30.0
    total_wall_sec: float = 90.0
    per_candidate_sec: float = 12.0
    max_sources: int = 10
    max_candidates: int = 2
    min_new_sources: int = 2
    min_new_evidence: float = .5
    min_new_coverage: float = .5
    max_no_progress_sources: int = 3
    association_radius_m: float = .45
    max_turn_rad: float = .6

    def __post_init__(self):
        if self.mode not in CONFIRMATION_MODES:
            raise ValueError(f'Unsupported confirmation mode: {self.mode}')
        if (self.total_sim_sec <= 0 or self.total_wall_sec <= 0 or self.per_candidate_sec <= 0
                or self.max_sources < self.min_new_sources or self.min_new_sources < 1
                or self.max_candidates < 1 or self.association_radius_m <= 0):
            raise ValueError('Confirmation needs positive budgets and reachable source limits')

    def needs_review(self, candidate):
        if self.mode == 'none':
            return False
        if self.mode == 'always':
            return True
        age = candidate.get('observed_age_sec')
        if age is None:
            return True
        aged = age > self.max_age_sec
        if self.mode == 'age':
            return aged
        weak = (candidate.get('class_support_mean', 0) < self.min_class_support
                or candidate.get('uncertainty_mean', 1) > self.max_uncertainty
                or candidate['evidence'] < self.min_memory_evidence)
        recent = candidate.get('recent_class_support')
        disagreement = recent is not None and recent < self.min_class_support
        unstable = (candidate.get('position_shift_m', 0) > self.max_position_shift_m
                    or not candidate.get('localization_valid', True))
        # Static policy: weakness can require review even for fresh records;
        # age alone does not force review of a strong, stable candidate.
        return weak or (aged and (disagreement or unstable))


class ConfirmationSession:
    """One query owns one cumulative time/source/candidate budget."""

    def __init__(self, policy, class_index, map_time_sec, wall_time_sec, baseline_source_ns,
                 unknown_index=None):
        self.policy = policy
        self.class_index = class_index
        self.unknown_index = unknown_index
        self.start_map_sec = map_time_sec
        self.start_wall_sec = wall_time_sec
        self.baseline_source_ns = baseline_source_ns
        self.seen_sources = set()
        self.sources_used = 0
        self.attempts = 0
        self.status = 'selecting'
        self.reason = ''
        self.candidate = None
        self.candidate_start_sec = map_time_sec
        self.supported_keys = set()
        self.new_evidence = 0.0
        self.new_class_mass = None
        self.valid_sources = 0
        self.min_valid_sources = policy.min_new_sources
        self.no_progress_sources = 0
        self.position_mass = np.zeros(3)
        self.position_weight = 0.0

    def expired(self, map_sec, wall_sec):
        return (map_sec - self.start_map_sec >= self.policy.total_sim_sec
                or wall_sec - self.start_wall_sec >= self.policy.total_wall_sec
                or self.sources_used >= self.policy.max_sources)

    def select(self, candidate, map_sec):
        if self.sources_used >= self.policy.max_sources:
            self.status, self.reason = 'unconfirmed', 'source_budget_exhausted'
            return
        if self.attempts >= self.policy.max_candidates:
            self.status, self.reason = 'unconfirmed', 'candidate_budget_exhausted'
            return
        self.attempts += 1
        self.candidate = candidate
        self.anchor = np.array(candidate['pos'], copy=True)
        self.anchor_keys = set(candidate['voxel_keys'])
        self.candidate_start_sec = map_sec
        self.supported_keys.clear()
        self.new_evidence = 0.0
        self.new_class_mass = None
        self.valid_sources = self.no_progress_sources = 0
        self.min_valid_sources = self.policy.min_new_sources
        self.position_mass[:] = 0
        self.position_weight = 0.0
        self.status = 'observing' if self.policy.needs_review(candidate) else 'confirmed'
        self.reason = 'review_required' if self.status == 'observing' else 'record_accepted'

    @property
    def coverage(self):
        return min(len(self.supported_keys) / max(len(self.anchor_keys), 1), 1.0)

    def matches_regional_update(self, key, probabilities, position):
        # Old support can change class. Newly visible target surfaces can add
        # support, but unrelated floor/wall voxels near a chair are not its vote.
        return (key in self.anchor_keys or (
            np.linalg.norm(position - self.anchor) <= self.policy.association_radius_m
            and int(np.argmax(probabilities)) == self.class_index
            and probabilities[self.class_index] >= self.policy.min_class_support))

    def observe(self, source_id, map_sec, wall_sec, contributions, visible):
        if self.status != 'observing':
            return
        if (map_sec - self.start_map_sec >= self.policy.total_sim_sec
                or wall_sec - self.start_wall_sec >= self.policy.total_wall_sec):
            self.status, self.reason = 'unconfirmed', 'time_budget_exhausted'
            return
        if (source_id in self.seen_sources or source_id[1] <= self.baseline_source_ns
                or map_sec <= self.start_map_sec):
            return
        if self.sources_used >= self.policy.max_sources:
            self.status, self.reason = 'unconfirmed', 'source_budget_exhausted'
            return
        self.seen_sources.add(source_id)
        self.sources_used += 1
        if contributions:
            self.valid_sources += 1
            self.no_progress_sources = 0
            for key, evidence, probabilities, position in contributions:
                if probabilities[self.class_index] >= self.policy.min_class_support:
                    self.supported_keys.add(key)
                self.new_evidence += evidence
                mass = evidence * np.asarray(probabilities)
                self.new_class_mass = (
                    mass if self.new_class_mass is None else self.new_class_mass + mass)
                weight = evidence * probabilities[self.class_index]
                self.position_mass += weight * np.asarray(position)
                self.position_weight += weight
        elif visible:
            self.no_progress_sources += 1
        # A new view may see different surface voxels of the same fixed region.
        if (self.valid_sources >= self.min_valid_sources
                and self.new_evidence >= self.policy.min_new_evidence):
            probabilities = self.new_class_mass / self.new_class_mass.sum()
            support = probabilities[self.class_index]
            if (visible and self.coverage >= self.policy.min_new_coverage
                    and support >= self.policy.min_class_support
                    and int(np.argmax(probabilities)) == self.class_index):
                self.status, self.reason = 'confirmed', 'new_sources_support_candidate'
            elif (support < .25 and np.max(probabilities) >= self.policy.min_class_support
                  and int(np.argmax(probabilities)) != self.unknown_index):
                self.status, self.reason = 'rejected', 'new_class_disagreement'
        if self.no_progress_sources >= self.policy.max_no_progress_sources:
            self.status, self.reason = 'rejected', 'no_new_regional_support'

    def tick(self, map_sec, wall_sec):
        if self.status not in ('selecting', 'observing', 'rejected', 'confirmed'):
            return
        # The final permitted source can confirm. A rejection on that same
        # source cannot spend a new candidate or bypass the common budget.
        if (map_sec - self.start_map_sec >= self.policy.total_sim_sec
                or wall_sec - self.start_wall_sec >= self.policy.total_wall_sec
                or (self.sources_used >= self.policy.max_sources
                    and self.status != 'confirmed')):
            self.status, self.reason = 'unconfirmed', 'cumulative_budget_exhausted'
        elif (self.status == 'observing'
              and map_sec - self.candidate_start_sec >= self.policy.per_candidate_sec):
            self.status, self.reason = 'rejected', 'candidate_wait_exhausted'
