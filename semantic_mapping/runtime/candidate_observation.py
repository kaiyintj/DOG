"""Candidate-level coverage and age on the point-cloud/map time axis."""

import numpy as np


def summarize_observations(cluster, weights, map_time_sec, recent_sec=60.0):
    """Use an 80%-support age, rather than the newest isolated voxel."""
    result = {'observed_age_sec': None, 'fresh_support_ratio': 0.0,
              'known_time_support_ratio': 0.0}
    if map_time_sec is None:
        return result
    stamps = np.array([
        np.nan if item.get('last_effective_observed_at_sec') is None
        else item['last_effective_observed_at_sec'] for item in cluster], dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    known = np.isfinite(stamps) & (stamps <= map_time_sec + 1e-6)
    if total <= 0:
        return result
    ages = np.maximum(map_time_sec - stamps, 0)
    result['known_time_support_ratio'] = float(weights[known].sum() / total)
    result['fresh_support_ratio'] = float(weights[known & (ages <= recent_sec)].sum() / total)
    if result['known_time_support_ratio'] < .8:
        return result
    order = np.argsort(ages[known])
    ordered_age, ordered_weight = ages[known][order], weights[known][order]
    index = np.searchsorted(np.cumsum(ordered_weight), .8 * total, side='left')
    result['observed_age_sec'] = float(ordered_age[min(index, len(order) - 1)])
    return result
