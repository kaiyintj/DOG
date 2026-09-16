"""Diagnostic labels must describe the active profile's actual projection."""

import numpy as np

from semantic_mapping.runtime.segformer_node import (
    find_relevant_model_labels,
    raw_prediction_statistics,
)


def test_indoor_debug_groups_match_checkpoint_projection():
    labels = {
        14: 'door', 19: 'chair', 24: 'shelf', 30: 'armchair', 31: 'seat',
        58: 'screen door', 62: 'bookcase', 64: 'coffee table', 75: 'swivel chair',
    }

    matches = find_relevant_model_labels(
        labels, ['chair', 'table', 'shelf', 'door', 'seat'], ontology_profile='indoor7')

    assert matches == {
        'chair': [(19, 'chair'), (30, 'armchair'), (75, 'swivel chair')],
        'table': [(64, 'coffee table')],
        'shelf': [(24, 'shelf'), (62, 'bookcase')],
        'door': [(14, 'door'), (58, 'screen door')],
        'seat': [(31, 'seat')],  # An explicit raw-label watch remains available.
    }


def test_indoor_statistics_keep_profile_mapping_in_global_and_roi_results():
    labels = {19: 'chair', 31: 'seat', 62: 'bookcase', 64: 'coffee table'}
    mask = np.asarray([[19, 31], [62, 64]], dtype=np.uint8)

    report = raw_prediction_statistics(
        mask, np.full((2, 2), .8), labels,
        watched_labels=['chair', 'shelf', 'table'], ontology_profile='indoor7',
        normalized_roi=(0., .5, 1., .5))

    assert report['global']['watched']['chair']['count'] == 1
    assert report['global']['watched']['shelf']['count'] == 1
    assert report['roi']['watched']['shelf']['fraction'] == .5
    assert report['roi']['watched']['table']['fraction'] == .5
    assert report['roi']['watched']['chair']['count'] == 0


def test_outdoor_electric_bicycle_watch_includes_checkpoint_aliases():
    labels = {0: 'bicycle', 1: 'ebike', 2: 'electric bike', 3: 'e-bike', 4: 'electric_bicycle'}

    matches = find_relevant_model_labels(labels, ['electric_bicycle'])

    assert matches['electric_bicycle'] == [
        (1, 'ebike'), (2, 'electric bike'), (3, 'e-bike'), (4, 'electric_bicycle')]
