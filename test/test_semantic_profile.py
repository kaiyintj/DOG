"""Public profile behavior shared by segmentation and navigation."""

import torch

from semantic_mapping.runtime.semantic_profile import open_profile


def test_outdoor_profile_preserves_order_and_navigation_roles():
    profile = open_profile('outdoor13')

    assert profile.classes == (
        'road', 'building', 'tree', 'person', 'car', 'truck', 'bus',
        'bicycle', 'electric_bicycle', 'motorcycle', 'chair', 'bench',
        'unknown background',
    )
    assert profile.K == 13
    assert profile.unknown_id == 12
    assert profile.traversable_ids == frozenset({0})
    assert profile.dynamic_ids == frozenset({3, 4, 5, 6, 7, 8, 9})
    assert profile.queryable_ids == frozenset(range(12))


def test_indoor_profile_separates_navigation_role_from_query_permission():
    profile = open_profile('indoor7')

    assert profile.classes == (
        'floor', 'wall', 'door', 'chair', 'table', 'shelf', 'bed',
        'unknown background',
    )
    assert profile.K == 8
    assert profile.unknown_id == 7
    assert profile.traversable_ids == frozenset({0})
    assert profile.dynamic_ids == frozenset()
    assert profile.queryable_ids == frozenset({3, 4, 5, 6})
    assert profile.class_specs[2].navigation_role == 'transition'
    assert profile.class_specs[5].navigation_role == 'structural_obstacle'


def test_indoor_projection_aggregates_full_posterior_before_argmax():
    profile = open_profile('indoor7', {
        0: 'chair', 1: 'armchair', 2: 'table', 3: 'rug',
    })
    raw = torch.tensor([0.26, 0.24, 0.40, 0.10]).reshape(1, 4, 1, 1)

    projected = profile.project_probabilities(raw)

    torch.testing.assert_close(
        projected.flatten(),
        torch.tensor([0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.1]),
    )
    assert projected.shape == (1, 8, 1, 1)
    assert projected.argmax(dim=1).item() == 3


def test_queries_distinguish_capability_permission_and_missing_model():
    partial = open_profile('indoor7', {0: 'chair', 1: 'wall'})
    supported = open_profile('indoor7', {0: 'bookcase', 1: 'bed '})

    assert partial.resolve_query('chair').status == 'accepted'
    assert partial.resolve_query('table').status == 'unsupported'
    assert partial.resolve_query('墙').status == 'nonqueryable'
    assert partial.resolve_query('unknown background').status == 'nonqueryable'
    assert partial.resolve_query('spaceship').status == 'unresolved'
    assert open_profile('indoor7').resolve_query('table').status == 'not_ready'
    shelf = supported.resolve_query('去白色书架旁边')
    assert (shelf.status, shelf.class_id, shelf.class_name, shelf.color) == (
        'accepted', 5, 'shelf', 'white',
    )
    assert supported.supported_ids == frozenset({5, 6})


def test_profile_supplies_matching_map_metadata_without_changing_outdoor():
    outdoor = open_profile('outdoor13')
    indoor = open_profile('indoor7')

    assert outdoor.colors == (
        (120, 120, 120), (210, 210, 210), (0, 160, 0), (255, 0, 0),
        (0, 90, 255), (255, 140, 0), (150, 70, 200), (0, 220, 220),
        (70, 170, 255), (255, 0, 180), (160, 100, 40), (255, 220, 0),
        (50, 50, 50),
    )
    assert outdoor.semantic_costs == (
        0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, -1,
    )
    assert outdoor.query_max_extents == (
        100.0, 100.0, 100.0, 1.5, 5.0, 9.0, 14.0, 3.0, 3.5, 3.5,
        2.0, 4.0, 100.0,
    )
    assert len(indoor.colors) == len(indoor.query_max_extents) == 8
    assert indoor.semantic_costs == (0, 100, 100, 100, 100, 100, 100, -1)
