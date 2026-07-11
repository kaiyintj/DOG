import numpy as np

from semantic_mapping.segformer_node import build_project_lookup


def test_segformer_labels_map_to_navigation_classes():
    lookup = build_project_lookup({
        0: 'wall',
        1: 'road',
        2: 'tree',
        3: 'person',
        4: 'car',
        5: 'sky',
    })

    assert np.array_equal(lookup, np.asarray([1, 0, 2, 3, 4, 5]))
