import numpy as np

from semantic_mapping.semantic_schema import (
    DEFAULT_CLASSES,
    color_membership_score,
    parse_semantic_query,
    resolve_clip_model_name,
)


def test_bicycle_and_color_query_are_parsed():
    class_index, color_name = parse_semantic_query(
        'blue bicycle', DEFAULT_CLASSES)

    assert DEFAULT_CLASSES[class_index] == 'bicycle'
    assert color_name == 'blue'


def test_chinese_color_car_query_is_parsed():
    class_index, color_name = parse_semantic_query(
        '蓝色的汽车', DEFAULT_CLASSES)

    assert DEFAULT_CLASSES[class_index] == 'car'
    assert color_name == 'blue'


def test_white_truck_query_is_parsed():
    class_index, color_name = parse_semantic_query(
        'white truck', DEFAULT_CLASSES)

    assert DEFAULT_CLASSES[class_index] == 'truck'
    assert color_name == 'white'


def test_blue_color_score_rejects_red_surface():
    blue_score = color_membership_score(
        np.array([20.0, 60.0, 220.0]), 'blue')
    red_score = color_membership_score(
        np.array([220.0, 30.0, 20.0]), 'blue')

    assert blue_score > 0.5
    assert red_score < 0.1


def test_white_color_score_rejects_red_surface():
    white_score = color_membership_score(
        np.array([225.0, 230.0, 235.0]), 'white')
    red_score = color_membership_score(
        np.array([220.0, 30.0, 20.0]), 'white')

    assert white_score > 0.5
    assert red_score < 0.1


def test_openai_clip_redirects_to_quickgelu_variant():
    assert resolve_clip_model_name('ViT-B-32', 'openai') == 'ViT-B-32-quickgelu'
    assert resolve_clip_model_name('ViT-L-14', 'openai') == 'ViT-L-14-quickgelu'


def test_quickgelu_variant_is_not_double_suffixed():
    assert (
        resolve_clip_model_name('ViT-B-32-quickgelu', 'openai')
        == 'ViT-B-32-quickgelu'
    )


def test_non_openai_weights_keep_gelu_model_name():
    assert (
        resolve_clip_model_name('ViT-B-32', 'laion2b_s34b_b79k')
        == 'ViT-B-32'
    )


def test_openai_clip_falls_back_when_quickgelu_architecture_is_unavailable():
    assert (
        resolve_clip_model_name(
            'Custom-ViT',
            'openai',
            available_models=['Custom-ViT'],
        )
        == 'Custom-ViT'
    )


def test_openai_clip_uses_available_quickgelu_architecture():
    assert (
        resolve_clip_model_name(
            'ViT-B-32',
            'openai',
            available_models=['ViT-B-32', 'ViT-B-32-quickgelu'],
        )
        == 'ViT-B-32-quickgelu'
    )
