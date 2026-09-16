"""Exercise the offline image interface with an actual tiny local checkpoint."""

import json

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from semantic_mapping.runtime.segformer_image import infer_image, main


@pytest.fixture
def local_image_checkpoint(tmp_path):
    labels = {0: 'wall', 1: 'floor', 2: 'chair', 3: 'armchair', 4: 'table', 5: 'sky'}
    config = SegformerConfig(
        hidden_sizes=[4, 8, 16, 32], depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 2, 4], sr_ratios=[1, 1, 1, 1],
        decoder_hidden_size=8, num_labels=6, id2label=labels,
        label2id={label: index for index, label in labels.items()},
    )
    model = SegformerForSemanticSegmentation(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Raw winner is table=.35; project winner must be chair=.22+.20=.42.
        model.decode_head.classifier.bias.copy_(
            torch.tensor([.1, .1, .22, .2, .35, .03]).log())
    checkpoint = tmp_path / 'checkpoint'
    model.save_pretrained(checkpoint)
    SegformerImageProcessor(size={'height': 32, 'width': 32}).save_pretrained(checkpoint)
    image = tmp_path / 'input.png'
    Image.new('RGB', (32, 32), color=(80, 90, 100)).save(image)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield image, str(checkpoint)
    finally:
        torch.set_num_threads(old_threads)


def test_indoor_image_aggregates_real_logits_before_argmax(local_image_checkpoint):
    image, checkpoint = local_image_checkpoint

    _, mask, raw_mask, confidence, summary = infer_image(
        image, model_id=checkpoint, device='cpu', confidence_threshold=.4,
        ontology_profile='indoor7')

    assert np.all(raw_mask == 4)
    assert np.all(mask == 3)
    assert np.allclose(confidence, .42, atol=1e-6)
    assert summary['ontology_profile'] == 'indoor7'
    assert summary['project_classes'] == [
        'floor', 'wall', 'door', 'chair', 'table', 'shelf', 'bed', 'unknown background']
    assert summary['supported_project_classes'] == ['floor', 'wall', 'chair', 'table']
    assert summary['project']['top'][0]['label'] == 'chair'


def test_image_command_uses_indoor_labels_and_palette(local_image_checkpoint, tmp_path):
    image, checkpoint = local_image_checkpoint
    overlay = tmp_path / 'overlay.png'
    report = tmp_path / 'report.json'

    assert main([
        '--image', str(image), '--model-id', checkpoint, '--device', 'cpu',
        '--ontology-profile', 'indoor7', '--confidence-threshold', '.4',
        '--overlay', str(overlay), '--json', str(report),
    ]) == 0

    summary = json.loads(report.read_text())
    assert summary['ontology_profile'] == 'indoor7'
    assert summary['project']['top'][0]['label'] == 'chair'
    # Blend input (80,90,100) with indoor chair (160,100,40) at alpha=.48.
    assert Image.open(overlay).getpixel((0, 0)) == (118, 94, 71)


@pytest.mark.parametrize('selection, unknown_id, profile_name', [
    ({}, 12, 'outdoor13'),
    ({'ontology_profile': 'indoor7'}, 7, 'indoor7'),
])
def test_low_confidence_image_uses_profile_unknown(
    local_image_checkpoint, selection, unknown_id, profile_name,
):
    image, checkpoint = local_image_checkpoint

    _, mask, _, _, summary = infer_image(
        image, model_id=checkpoint, device='cpu', confidence_threshold=.5, **selection)

    assert np.all(mask == unknown_id)
    assert summary['ontology_profile'] == profile_name
    assert summary['project']['top'][0]['label'] == 'unknown background'
