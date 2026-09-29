#!/usr/bin/env python3
"""
Run the project's SegFormer checkpoint on one local image.

This is an offline inspection entry point.  It uses the same checkpoint,
project-class mapping, confidence threshold and logit resizing as
``segformer_node`` but does not start ROS or publish navigation commands.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_mapping.runtime.segformer_core import PROJECT_COLORS
from semantic_mapping.runtime.semantic_profile import open_profile


def _parse_watch_labels(values):
    labels = []
    for value in values or ():
        labels.extend(part.strip() for part in str(value).split(','))
    return [label for label in labels if label]


def _project_statistics(project_mask, confidence, top_k, classes):
    pixel_count = int(project_mask.size)
    counts = np.bincount(
        project_mask.reshape(-1), minlength=len(classes))
    ranked_ids = np.argsort(counts)[::-1]
    entries = []
    for class_id in ranked_ids:
        count = int(counts[class_id])
        if count <= 0:
            continue
        selected = project_mask == class_id
        entries.append({
            'id': int(class_id),
            'label': classes[int(class_id)],
            'count': count,
            'fraction': count / max(pixel_count, 1),
            'mean_confidence': float(np.mean(confidence[selected])),
        })
        if len(entries) >= max(1, int(top_k)):
            break
    return {
        'pixel_count': pixel_count,
        'top': entries,
    }


def _overlay(rgb_image, project_mask, alpha=0.48, colors=PROJECT_COLORS):
    colors = np.asarray(colors)[project_mask].astype(np.float32)
    source = np.asarray(rgb_image, dtype=np.float32)
    blended = source * (1.0 - alpha) + colors * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _load_model(model_id, device_name, use_fp16):
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            'SegFormer dependencies are missing. Install the packages from '
            'requirements-segformer.txt.') from exc

    requested = str(device_name)
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA requested but unavailable; using CPU.')
        requested = 'cpu'
    if requested not in {'cpu', 'cuda'}:
        raise ValueError(
            f'Unsupported device {device_name!r}; use cpu, cuda or auto.')

    print(f'Loading SegFormer {model_id} on {requested}...')
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.eval().to(requested)
    if requested == 'cuda' and use_fp16:
        model.half()
    id2label = {
        int(class_id): str(label)
        for class_id, label in model.config.id2label.items()
    }
    return torch, functional, processor, model, requested, id2label


def infer_image(
    image_path,
    model_id='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    device='auto',
    use_fp16=True,
    confidence_threshold=0.45,
    posterior_temperature=1.0,
    top_k=8,
    ontology_profile='outdoor13',
):
    """Return project mask, confidence map and a JSON-serializable summary."""
    profile = open_profile(ontology_profile)
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Image does not exist: {image_path}')

    with Image.open(image_path) as image:
        rgb_image = np.asarray(image.convert('RGB'), dtype=np.uint8)

    torch, functional, processor, model, device, id2label = _load_model(
        model_id, device, use_fp16)
    profile = open_profile(profile.id, id2label)
    posterior_temperature = float(posterior_temperature)
    if not np.isfinite(posterior_temperature) or posterior_temperature <= 0.0:
        raise ValueError('posterior_temperature must be finite and positive')
    height, width = rgb_image.shape[:2]
    inputs = processor(
        images=Image.fromarray(rgb_image, mode='RGB'),
        return_tensors='pt',
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    if device == 'cuda' and use_fp16:
        inputs = {
            name: value.half() if value.is_floating_point() else value
            for name, value in inputs.items()
        }

    with torch.inference_mode():
        native_logits = model(**inputs).logits.float()
        raw_native_probabilities = torch.softmax(
            native_logits / posterior_temperature,
            dim=1,
        )
        project_native_probabilities = profile.project_probabilities(raw_native_probabilities)
        project_probabilities = functional.interpolate(
            project_native_probabilities,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        project_probabilities /= project_probabilities.sum(
            dim=1, keepdim=True).clamp_min(1e-12)
        confidence, project_prediction = project_probabilities.max(dim=1)

        raw_full_logits = functional.interpolate(
            native_logits,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        raw_prediction = torch.softmax(
            raw_full_logits / posterior_temperature,
            dim=1,
        ).argmax(dim=1)

    raw_mask = raw_prediction[0].cpu().numpy().astype(np.uint8)
    confidence_map = confidence[0].cpu().numpy().astype(np.float32)
    project_mask = project_prediction[0].cpu().numpy().astype(np.uint8)
    project_mask[confidence_map < float(np.clip(
        confidence_threshold, 0.0, 1.0))] = profile.unknown_id

    summary = {
        'image': str(image_path),
        'model': model_id,
        'ontology_profile': profile.id,
        'project_classes': list(profile.classes),
        'device': device,
        'confidence_threshold': float(confidence_threshold),
        'posterior_temperature': posterior_temperature,
        'probability_mapping': 'aggregate_before_argmax',
        'image_size': [int(width), int(height)],
        'checkpoint_labels': id2label,
        'supported_project_classes': [
            profile.classes[index] for index in sorted(profile.supported_ids)],
        'project': _project_statistics(project_mask, confidence_map, top_k, profile.classes),
    }
    return rgb_image, project_mask, raw_mask, confidence_map, summary


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Run the project SegFormer model on a local image.')
    parser.add_argument('--image', required=True, help='Input JPG/PNG image.')
    parser.add_argument(
        '--ontology-profile', choices=('outdoor13', 'indoor7'), default='outdoor13',
        help='Project label/role profile; the selected checkpoint must supply its labels.')
    parser.add_argument(
        '--model-id',
        default='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
        help='Hugging Face model ID or local checkpoint directory.')
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument(
        '--fp16', action=argparse.BooleanOptionalAction, default=True,
        help='Use FP16 on CUDA (default: enabled).')
    parser.add_argument('--confidence-threshold', type=float, default=0.45)
    parser.add_argument(
        '--posterior-temperature', type=float, default=1.0,
        help=(
            'Positive temperature applied before project probability '
            'aggregation.'))
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument(
        '--overlay', type=Path,
        help='Optional path for an RGB segmentation overlay.')
    parser.add_argument(
        '--mask', type=Path,
        help='Optional path for the selected profile project mask (mono8).')
    parser.add_argument(
        '--json', dest='json_path', type=Path,
        help='Optional path for the JSON result.')
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    rgb, project_mask, _, _, summary = infer_image(
        args.image,
        model_id=args.model_id,
        device=args.device,
        use_fp16=args.fp16,
        confidence_threshold=args.confidence_threshold,
        posterior_temperature=args.posterior_temperature,
        top_k=args.top_k,
        ontology_profile=args.ontology_profile,
    )

    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            _overlay(rgb, project_mask, colors=open_profile(args.ontology_profile).colors),
            mode='RGB').save(args.overlay)
        summary['overlay'] = str(args.overlay.resolve())
    if args.mask:
        args.mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(project_mask, mode='L').save(args.mask)
        summary['mask'] = str(args.mask.resolve())
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
