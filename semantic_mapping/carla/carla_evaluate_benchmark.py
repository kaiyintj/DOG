#!/usr/bin/env python3
"""Evaluate CLIP and SegFormer on a CARLA image-recognition dataset."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from semantic_mapping.carla.carla_benchmark import (
    COARSE_LABELS,
    SPAWN_CLASSES,
    TWO_WHEEL_CLASSES,
    VEHICLE_CLASSES,
    actor_instance_id,
    augment_grid_truth_with_instances,
    bbox_from_mask,
    build_project_lookup,
    byte_swap_instance_ids,
    classify_instance_color,
    confusion_matrix,
    grid_cell_truth,
    make_coarse_masks,
    metrics_from_confusion,
)
from semantic_mapping.runtime.semantic_schema import (
    COLOR_ALIASES,
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    resolve_clip_model_name,
)
from semantic_mapping.runtime.segformer_core import (
    aggregate_project_probability_tensor,
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _device_name(torch, requested):
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA is unavailable; using CPU.', file=sys.stderr)
        return 'cpu'
    return requested


def _encode_clip_texts(model, tokenizer, torch, device, templates, texts):
    features = []
    with torch.inference_mode():
        for text in texts:
            prompts = [template.format(text) for template in templates]
            tokens = tokenizer(prompts).to(device)
            prompt_features = model.encode_text(tokens)
            prompt_features = prompt_features / prompt_features.norm(
                dim=-1, keepdim=True).clamp(min=1e-12)
            feature = prompt_features.mean(dim=0)
            feature = feature / feature.norm().clamp(min=1e-12)
            features.append(feature)
    return torch.stack(features)


class ClipEvaluator:
    """Offline equivalent of the current grid-based ``clip_node``."""

    def __init__(
        self,
        model_name,
        pretrained,
        device,
        grid_rows,
        grid_cols,
        vocab,
        templates,
    ):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'CLIP evaluation needs torch and open_clip_torch.') from exc
        self.torch = torch
        self.device = _device_name(torch, device)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.vocab = list(vocab)
        self.templates = list(templates)
        resolved_model_name = resolve_clip_model_name(
            model_name,
            pretrained,
            available_models=open_clip.list_models(),
        )
        self.resolved_model_name = resolved_model_name
        print(
            f'Loading CLIP {resolved_model_name}/{pretrained} '
            f'on {self.device}...',
            flush=True,
        )
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                resolved_model_name, pretrained=pretrained))
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(resolved_model_name)
        self.text_features = _encode_clip_texts(
            self.model,
            self.tokenizer,
            torch,
            self.device,
            self.templates,
            self.vocab,
        )
        self.query_cache = {}

    def _patches(self, image):
        width, height = image.size
        patch_h = height // self.grid_rows
        patch_w = width // self.grid_cols
        patches = []
        bounds = []
        for row in range(self.grid_rows):
            for column in range(self.grid_cols):
                x0 = column * patch_w
                y0 = row * patch_h
                x1 = width if column == self.grid_cols - 1 else (
                    column + 1) * patch_w
                y1 = height if row == self.grid_rows - 1 else (
                    row + 1) * patch_h
                patches.append(image.crop((x0, y0, x1, y1)))
                bounds.append((x0, y0, x1, y1))
        return patches, bounds

    def infer(self, image):
        patches, bounds = self._patches(image)
        batch = self.torch.stack([
            self.preprocess(patch) for patch in patches
        ]).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(
                dim=-1, keepdim=True).clamp(min=1e-12)
            similarities = features @ self.text_features.T
        return (
            features.cpu().numpy(),
            similarities.cpu().numpy(),
            bounds,
        )

    def query_feature(self, text):
        if text not in self.query_cache:
            feature = _encode_clip_texts(
                self.model,
                self.tokenizer,
                self.torch,
                self.device,
                self.templates,
                [text],
            )[0]
            self.query_cache[text] = feature.cpu().numpy()
        return self.query_cache[text]

    def image_feature(self, image):
        """Encode one RGB crop for object and attribute classification."""
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            feature = self.model.encode_image(batch)
            feature = feature / feature.norm(
                dim=-1, keepdim=True).clamp(min=1e-12)
        return feature[0].cpu().numpy()


class SegformerEvaluator:
    """Offline equivalent of ``segformer_node`` preprocessing and mapping."""

    def __init__(
        self,
        model_id,
        device,
        confidence_threshold,
        use_fp16,
        posterior_temperature=1.0,
        revision=None,
    ):
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                'SegFormer evaluation needs torch, transformers, tokenizers '
                'and pillow.') from exc
        self.torch = torch
        self.functional = functional
        self.device = _device_name(torch, device)
        self.confidence_threshold = float(confidence_threshold)
        self.posterior_temperature = float(posterior_temperature)
        if (
            not np.isfinite(self.posterior_temperature)
            or self.posterior_temperature <= 0.0
        ):
            raise ValueError(
                'SegFormer posterior temperature must be finite and positive')
        self.use_fp16 = bool(use_fp16 and self.device == 'cuda')
        self.model_id = str(model_id)
        self.requested_revision = revision
        print(f'Loading SegFormer {model_id} on {self.device}...', flush=True)
        pretrained_kwargs = (
            {'revision': revision} if revision is not None else {})
        self.processor = SegformerImageProcessor.from_pretrained(
            model_id, **pretrained_kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, **pretrained_kwargs)
        self.model.eval().to(self.device)
        if self.use_fp16:
            self.model.half()
        self.id2label = {
            int(class_id): str(label)
            for class_id, label in self.model.config.id2label.items()
        }
        self.project_lookup = build_project_lookup(self.id2label)
        self.project_lookup_tensor = self.torch.as_tensor(
            self.project_lookup,
            dtype=self.torch.long,
            device=self.device,
        )

    def _infer_native_tensors(self, image):
        """Return raw logits and the full project posterior at decoder scale."""
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {
            name: value.to(self.device) for name, value in inputs.items()
        }
        if self.use_fp16:
            inputs = {
                name: (
                    value.half() if value.is_floating_point() else value)
                for name, value in inputs.items()
            }
        with self.torch.inference_mode():
            native_logits = self.model(**inputs).logits.float()
            raw_native_probabilities = self.torch.softmax(
                native_logits / self.posterior_temperature,
                dim=1,
            )
            project_native_probabilities = (
                aggregate_project_probability_tensor(
                    raw_native_probabilities,
                    self.project_lookup_tensor,
                    len(DEFAULT_CLASSES),
                )
            )
        return native_logits, project_native_probabilities

    def infer_project_posterior(self, image):
        """
        Return the native ``H x W x K`` project posterior used at runtime.

        Unlike :meth:`infer`, this method deliberately does not resize or take
        an argmax.  The reliability benchmark samples this grid with
        ``semantic_posterior.sample_posterior_bilinear``, matching the online
        SegFormer-to-LiDAR association path.
        """
        _, probabilities = self._infer_native_tensors(image)
        return (
            probabilities[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def infer(self, image):
        """Run the unchanged legacy image-benchmark inference path."""
        width, height = image.size
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {
            name: value.to(self.device) for name, value in inputs.items()
        }
        if self.use_fp16:
            inputs = {
                name: (
                    value.half() if value.is_floating_point() else value)
                for name, value in inputs.items()
            }
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.functional.interpolate(
                logits,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
            # Keep the pre-existing image benchmark byte-for-byte in meaning:
            # resize raw checkpoint logits, then softmax/argmax, then map the
            # winning raw class into the project ontology.  The new reliability
            # benchmark uses infer_project_posterior() instead.
            probabilities = self.torch.softmax(logits.float(), dim=1)
            confidence, prediction = probabilities.max(dim=1)
        raw_mask = prediction[0].cpu().numpy().astype(np.uint8)
        confidence_map = confidence[0].cpu().numpy().astype(np.float32)
        project_mask = self.project_lookup[raw_mask]
        project_mask[confidence_map < self.confidence_threshold] = (
            len(DEFAULT_CLASSES) - 1)
        return project_mask, confidence_map, raw_mask


def _candidate_classes(true_class):
    """
    Return the fine classes a detector should choose among for one object.

    Detection and fine classification are scored inside the object's own coarse
    group so a bicycle is never counted as ``detected`` because SegFormer
    painted it as a car. Four-wheelers compete over car/truck/bus, ridden
    two-wheelers over bicycle/motorcycle.
    """
    if true_class in TWO_WHEEL_CLASSES:
        return TWO_WHEEL_CLASSES
    return VEHICLE_CLASSES


def _semantic_name(semantic_tag, semantic_tag_ids):
    semantic_tag = int(semantic_tag)
    for name in SPAWN_CLASSES + ('vehicle', 'person', 'rider', 'road'):
        if semantic_tag in semantic_tag_ids.get(name, ()):
            return 'person' if name == 'rider' else name
    return 'background'


def _class_counter():
    return {
        name: {'tp': 0, 'fp': 0, 'fn': 0}
        for name in ('road', 'person') + SPAWN_CLASSES + ('vehicle',)
    }


def _update_binary_counter(counter, truth, prediction):
    labels = set(counter)
    for label in labels:
        truth_active = label in truth
        prediction_active = label in prediction
        if truth_active and prediction_active:
            counter[label]['tp'] += 1
        elif prediction_active:
            counter[label]['fp'] += 1
        elif truth_active:
            counter[label]['fn'] += 1


def _counter_metrics(counter):
    result = {}
    for label, values in counter.items():
        tp = values['tp']
        fp = values['fp']
        fn = values['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result[label] = {
            **values,
            'precision': precision,
            'recall': recall,
            'f1': (
                2.0 * precision * recall / max(precision + recall, 1e-12)),
        }
    return result


def _truth_labels_for_grid(item):
    labels = set(item['labels'])
    if any(label in labels for label in SPAWN_CLASSES):
        labels.add('vehicle')
    return labels


def _prediction_labels_for_grid(predicted_class):
    labels = {predicted_class}
    if predicted_class in SPAWN_CLASSES:
        labels.add('vehicle')
    return labels


def _instance_grid_hits(instance_mask, bounds, min_pixels):
    hits = set()
    for index, (x0, y0, x1, y1) in enumerate(bounds):
        if np.count_nonzero(instance_mask[y0:y1, x0:x1]) >= min_pixels:
            hits.add(index)
    return hits


def _masked_instance_crop(rgb, object_mask, bbox, padding_fraction=0.08):
    """Crop one exact CARLA instance on a neutral background for CLIP."""
    image = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(object_mask, dtype=bool)
    height, width = mask.shape
    x0, y0, x1, y1 = bbox
    padding = int(round(max(x1 - x0, y1 - y0) * padding_fraction))
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 127
    return Image.fromarray(crop, mode='RGB')


def _grouped_accuracy(rows, group_key, metric_keys):
    """Summarize supported per-instance metrics by class or color."""
    grouped = {}
    values = sorted({str(row[group_key]) for row in rows})
    for value in values:
        selected = [row for row in rows if str(row[group_key]) == value]
        grouped[value] = {'count': len(selected)}
        for metric_key in metric_keys:
            if metric_key not in selected[0]:
                continue
            grouped[value][metric_key] = float(np.mean([
                row[metric_key] for row in selected
            ]))
    return grouped


def _save_segformer_overlay(path, rgb, project_mask):
    palette = np.asarray(DEFAULT_CLASS_COLORS, dtype=np.uint8)
    color_mask = palette[project_mask]
    overlay = (
        0.55 * np.asarray(rgb, dtype=np.float32)
        + 0.45 * color_mask.astype(np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay, mode='RGB').save(path)


def _save_instance_overlay(path, image, annotations):
    """Save human-readable GT/CLIP/SegFormer labels over CARLA RGB."""
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for annotation in annotations:
        x0, y0, x1, y1 = annotation['bbox']
        draw.rectangle((x0, y0, x1, y1), outline=(255, 220, 0), width=3)
        text = '\n'.join(annotation['lines'])
        left, top, right, bottom = draw.multiline_textbbox((x0, y0), text)
        text_y = max(0, y0 - (bottom - top) - 6)
        draw.rectangle(
            (left - 2, text_y - 2, right + 2, text_y + bottom - top + 2),
            fill=(0, 0, 0),
        )
        draw.multiline_text((x0, text_y), text, fill=(255, 255, 255))
    rendered.save(path)


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Run the project CLIP and/or SegFormer image frontends against '
            'CARLA semantic and instance ground truth.'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument(
        '--backend',
        choices=('both', 'clip', 'segformer'),
        default='both',
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--save-overlay-every', type=int, default=10)
    parser.add_argument('--min-grid-fraction', type=float, default=0.02)
    parser.add_argument('--min-instance-pixels', type=int, default=80)
    parser.add_argument('--min-instance-cell-pixels', type=int, default=20)
    parser.add_argument(
        '--min-instance-semantic-fraction',
        type=float,
        default=0.10,
        help=(
            'Minimum part of a CARLA instance that SegFormer must label as '
            'one of the evaluated vehicle classes before its fine class is '
            'scored as detected.'),
    )
    parser.add_argument(
        '--grid-rows',
        type=int,
        default=0,
        help='CLIP grid rows; 0 uses the dataset manifest recommendation.',
    )
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=0,
        help='CLIP grid columns; 0 uses the dataset manifest recommendation.',
    )
    parser.add_argument('--clip-model', default='ViT-B-32')
    parser.add_argument('--clip-pretrained', default='openai')
    parser.add_argument(
        '--clip-vocab',
        nargs='+',
        default=list(DEFAULT_CLASSES),
    )
    parser.add_argument(
        '--segformer-model',
        default='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    )
    parser.add_argument('--segformer-confidence', type=float, default=0.45)
    parser.add_argument(
        '--segformer-revision', default='',
        help='Optional Hugging Face revision/commit for reproducible loading.')
    parser.add_argument('--fp16', action='store_true')
    return parser


def evaluate(args):
    dataset = Path(args.dataset).expanduser().resolve()
    manifest = _read_json(dataset / 'manifest.json')
    instance_encoding = str(manifest.get('instance_id_encoding', ''))
    legacy_instance_encoding = (
        instance_encoding != 'actor_id=G+(B<<8)'
    )
    report_warnings = []
    if legacy_instance_encoding:
        warning = (
            'Dataset predates the corrected CARLA instance-ID byte order. '
            'The evaluator will byte-swap masks and match actor IDs, but '
            'formal results should use a newly captured format-v2 dataset.'
        )
        print(f'WARNING: {warning}', file=sys.stderr)
        report_warnings.append(warning)
    grid_recommendation = manifest.get('grid_recommendation', {})
    recommended_rows = int(grid_recommendation.get('rows', 4))
    recommended_cols = int(grid_recommendation.get('cols', 6))
    if args.grid_rows <= 0:
        args.grid_rows = recommended_rows
    elif args.grid_rows != recommended_rows:
        warning = (
            f'CLI grid_rows={args.grid_rows} differs from dataset '
            f'recommendation {recommended_rows}.')
        print(f'WARNING: {warning}', file=sys.stderr)
        report_warnings.append(warning)
    if args.grid_cols <= 0:
        args.grid_cols = recommended_cols
    elif args.grid_cols != recommended_cols:
        warning = (
            f'CLI grid_cols={args.grid_cols} differs from dataset '
            f'recommendation {recommended_cols}.')
        print(f'WARNING: {warning}', file=sys.stderr)
        report_warnings.append(warning)
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        raise ValueError('grid_rows and grid_cols must both be positive.')
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dataset / 'results' / time.strftime('%Y%m%d_%H%M%S')
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / 'overlays').mkdir(exist_ok=True)
    (output / 'detections').mkdir(exist_ok=True)

    semantic_tag_ids = {
        name: [int(value) for value in values]
        for name, values in manifest['semantic_tag_ids'].items()
    }
    frame_paths = list(manifest['frames'])
    if args.limit > 0:
        frame_paths = frame_paths[:args.limit]
    if not frame_paths:
        raise RuntimeError('Dataset manifest contains no captured frames.')

    clip = None
    segformer = None
    templates = (
        'a photo of a {}.',
        'a photo of the {}.',
        'a {} in the scene.',
    )
    if args.backend in ('both', 'clip'):
        clip = ClipEvaluator(
            args.clip_model,
            args.clip_pretrained,
            args.device,
            args.grid_rows,
            args.grid_cols,
            args.clip_vocab,
            templates,
        )
    if args.backend in ('both', 'segformer'):
        segformer = SegformerEvaluator(
            args.segformer_model,
            args.device,
            args.segformer_confidence,
            args.fp16,
            revision=args.segformer_revision or None,
        )

    coarse_confusion = np.zeros(
        (len(COARSE_LABELS), len(COARSE_LABELS)), dtype=np.int64)
    clip_grid_counter = _class_counter()
    clip_frame_counter = _class_counter()
    clip_query_results = {
        'object': {'attempts': 0, 'hits': 0},
        'color_object': {'attempts': 0, 'hits': 0},
        'person': {'attempts': 0, 'hits': 0},
    }
    instance_rows = []
    person_rows = []
    frame_rows = []
    usable_vehicle_observations = {
        name: 0 for name in SPAWN_CLASSES
    }

    class_index = {
        name: index for index, name in enumerate(DEFAULT_CLASSES)
    }
    for frame_index, relative_metadata in enumerate(frame_paths):
        metadata = _read_json(dataset / relative_metadata)
        rgb_image = Image.open(dataset / metadata['rgb']).convert('RGB')
        rgb = np.asarray(rgb_image)
        semantic = np.asarray(
            Image.open(dataset / metadata['semantic']), dtype=np.uint8)
        instance = np.asarray(
            Image.open(dataset / metadata['instance']), dtype=np.uint16)
        if legacy_instance_encoding:
            instance = byte_swap_instance_ids(instance)
        gt_grids = grid_cell_truth(
            semantic,
            semantic_tag_ids,
            args.grid_rows,
            args.grid_cols,
            args.min_grid_fraction,
        )
        gt_grids = augment_grid_truth_with_instances(
            gt_grids,
            instance,
            metadata.get('visible_vehicles', []),
            args.grid_rows,
            args.grid_cols,
            args.min_instance_cell_pixels,
        )

        frame_record = {
            'frame': int(metadata['frame']),
            'visible_vehicles': len(metadata.get('visible_vehicles', [])),
            'visible_people': len(metadata.get('visible_people', [])),
        }
        clip_features = None
        clip_bounds = None
        if clip is not None:
            start = time.perf_counter()
            clip_features, similarities, clip_bounds = clip.infer(rgb_image)
            predicted_indices = np.argmax(similarities, axis=1)
            predicted_names = [
                clip.vocab[int(index)] for index in predicted_indices
            ]
            frame_record['clip_seconds'] = time.perf_counter() - start
            truth_frame = set()
            prediction_frame = set()
            for grid_truth, predicted_name in zip(
                    gt_grids, predicted_names):
                truth_labels = _truth_labels_for_grid(grid_truth)
                prediction_labels = _prediction_labels_for_grid(
                    predicted_name)
                _update_binary_counter(
                    clip_grid_counter, truth_labels, prediction_labels)
                truth_frame.update(truth_labels)
                prediction_frame.update(prediction_labels)
            _update_binary_counter(
                clip_frame_counter, truth_frame, prediction_frame)

        project_mask = None
        confidence = None
        if segformer is not None:
            start = time.perf_counter()
            project_mask, confidence, _ = segformer.infer(rgb_image)
            frame_record['segformer_seconds'] = time.perf_counter() - start
            truth_coarse, prediction_coarse = make_coarse_masks(
                semantic, project_mask, semantic_tag_ids)
            coarse_confusion += confusion_matrix(
                truth_coarse,
                prediction_coarse,
                len(COARSE_LABELS),
            )
            frame_record['segformer_mean_confidence'] = float(
                np.mean(confidence))
            if (
                args.save_overlay_every > 0
                and frame_index % args.save_overlay_every == 0
            ):
                _save_segformer_overlay(
                    output / 'overlays' / (
                        f'{int(metadata["frame"]):08d}.png'),
                    rgb,
                    project_mask,
                )

        query_masks = {}
        frame_annotations = []
        for visible in metadata.get('visible_vehicles', []):
            instance_id = actor_instance_id(visible['actor_id'])
            object_mask = instance == instance_id
            visible_pixels = int(np.count_nonzero(object_mask))
            if visible_pixels < args.min_instance_pixels:
                continue
            bbox = bbox_from_mask(object_mask)
            if bbox is None:
                continue
            true_class = visible.get('benchmark_class')
            if true_class not in SPAWN_CLASSES:
                true_class = _semantic_name(
                    visible['semantic_tag'], semantic_tag_ids)
            candidate_classes = _candidate_classes(true_class)
            true_color = visible.get(
                'canonical_color_name',
                visible.get('requested_color_name', 'unknown'),
            )
            if true_class in usable_vehicle_observations:
                usable_vehicle_observations[true_class] += 1
            predicted_color, color_scores = classify_instance_color(
                rgb[object_mask])
            base = {
                'frame': int(metadata['frame']),
                'actor_id': int(visible['actor_id']),
                'type_id': visible['type_id'],
                'instance_id': instance_id,
                'visible_pixels': visible_pixels,
                'true_class': true_class,
                'true_color': true_color,
                'predicted_color': predicted_color,
                'color_correct': int(predicted_color == true_color),
                'true_color_score': float(
                    color_scores.get(true_color, 0.0)),
            }

            if project_mask is not None:
                counts = np.bincount(
                    project_mask[object_mask],
                    minlength=len(DEFAULT_CLASSES),
                )
                group_counts = {
                    name: int(counts[class_index[name]])
                    for name in candidate_classes
                }
                vehicle_fraction = sum(group_counts.values()) / max(
                    visible_pixels, 1)
                if (
                    vehicle_fraction
                    >= float(args.min_instance_semantic_fraction)
                ):
                    predicted_class = max(
                        group_counts, key=group_counts.get)
                else:
                    predicted_class = 'not detected'
                base.update({
                    'segformer_predicted_class': predicted_class,
                    'segformer_vehicle_fraction': vehicle_fraction,
                    'segformer_detected': int(
                        predicted_class != 'not detected'),
                    'segformer_class_correct': int(
                        predicted_class == true_class),
                    'segformer_attribute_correct': int(
                        predicted_class == true_class
                        and predicted_color == true_color),
                })

            if clip_features is not None:
                target_cells = _instance_grid_hits(
                    object_mask,
                    clip_bounds,
                    args.min_instance_cell_pixels,
                )
                for query_kind, query_text in (
                    ('object', true_class),
                    ('color_object', f'{true_color} {true_class}'),
                ):
                    query_masks.setdefault(query_kind, {}).setdefault(
                        query_text, set()).update(target_cells)

                class_query = clip.query_feature(true_class)
                class_scores = clip_features @ class_query
                best_class_cell = int(np.argmax(class_scores))
                attribute_query = clip.query_feature(
                    f'{true_color} {true_class}')
                attribute_scores = clip_features @ attribute_query
                best_attribute_cell = int(np.argmax(attribute_scores))

                crop_feature = clip.image_feature(
                    _masked_instance_crop(rgb, object_mask, bbox))
                object_scores = {
                    name: float(
                        crop_feature @ clip.query_feature(name))
                    for name in candidate_classes
                }
                clip_predicted_class = max(
                    object_scores, key=object_scores.get)
                color_scores_clip = {
                    color: float(
                        crop_feature @ clip.query_feature(
                            f'{color} {true_class}'))
                    for color in COLOR_ALIASES
                }
                clip_predicted_color = max(
                    color_scores_clip, key=color_scores_clip.get)
                base.update({
                    'clip_predicted_class': clip_predicted_class,
                    'clip_class_correct': int(
                        clip_predicted_class == true_class),
                    'clip_predicted_color': clip_predicted_color,
                    'clip_color_correct': int(
                        clip_predicted_color == true_color),
                    'clip_attribute_correct': int(
                        clip_predicted_class == true_class
                        and clip_predicted_color == true_color),
                    'clip_object_best_cell': best_class_cell,
                    'clip_object_localized': int(
                        best_class_cell in target_cells),
                    'clip_attribute_best_cell': best_attribute_cell,
                    'clip_attribute_localized': int(
                        best_attribute_cell in target_cells),
                })
            instance_rows.append(base)
            annotation_lines = [
                f'GT: {true_color} {true_class}',
            ]
            if clip_features is not None:
                annotation_lines.append(
                    'CLIP: '
                    f'{base["clip_predicted_color"]} '
                    f'{base["clip_predicted_class"]}'
                )
            if project_mask is not None:
                annotation_lines.append(
                    'SegFormer: '
                    f'{base["segformer_predicted_class"]} '
                    f'({base["segformer_vehicle_fraction"]:.0%})'
                )
            frame_annotations.append({
                'bbox': list(bbox),
                'lines': annotation_lines,
            })

        person_class_id = class_index['person']
        for visible in metadata.get('visible_people', []):
            instance_id = actor_instance_id(visible['actor_id'])
            object_mask = instance == instance_id
            visible_pixels = int(np.count_nonzero(object_mask))
            if visible_pixels < args.min_instance_pixels:
                continue
            bbox = bbox_from_mask(object_mask)
            if bbox is None:
                continue
            person_base = {
                'frame': int(metadata['frame']),
                'actor_id': int(visible['actor_id']),
                'instance_id': instance_id,
                'visible_pixels': visible_pixels,
                'true_class': 'person',
            }
            if project_mask is not None:
                person_pixels = int(np.count_nonzero(
                    project_mask[object_mask] == person_class_id))
                person_fraction = person_pixels / max(visible_pixels, 1)
                person_base.update({
                    'segformer_person_fraction': person_fraction,
                    'segformer_detected': int(
                        person_fraction
                        >= float(args.min_instance_semantic_fraction)),
                })
            if clip_features is not None:
                target_cells = _instance_grid_hits(
                    object_mask,
                    clip_bounds,
                    args.min_instance_cell_pixels,
                )
                person_query = clip.query_feature('person')
                best_cell = int(np.argmax(clip_features @ person_query))
                person_base['clip_person_localized'] = int(
                    best_cell in target_cells)
                query_masks.setdefault('person', {}).setdefault(
                    'person', set()).update(target_cells)
            person_rows.append(person_base)

        if clip_features is not None:
            for query_kind, queries in query_masks.items():
                for query_text, target_cells in queries.items():
                    if not target_cells:
                        continue
                    query_feature = clip.query_feature(query_text)
                    best_cell = int(np.argmax(
                        clip_features @ query_feature))
                    clip_query_results[query_kind]['attempts'] += 1
                    clip_query_results[query_kind]['hits'] += int(
                        best_cell in target_cells)
        if (
            frame_annotations
            and args.save_overlay_every > 0
            and frame_index % args.save_overlay_every == 0
        ):
            _save_instance_overlay(
                output / 'detections' / (
                    f'{int(metadata["frame"]):08d}.png'),
                rgb_image,
                frame_annotations,
            )
        frame_rows.append(frame_record)

        if (frame_index + 1) % 10 == 0:
            print(
                f'Evaluated {frame_index + 1}/{len(frame_paths)} frames',
                flush=True,
            )

    report = {
        'dataset': str(dataset),
        'output': str(output),
        'backend': args.backend,
        'frame_count': len(frame_paths),
        'warnings': report_warnings,
        'dataset_support': {
            'format_version': int(manifest.get('format_version', 1)),
            'instance_id_encoding': instance_encoding or 'legacy_repaired',
            'map': manifest.get('map', ''),
            'missing_vehicle_classes': manifest.get(
                'missing_vehicle_classes', []),
            'capture_valid_for_vehicle_benchmark': bool(
                manifest.get('valid_for_vehicle_benchmark', False)),
            'valid_for_formal_vehicle_evaluation': bool(
                not legacy_instance_encoding
                and manifest.get('valid_for_vehicle_benchmark', False)
            ),
            'spawned_vehicle_class_counts': (
                manifest.get('spawned_vehicle_class_counts', {})),
            'captured_vehicle_observations_by_class': (
                manifest.get(
                    'visible_vehicle_observations_by_class', {})),
            'usable_vehicle_observations_by_class': (
                usable_vehicle_observations),
        },
        'models': {
            'clip': (
                {
                    'model': args.clip_model,
                    'resolved_model': clip.resolved_model_name,
                    'pretrained': args.clip_pretrained,
                    'grid_rows': args.grid_rows,
                    'grid_cols': args.grid_cols,
                    'vocab': args.clip_vocab,
                } if clip is not None else None
            ),
            'segformer': (
                {
                    'model': args.segformer_model,
                    'requested_revision': args.segformer_revision or None,
                    'resolved_revision': getattr(
                        segformer.model.config, '_commit_hash', None),
                    'confidence_threshold': args.segformer_confidence,
                    'legacy_probability_mapping': (
                        'resize_raw_logits_then_argmax_then_project_lookup'),
                    'min_instance_semantic_fraction': (
                        args.min_instance_semantic_fraction),
                } if segformer is not None else None
            ),
        },
    }
    if clip is not None:
        for values in clip_query_results.values():
            values['accuracy'] = (
                values['hits'] / max(values['attempts'], 1))
        report['clip'] = {
            'grid_classification': _counter_metrics(clip_grid_counter),
            'frame_presence': _counter_metrics(clip_frame_counter),
            'query_localization': clip_query_results,
        }
    if segformer is not None:
        semantic_metrics = metrics_from_confusion(
            coarse_confusion, COARSE_LABELS)
        foreground_ious = [
            semantic_metrics[label]['iou']
            for label in ('road', 'person', 'vehicle')
            if semantic_metrics[label]['support_pixels'] > 0
        ]
        report['segformer'] = {
            'coarse_labels': list(COARSE_LABELS),
            'confusion_matrix': coarse_confusion.tolist(),
            'semantic_metrics': semantic_metrics,
            'foreground_mean_iou': (
                float(np.mean(foreground_ious))
                if foreground_ious else 0.0),
        }

    if instance_rows:
        metric_keys = ['color_correct']
        if segformer is not None:
            metric_keys.extend([
                'segformer_detected',
                'segformer_class_correct',
                'segformer_attribute_correct',
            ])
        if clip is not None:
            metric_keys.extend([
                'clip_class_correct',
                'clip_color_correct',
                'clip_attribute_correct',
                'clip_object_localized',
                'clip_attribute_localized',
            ])
        report['vehicle_instances'] = {
            'count': len(instance_rows),
            'color_accuracy': float(np.mean([
                row['color_correct'] for row in instance_rows
            ])),
            'by_class': _grouped_accuracy(
                instance_rows, 'true_class', metric_keys),
            'by_color': _grouped_accuracy(
                instance_rows, 'true_color', metric_keys),
        }
        if segformer is not None:
            report['vehicle_instances'].update({
                'segformer_class_accuracy': float(np.mean([
                    row['segformer_class_correct'] for row in instance_rows
                ])),
                'segformer_detection_rate': float(np.mean([
                    row['segformer_detected'] for row in instance_rows
                ])),
                'segformer_attribute_accuracy': float(np.mean([
                    row['segformer_attribute_correct'] for row in instance_rows
                ])),
            })
        if clip is not None:
            report['vehicle_instances'].update({
                'clip_class_accuracy': float(np.mean([
                    row['clip_class_correct'] for row in instance_rows
                ])),
                'clip_color_accuracy': float(np.mean([
                    row['clip_color_correct'] for row in instance_rows
                ])),
                'clip_attribute_accuracy': float(np.mean([
                    row['clip_attribute_correct'] for row in instance_rows
                ])),
                'clip_object_localization_accuracy': float(np.mean([
                    row['clip_object_localized'] for row in instance_rows
                ])),
                'clip_attribute_localization_accuracy': float(np.mean([
                    row['clip_attribute_localized']
                    for row in instance_rows
                ])),
            })

    if person_rows:
        person_report = {'count': len(person_rows)}
        if segformer is not None:
            person_report['segformer_detection_rate'] = float(np.mean([
                row['segformer_detected'] for row in person_rows
            ]))
            person_report['segformer_mean_person_fraction'] = float(np.mean([
                row['segformer_person_fraction'] for row in person_rows
            ]))
        if clip is not None:
            person_report['clip_localization_accuracy'] = float(np.mean([
                row['clip_person_localized'] for row in person_rows
            ]))
        report['person_instances'] = person_report

    unsupported_classes = [
        name for name, count in usable_vehicle_observations.items()
        if count == 0
    ]
    if unsupported_classes:
        warning = (
            'No usable instance observations for: '
            + ', '.join(unsupported_classes)
            + '. Their accuracy values must not be interpreted.'
        )
        report['warnings'].append(warning)

    _write_json(output / 'report.json', report)
    _write_csv(output / 'per_frame.csv', frame_rows)
    _write_csv(output / 'per_vehicle_instance.csv', instance_rows)
    _write_csv(output / 'per_person_instance.csv', person_rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Full report: {output / "report.json"}')
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return evaluate(args)
    except Exception as exc:
        print(f'CARLA evaluation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
