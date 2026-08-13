#!/usr/bin/env python3
"""Validate data, fine-tune SegFormer, and inspect checkpoint label support."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from semantic_mapping.runtime.semantic_schema import (
    DEFAULT_CLASSES,
    SEGFORMER_LABEL_ALIASES,
    normalize_label,
)


IGNORE_LABEL = 255
DEFAULT_REQUIRED_CLASSES = (
    'road',
    'car',
    'bicycle',
    'electric_bicycle',
    'motorcycle',
)
DEFAULT_MIN_TARGET_IOU = 0.30
DEFAULT_MIN_ELECTRIC_BICYCLE_IOU = 0.35
DEFAULT_MIN_ROAD_IOU = 0.50
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
REQUIRED_SPLITS = ('train', 'val')
OPTIONAL_SPLITS = ('calibration', 'test')


class DatasetValidationError(ValueError):
    """Raised when a fine-tuning dataset cannot support formal training."""


def split_model_label(label):
    """Normalize one model label and its comma/slash-separated synonyms."""
    terms = {normalize_label(label)}
    terms.update(
        normalize_label(part)
        for part in re.split(r'[,;/|]+', str(label))
    )
    return {term for term in terms if term}


def build_classifier_copy_plan(id2label, project_classes=DEFAULT_CLASSES):
    """Map each project classifier row to reusable source classifier rows."""
    normalized_aliases = {
        project_id: {normalize_label(alias) for alias in aliases}
        for project_id, aliases in SEGFORMER_LABEL_ALIASES.items()
    }
    unknown_index = len(project_classes) - 1
    plan = {index: [] for index in range(len(project_classes))}
    for source_id, source_label in id2label.items():
        terms = split_model_label(source_label)
        target_id = unknown_index
        for project_id, aliases in normalized_aliases.items():
            if terms.intersection(aliases):
                target_id = int(project_id)
                break
        plan[target_id].append(int(source_id))

    electric_id = project_classes.index('electric_bicycle')
    if not plan[electric_id]:
        bicycle_id = project_classes.index('bicycle')
        motorcycle_id = project_classes.index('motorcycle')
        plan[electric_id] = sorted(set(
            plan[bicycle_id] + plan[motorcycle_id]
        ))
    return plan


def supported_project_classes(id2label, project_classes=DEFAULT_CLASSES):
    """Return project classes directly represented in checkpoint labels."""
    plan = build_classifier_copy_plan(id2label, project_classes)
    electric_id = project_classes.index('electric_bicycle')
    direct_electric_terms = {
        normalize_label(alias)
        for alias in SEGFORMER_LABEL_ALIASES[electric_id]
    }
    direct_electric = any(
        split_model_label(label).intersection(direct_electric_terms)
        for label in id2label.values()
    )
    result = []
    for class_id, class_name in enumerate(project_classes[:-1]):
        if class_id == electric_id and not direct_electric:
            continue
        if plan[class_id]:
            result.append(class_name)
    return tuple(result)


def _files_by_stem(directory, suffixes):
    files = {}
    if not directory.is_dir():
        return files
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise DatasetValidationError(
                f'Duplicate stem {path.stem!r} in {directory}.')
        files[path.stem] = path
    return files


def discover_dataset_pairs(dataset_root, split):
    """Return strictly paired image/mask paths for one dataset split."""
    split_root = Path(dataset_root).expanduser().resolve() / split
    images = _files_by_stem(split_root / 'images', IMAGE_SUFFIXES)
    masks = _files_by_stem(split_root / 'masks', ('.png',))
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise DatasetValidationError(
            f'{split} image/mask pairing failed: '
            f'missing_masks={missing_masks[:10]}, '
            f'missing_images={missing_images[:10]}.')
    if not images:
        raise DatasetValidationError(
            f'{split} contains no paired samples under '
            f'{split_root / "images"} and {split_root / "masks"}.')
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def _optional_split_present(dataset_root, split):
    """Return whether an optional split contains any image or mask file."""
    split_root = Path(dataset_root).expanduser().resolve() / split
    images = _files_by_stem(split_root / 'images', IMAGE_SUFFIXES)
    masks = _files_by_stem(split_root / 'masks', ('.png',))
    return bool(images or masks)


def validate_segformer_dataset(
    dataset_root,
    required_classes=DEFAULT_REQUIRED_CLASSES,
):
    """Validate masks, dimensions, pairing, and required class coverage."""
    class_to_id = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
    unknown_required = sorted(set(required_classes) - set(class_to_id))
    if unknown_required:
        raise DatasetValidationError(
            f'Unknown required classes: {unknown_required}.')

    report = {
        'dataset': str(Path(dataset_root).expanduser().resolve()),
        'classes': list(DEFAULT_CLASSES),
        'ignore_label': IGNORE_LABEL,
        'required_classes': list(required_classes),
        'splits': {},
        'valid': False,
    }
    allowed_ids = set(range(len(DEFAULT_CLASSES))) | {IGNORE_LABEL}
    present_optional_splits = [
        split
        for split in OPTIONAL_SPLITS
        if _optional_split_present(dataset_root, split)
    ]
    evaluated_splits = REQUIRED_SPLITS + tuple(present_optional_splits)
    for split in evaluated_splits:
        pairs = discover_dataset_pairs(dataset_root, split)
        pixel_counts = np.zeros(len(DEFAULT_CLASSES), dtype=np.int64)
        ignored_pixels = 0
        for image_path, mask_path in pairs:
            with Image.open(image_path) as image:
                image_size = image.size
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)
                mask_size = mask_image.size
            if mask.ndim != 2:
                raise DatasetValidationError(
                    f'Mask must be single-channel: {mask_path} has '
                    f'shape {mask.shape}.')
            if image_size != mask_size:
                raise DatasetValidationError(
                    f'Image/mask size mismatch for {image_path.name}: '
                    f'{image_size} != {mask_size}.')
            values, counts = np.unique(
                mask.astype(np.int64), return_counts=True)
            invalid = sorted(set(values.tolist()) - allowed_ids)
            if invalid:
                raise DatasetValidationError(
                    f'{mask_path} contains invalid label ids {invalid}; '
                    f'allowed ids are 0..{len(DEFAULT_CLASSES) - 1} and 255.')
            for value, count in zip(values, counts):
                if int(value) == IGNORE_LABEL:
                    ignored_pixels += int(count)
                else:
                    pixel_counts[int(value)] += int(count)

        missing = [
            class_name
            for class_name in required_classes
            if pixel_counts[class_to_id[class_name]] == 0
        ]
        if missing:
            raise DatasetValidationError(
                f'{split} has no labeled pixels for required classes: '
                + ', '.join(missing))
        report['splits'][split] = {
            'sample_count': len(pairs),
            'class_pixels': {
                class_name: int(pixel_counts[class_id])
                for class_id, class_name in enumerate(DEFAULT_CLASSES)
            },
            'ignored_pixels': int(ignored_pixels),
        }
    has_calibration = 'calibration' in report['splits']
    has_test = 'test' in report['splits']
    report['split_protocol'] = {
        'training_split': 'train',
        'model_selection_split': 'val',
        'calibration_split': 'calibration' if has_calibration else None,
        'final_evaluation_split': 'test' if has_test else 'val',
        'independent_test_present': has_test,
        'formal_evaluation_ready': has_test,
    }
    report['warnings'] = []
    if not has_test:
        report['warnings'].append(
            'NO INDEPENDENT TEST SPLIT: final thresholds will be reported in '
            'legacy compatibility mode and must not be cited as formal test '
            'performance.')
    if not has_calibration:
        report['warnings'].append(
            'No calibration split: no temperature or decision-threshold '
            'calibration may be fitted in this run.')
    report['valid'] = True
    return report


def initialize_dataset_layout(dataset_root):
    """Create a non-destructive paired-image segmentation dataset layout."""
    root = Path(dataset_root).expanduser().resolve()
    for split in REQUIRED_SPLITS + OPTIONAL_SPLITS:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'masks').mkdir(parents=True, exist_ok=True)
    labels_path = root / 'labels.json'
    labels = {
        'classes': {
            str(index): class_name
            for index, class_name in enumerate(DEFAULT_CLASSES)
        },
        'ignore_label': IGNORE_LABEL,
        'required_classes': list(DEFAULT_REQUIRED_CLASSES),
        'mask_format': 'single-channel PNG with integer class ids',
        'split_roles': {
            'train': 'parameter optimization',
            'val': 'model selection only',
            'calibration': 'optional calibration only',
            'test': 'optional independent final evaluation',
        },
    }
    if not labels_path.exists():
        labels_path.write_text(
            json.dumps(labels, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    return root


def checkpoint_integrity(checkpoint):
    """Return stable per-file and aggregate SHA-256 hashes for a checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f'Checkpoint directory not found: {checkpoint}')
    file_hashes = {}
    aggregate = hashlib.sha256()
    paths = sorted(path for path in checkpoint.rglob('*') if path.is_file())
    for path in paths:
        relative_path = path.relative_to(checkpoint).as_posix()
        file_digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                file_digest.update(block)
        digest = file_digest.hexdigest()
        file_hashes[relative_path] = digest
        aggregate.update(relative_path.encode('utf-8'))
        aggregate.update(b'\0')
        aggregate.update(bytes.fromhex(digest))
    if not file_hashes:
        raise DatasetValidationError(
            f'Checkpoint contains no files to hash: {checkpoint}')
    return {
        'algorithm': 'sha256',
        'aggregate_sha256': aggregate.hexdigest(),
        'files': file_hashes,
    }


def _pad_to_minimum(image, mask, size):
    pad_width = max(0, size - image.width)
    pad_height = max(0, size - image.height)
    if pad_width == 0 and pad_height == 0:
        return image, mask
    left = pad_width // 2
    top = pad_height // 2
    border = (
        left,
        top,
        pad_width - left,
        pad_height - top,
    )
    return (
        ImageOps.expand(image, border=border, fill=(0, 0, 0)),
        ImageOps.expand(mask, border=border, fill=IGNORE_LABEL),
    )


def paired_training_crop(image, mask, size, rng):
    """Apply paired scale, flip, padding, and random crop augmentation."""
    shortest = max(1, min(image.size))
    target_shortest = rng.uniform(0.8, 1.4) * size
    scale = target_shortest / shortest
    resized_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    image = image.resize(resized_size, Image.Resampling.BILINEAR)
    mask = mask.resize(resized_size, Image.Resampling.NEAREST)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image, mask = _pad_to_minimum(image, mask, size)
    left = rng.randint(0, image.width - size)
    top = rng.randint(0, image.height - size)
    crop_box = (left, top, left + size, top + size)
    return image.crop(crop_box), mask.crop(crop_box)


def paired_validation_resize(image, mask, size):
    """Letterbox one validation pair without changing its aspect ratio."""
    scale = min(size / image.width, size / image.height)
    resized_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    image = image.resize(resized_size, Image.Resampling.BILINEAR)
    mask = mask.resize(resized_size, Image.Resampling.NEAREST)
    return _pad_to_minimum(image, mask, size)


class SegformerFolderDataset:
    """Small dependency-free paired image/mask dataset for SegFormer."""

    def __init__(self, pairs, processor, image_size, training):
        """Store sample pairs and deterministic preprocessing settings."""
        self.pairs = list(pairs)
        self.processor = processor
        self.image_size = int(image_size)
        self.training = bool(training)

    def __len__(self):
        """Return the number of paired samples."""
        return len(self.pairs)

    def __getitem__(self, index):
        """Load and preprocess one paired RGB image and class mask."""
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert('RGB')
        with Image.open(mask_path) as source_mask:
            mask = source_mask.copy()
        if self.training:
            image, mask = paired_training_crop(
                image, mask, self.image_size, random)
        else:
            image, mask = paired_validation_resize(
                image, mask, self.image_size)
        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            do_resize=False,
            do_reduce_labels=False,
            return_tensors='pt',
        )
        return {
            'pixel_values': encoded['pixel_values'][0],
            'labels': encoded['labels'][0].long(),
        }


def confusion_metrics(confusion, class_names=DEFAULT_CLASSES):
    """Convert a pixel confusion matrix into per-class IoU and mean IoU."""
    confusion = np.asarray(confusion, dtype=np.float64)
    true_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    union = true_pixels + predicted_pixels - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )
    supported = true_pixels > 0
    return {
        'mean_iou': (
            float(np.mean(iou[supported])) if np.any(supported) else 0.0),
        'per_class_iou': {
            class_name: float(iou[index])
            for index, class_name in enumerate(class_names)
            if supported[index]
        },
        'support_pixels': {
            class_name: int(true_pixels[index])
            for index, class_name in enumerate(class_names)
            if supported[index]
        },
    }


def _copy_project_classifier_rows(source_model, target_model):
    source_classifier = source_model.decode_head.classifier
    target_classifier = target_model.decode_head.classifier
    id2label = {
        int(class_id): str(label)
        for class_id, label in source_model.config.id2label.items()
    }
    plan = build_classifier_copy_plan(id2label)
    copied = {}
    import torch

    with torch.no_grad():
        for target_id, source_ids in plan.items():
            valid_source_ids = [
                source_id
                for source_id in source_ids
                if 0 <= source_id < source_classifier.weight.shape[0]
            ]
            if not valid_source_ids:
                continue
            target_classifier.weight[target_id].copy_(
                source_classifier.weight[valid_source_ids].mean(dim=0))
            if (
                target_classifier.bias is not None
                and source_classifier.bias is not None
            ):
                target_classifier.bias[target_id].copy_(
                    source_classifier.bias[valid_source_ids].mean(dim=0))
            copied[DEFAULT_CLASSES[target_id]] = valid_source_ids
    return copied


def _create_project_model(base_model):
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    source_model = SegformerForSemanticSegmentation.from_pretrained(base_model)
    id2label = {
        index: class_name
        for index, class_name in enumerate(DEFAULT_CLASSES)
    }
    label2id = {class_name: index for index, class_name in id2label.items()}
    target_model = SegformerForSemanticSegmentation.from_pretrained(
        base_model,
        num_labels=len(DEFAULT_CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    copied = _copy_project_classifier_rows(source_model, target_model)
    processor = SegformerImageProcessor.from_pretrained(
        base_model,
        do_reduce_labels=False,
    )
    del source_model
    return target_model, processor, copied


def _update_confusion(confusion, predictions, labels):
    valid = labels != IGNORE_LABEL
    encoded = labels[valid] * len(DEFAULT_CLASSES) + predictions[valid]
    counts = np.bincount(
        encoded.astype(np.int64),
        minlength=len(DEFAULT_CLASSES) ** 2,
    )
    confusion += counts.reshape(len(DEFAULT_CLASSES), len(DEFAULT_CLASSES))


def _evaluate_model(model, loader, device, use_fp16):
    import torch
    import torch.nn.functional as functional

    confusion = np.zeros(
        (len(DEFAULT_CLASSES), len(DEFAULT_CLASSES)), dtype=np.int64)
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(pixel_values=pixel_values, labels=labels)
            logits = functional.interpolate(
                outputs.logits.float(),
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            predictions = logits.argmax(dim=1)
            _update_confusion(
                confusion,
                predictions.cpu().numpy(),
                labels.cpu().numpy(),
            )
            total_loss += float(outputs.loss.detach().cpu())
            batch_count += 1
    metrics = confusion_metrics(confusion)
    metrics['loss'] = total_loss / max(batch_count, 1)
    return metrics


def _acceptance_metrics(training_report):
    """Choose final metrics, preferring an explicitly recorded test result."""
    for key in ('acceptance_metrics', 'test_metrics', 'best_metrics'):
        metrics = training_report.get(key)
        if isinstance(metrics, dict):
            return key, metrics
    return None, {}


def _threshold_failures(
    metrics,
    min_target_iou,
    min_electric_bicycle_iou,
    min_road_iou,
):
    """Return per-class failures for one immutable evaluation result."""
    class_ious = metrics.get('per_class_iou', {})
    failures = {}
    for class_name in ('car', 'bicycle', 'motorcycle'):
        value = float(class_ious.get(class_name, 0.0))
        if value < min_target_iou:
            failures[class_name] = value
    electric_iou = float(class_ious.get('electric_bicycle', 0.0))
    electric_minimum = max(
        float(min_target_iou), float(min_electric_bicycle_iou))
    if electric_iou < electric_minimum:
        failures['electric_bicycle'] = electric_iou
    road_iou = float(class_ious.get('road', 0.0))
    if road_iou < min_road_iou:
        failures['road'] = road_iou
    return failures


def train_segformer(args):
    """Fine-tune 13 classes and save the best validation checkpoint."""
    import torch
    from torch.utils.data import DataLoader

    report = validate_segformer_dataset(
        args.dataset, args.required_classes)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f'Output directory is not empty: {output}. Use --overwrite only '
            'when replacing a known training run.')
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested_device = args.device
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA requested but torch.cuda.is_available() is false.')
    device = torch.device(requested_device)
    use_fp16 = bool(args.fp16 and device.type == 'cuda')

    model, processor, copied_rows = _create_project_model(args.base_model)
    model.to(device)
    train_pairs = discover_dataset_pairs(args.dataset, 'train')
    val_pairs = discover_dataset_pairs(args.dataset, 'val')
    train_dataset = SegformerFolderDataset(
        train_pairs, processor, args.image_size, training=True)
    val_dataset = SegformerFolderDataset(
        val_pairs, processor, args.image_size, training=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )
    optional_loaders = {}
    for split in OPTIONAL_SPLITS:
        if split not in report['splits']:
            continue
        split_dataset = SegformerFolderDataset(
            discover_dataset_pairs(args.dataset, split),
            processor,
            args.image_size,
            training=False,
        )
        optional_loaders[split] = DataLoader(
            split_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == 'cuda',
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    training_report = {
        'dataset_validation': report,
        'base_model': args.base_model,
        'classes': list(DEFAULT_CLASSES),
        'copied_classifier_rows': copied_rows,
        'device': str(device),
        'fp16': use_fp16,
        'epochs': [],
        'best_epoch': None,
        'best_selection_score': -1.0,
        'best_electric_bicycle_iou': -1.0,
        'best_metrics': None,
        'model_selection_split': 'val',
        'calibration': {
            'split': (
                'calibration' if 'calibration' in optional_loaders else None),
            'fitted_parameters': [],
            'note': (
                'This trainer does not fit temperature or decision '
                'thresholds. '
                'Any future calibration must use only the calibration split.'
            ),
        },
        'test_metrics': None,
        'acceptance_metrics': None,
        'acceptance_metrics_split': None,
        'evaluation_mode': None,
        'valid_for_formal_evaluation': False,
        'formal_acceptance_passed': False,
        'warnings': list(report.get('warnings', [])),
        'accepted': False,
    }
    best_dir = output / 'best'
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(outputs.loss.detach().cpu())

        metrics = _evaluate_model(model, val_loader, device, use_fp16)
        metrics['epoch'] = epoch
        metrics['train_loss'] = running_loss / max(len(train_loader), 1)
        selection_classes = (
            'road', 'car', 'bicycle', 'electric_bicycle', 'motorcycle')
        selection_ious = [
            metrics['per_class_iou'].get(class_name, 0.0)
            for class_name in selection_classes
        ]
        metrics['selection_mean_iou'] = float(np.mean(selection_ious))
        training_report['epochs'].append(metrics)
        electric_iou = metrics['per_class_iou'].get(
            'electric_bicycle', 0.0)
        print(
            f'Epoch {epoch}/{args.epochs}: '
            f'train_loss={metrics["train_loss"]:.4f}, '
            f'val_loss={metrics["loss"]:.4f}, '
            f'mIoU={metrics["mean_iou"]:.3f}, '
            f'electric_bicycle_IoU={electric_iou:.3f}',
            flush=True,
        )
        selection_score = metrics['selection_mean_iou']
        if selection_score > training_report['best_selection_score']:
            training_report['best_selection_score'] = selection_score
            training_report['best_electric_bicycle_iou'] = electric_iou
            training_report['best_epoch'] = epoch
            training_report['best_metrics'] = metrics
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

    if optional_loaders:
        from transformers import SegformerForSemanticSegmentation

        evaluation_model = SegformerForSemanticSegmentation.from_pretrained(
            best_dir)
        evaluation_model.to(device)
        if 'calibration' in optional_loaders:
            training_report['calibration']['metrics'] = _evaluate_model(
                evaluation_model,
                optional_loaders['calibration'],
                device,
                use_fp16,
            )
        if 'test' in optional_loaders:
            training_report['test_metrics'] = _evaluate_model(
                evaluation_model,
                optional_loaders['test'],
                device,
                use_fp16,
            )
        del evaluation_model

    has_test = training_report['test_metrics'] is not None
    acceptance_metrics = (
        training_report['test_metrics']
        if has_test
        else training_report['best_metrics'] or {}
    )
    training_report['acceptance_metrics'] = acceptance_metrics
    training_report['acceptance_metrics_split'] = 'test' if has_test else 'val'
    training_report['evaluation_mode'] = (
        'independent_test' if has_test else 'legacy_val_compatibility')
    training_report['valid_for_formal_evaluation'] = has_test
    if not has_test:
        warning = (
            'NON-FORMAL RESULT: acceptance thresholds were evaluated on the '
            'model-selection val split because no independent test split was '
            'provided. Do not report these numbers as test performance.'
        )
        if warning not in training_report['warnings']:
            training_report['warnings'].append(warning)

    threshold_failures = _threshold_failures(
        acceptance_metrics,
        args.min_target_iou,
        args.min_electric_bicycle_iou,
        args.min_road_iou,
    )
    class_ious = acceptance_metrics.get('per_class_iou', {})
    electric_iou = float(class_ious.get('electric_bicycle', 0.0))
    road_iou = float(class_ious.get('road', 0.0))
    training_report['accepted'] = not threshold_failures
    training_report['formal_acceptance_passed'] = bool(
        training_report['accepted']
        and training_report['valid_for_formal_evaluation'])
    training_report['acceptance_thresholds'] = {
        'min_target_iou': args.min_target_iou,
        'min_electric_bicycle_iou': args.min_electric_bicycle_iou,
        'min_road_iou': args.min_road_iou,
    }
    training_report['acceptance_failures'] = {
        'classes_below_minimum': threshold_failures,
        'electric_bicycle_iou': electric_iou,
        'road_iou': road_iou,
    }
    integrity = checkpoint_integrity(best_dir)
    training_report['checkpoint_integrity'] = integrity
    training_report['checkpoint_sha256'] = integrity['aggregate_sha256']
    (output / 'training_report.json').write_text(
        json.dumps(training_report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'Best checkpoint: {best_dir}')
    print(
        'Acceptance: '
        + ('PASS' if training_report['accepted'] else 'FAIL')
        + f' [{training_report["evaluation_mode"]}]'
        + f' (road IoU={road_iou:.3f}, '
        f'electric_bicycle IoU={electric_iou:.3f}, '
        f'failures={threshold_failures})')
    return 0 if training_report['accepted'] else 2


def checkpoint_report(
    model_id,
    local_files_only=False,
    require_training_report=True,
    min_target_iou=DEFAULT_MIN_TARGET_IOU,
    min_electric_bicycle_iou=DEFAULT_MIN_ELECTRIC_BICYCLE_IOU,
    min_road_iou=DEFAULT_MIN_ROAD_IOU,
):
    """Inspect checkpoint id2label without loading model weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only)
    id2label = {
        int(class_id): str(label)
        for class_id, label in config.id2label.items()
    }
    supported = supported_project_classes(id2label)
    required = ('car', 'bicycle', 'electric_bicycle', 'motorcycle')
    schema_valid = all(
        class_name in supported for class_name in required)
    training_report_path, training_report = load_training_report(model_id)
    metrics_accepted, metric_failures = training_report_meets_thresholds(
        training_report,
        min_target_iou=min_target_iou,
        min_electric_bicycle_iou=min_electric_bicycle_iou,
        min_road_iou=min_road_iou,
    )
    model_path = Path(str(model_id)).expanduser()
    current_integrity = None
    if model_path.is_dir():
        current_integrity = checkpoint_integrity(model_path)
    recorded_hash = (
        training_report.get('checkpoint_sha256')
        if isinstance(training_report, dict)
        else None
    )
    current_hash = (
        current_integrity['aggregate_sha256']
        if current_integrity is not None
        else None
    )
    hash_verified = bool(recorded_hash and current_hash == recorded_hash)
    formal_protocol = bool(
        isinstance(training_report, dict)
        and training_report.get('valid_for_formal_evaluation', False)
        and training_report.get('acceptance_metrics_split') == 'test'
    )
    warnings = []
    if training_report is not None and not formal_protocol:
        warnings.append(
            'NON-FORMAL TRAINING REPORT: no independent test-based acceptance '
            'is recorded.')
    if training_report is not None and not recorded_hash:
        warnings.append(
            'LEGACY TRAINING REPORT: checkpoint SHA-256 is absent, so metric '
            'provenance cannot be cryptographically bound to this checkpoint.')
    elif recorded_hash and not hash_verified:
        warnings.append(
            'CHECKPOINT HASH MISMATCH: the checkpoint files differ from the '
            'files evaluated by the training report.')
    valid_for_formal_evaluation = bool(
        schema_valid
        and metrics_accepted
        and formal_protocol
        and hash_verified
    )
    return {
        'model': str(model_id),
        'num_labels': int(config.num_labels),
        'id2label': id2label,
        'supported_navigation_classes': list(supported),
        'required_target_classes': list(required),
        'missing_target_classes': [
            class_name
            for class_name in required
            if class_name not in supported
        ],
        'schema_support_valid': schema_valid,
        'training_report': (
            str(training_report_path) if training_report_path else None),
        'metrics_accepted': metrics_accepted,
        'metric_failures': metric_failures,
        'acceptance_metrics_split': (
            training_report.get('acceptance_metrics_split')
            if isinstance(training_report, dict)
            else None
        ),
        'checkpoint_sha256_recorded': recorded_hash,
        'checkpoint_sha256_current': current_hash,
        'checkpoint_hash_verified': hash_verified,
        'valid_for_formal_evaluation': valid_for_formal_evaluation,
        'warnings': warnings,
        'runtime_acceptance_thresholds': {
            'min_target_iou': float(min_target_iou),
            'min_electric_bicycle_iou': float(
                min_electric_bicycle_iou),
            'min_road_iou': float(min_road_iou),
        },
        'training_report_required': bool(require_training_report),
        'valid_for_requested_target_distinction': bool(
            schema_valid
            and (metrics_accepted or not require_training_report)),
    }


def load_training_report(model_id):
    """Load the nearest local training report for a checkpoint, if present."""
    model_path = Path(str(model_id)).expanduser()
    if not model_path.exists():
        return None, None
    candidates = (
        model_path / 'training_report.json',
        model_path.parent / 'training_report.json',
    )
    for candidate in candidates:
        if candidate.is_file():
            return (
                candidate.resolve(),
                json.loads(candidate.read_text(encoding='utf-8')),
            )
    return None, None


def training_report_meets_thresholds(
    training_report,
    min_target_iou=DEFAULT_MIN_TARGET_IOU,
    min_electric_bicycle_iou=DEFAULT_MIN_ELECTRIC_BICYCLE_IOU,
    min_road_iou=DEFAULT_MIN_ROAD_IOU,
):
    """Independently verify saved metrics against runtime safety thresholds."""
    if training_report is None:
        return False, {'training_report': 'missing'}
    if not training_report.get('accepted', False):
        return False, {'training_report': 'training command did not accept it'}
    _, metrics = _acceptance_metrics(training_report)
    failures = _threshold_failures(
        metrics,
        min_target_iou,
        min_electric_bicycle_iou,
        min_road_iou,
    )
    return not failures, failures


def _required_classes(value):
    classes = tuple(part.strip() for part in value.split(',') if part.strip())
    if not classes:
        raise argparse.ArgumentTypeError('required class list cannot be empty')
    return classes


def dataset_main(argv=None):
    """Run the dataset initialization or validation command."""
    parser = argparse.ArgumentParser(
        description='Initialize or validate a paired SegFormer dataset.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--init', action='store_true')
    parser.add_argument(
        '--required-classes',
        type=_required_classes,
        default=DEFAULT_REQUIRED_CLASSES,
        help='Comma-separated required classes for both train and val.',
    )
    args = parser.parse_args(argv)
    try:
        if args.init:
            root = initialize_dataset_layout(args.dataset)
            print(f'Dataset layout ready: {root}')
            return 0
        report = validate_segformer_dataset(
            args.dataset, args.required_classes)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f'SegFormer dataset validation failed: {exc}')
        return 1


def checkpoint_main(argv=None):
    """Run the checkpoint label-support inspection command."""
    parser = argparse.ArgumentParser(
        description='Report whether a SegFormer checkpoint supports targets.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument(
        '--schema-only',
        action='store_true',
        help='Check id2label only; never treat this as navigation acceptance.',
    )
    parser.add_argument(
        '--min-target-iou', type=float, default=DEFAULT_MIN_TARGET_IOU)
    parser.add_argument(
        '--min-electric-bicycle-iou',
        type=float,
        default=DEFAULT_MIN_ELECTRIC_BICYCLE_IOU,
    )
    parser.add_argument(
        '--min-road-iou', type=float, default=DEFAULT_MIN_ROAD_IOU)
    args = parser.parse_args(argv)
    try:
        report = checkpoint_report(
            args.model,
            args.local_files_only,
            require_training_report=not args.schema_only,
            min_target_iou=args.min_target_iou,
            min_electric_bicycle_iou=args.min_electric_bicycle_iou,
            min_road_iou=args.min_road_iou,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['valid_for_requested_target_distinction'] else 2
    except Exception as exc:
        print(f'SegFormer checkpoint validation failed: {exc}')
        return 1


def finetune_main(argv=None):
    """Run the guarded SegFormer fine-tuning command."""
    parser = argparse.ArgumentParser(
        description='Fine-tune a 13-class SegFormer for semantic navigation.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--base-model',
        default='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    )
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--min-target-iou', type=float, default=DEFAULT_MIN_TARGET_IOU)
    parser.add_argument(
        '--min-electric-bicycle-iou',
        type=float,
        default=DEFAULT_MIN_ELECTRIC_BICYCLE_IOU,
    )
    parser.add_argument(
        '--min-road-iou', type=float, default=DEFAULT_MIN_ROAD_IOU)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--required-classes',
        type=_required_classes,
        default=DEFAULT_REQUIRED_CLASSES,
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        parser.error('epochs, batch-size and image-size must be positive')
    if args.gradient_accumulation_steps <= 0:
        parser.error('gradient-accumulation-steps must be positive')
    try:
        return train_segformer(args)
    except Exception as exc:
        print(f'SegFormer fine-tuning failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(finetune_main())
