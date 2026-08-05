#!/usr/bin/env python3
import re
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from semantic_mapping.semantic_schema import (
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    SEGFORMER_LABEL_ALIASES,
    convert_image_to_rgb,
    normalize_label,
)

PROJECT_CLASSES = list(DEFAULT_CLASSES)
PROJECT_COLORS = np.asarray(DEFAULT_CLASS_COLORS, dtype=np.uint8)
LABEL_ALIASES = SEGFORMER_LABEL_ALIASES


def split_model_label(label):
    """Return a model label and each comma/slash-separated synonym."""
    terms = {normalize_label(label)}
    terms.update(
        normalize_label(part)
        for part in re.split(r'[,;/|]+', str(label))
    )
    return {term for term in terms if term}


def build_project_lookup(id2label):
    """Map a model label set into the compact project classes."""
    class_count = max(int(class_id) for class_id in id2label) + 1
    unknown_index = len(PROJECT_CLASSES) - 1
    lookup = np.full(class_count, unknown_index, dtype=np.uint8)
    for class_id, original_label in id2label.items():
        label_terms = split_model_label(original_label)
        for project_id, aliases in LABEL_ALIASES.items():
            normalized_aliases = {normalize_label(alias) for alias in aliases}
            if label_terms.intersection(normalized_aliases):
                lookup[int(class_id)] = project_id
                break
    return lookup


def find_relevant_model_labels(id2label, watched_labels):
    """Find original model ids whose label terms match watched labels."""
    result = {}
    for watched_label in watched_labels:
        watched_term = normalize_label(watched_label)
        watched_terms = {watched_term}
        if watched_term in PROJECT_CLASSES:
            project_id = PROJECT_CLASSES.index(watched_term)
            watched_terms.update(
                normalize_label(alias)
                for alias in LABEL_ALIASES.get(project_id, ())
            )
        result[watched_label] = sorted(
            (
                (int(class_id), str(original_label))
                for class_id, original_label in id2label.items()
                if watched_terms.intersection(split_model_label(original_label))
            ),
            key=lambda item: item[0],
        )
    return result


def _normalized_roi_slices(image_shape, normalized_roi):
    if len(normalized_roi) != 4:
        raise ValueError(
            'debug_roi_xywh_normalized must contain [x, y, width, height].')
    x, y, width, height = (float(value) for value in normalized_roi)
    if width <= 0.0 or height <= 0.0:
        return None

    image_height, image_width = image_shape
    x0 = int(np.floor(np.clip(x, 0.0, 1.0) * image_width))
    y0 = int(np.floor(np.clip(y, 0.0, 1.0) * image_height))
    x1 = int(np.ceil(np.clip(x + width, 0.0, 1.0) * image_width))
    y1 = int(np.ceil(np.clip(y + height, 0.0, 1.0) * image_height))
    if x1 <= x0 or y1 <= y0:
        return None
    return slice(y0, y1), slice(x0, x1), (x0, y0, x1, y1)


def _region_class_statistics(mask, confidence, id2label, top_k, watched_labels):
    pixel_count = int(mask.size)
    class_count = max(
        max((int(class_id) for class_id in id2label), default=-1) + 1,
        int(np.max(mask)) + 1 if pixel_count else 0,
    )
    counts = np.bincount(mask.ravel(), minlength=class_count)

    def make_entry(class_id):
        class_pixels = mask == class_id
        count = int(counts[class_id])
        mean_confidence = (
            float(np.mean(confidence[class_pixels])) if count else 0.0)
        return {
            'id': int(class_id),
            'label': str(id2label.get(int(class_id), f'class_{class_id}')),
            'count': count,
            'fraction': count / max(pixel_count, 1),
            'mean_confidence': mean_confidence,
        }

    ranked_ids = np.argsort(counts)[::-1]
    top = [
        make_entry(int(class_id))
        for class_id in ranked_ids
        if counts[class_id] > 0
    ][:max(1, int(top_k))]

    watched = {}
    relevant_ids = find_relevant_model_labels(id2label, watched_labels)
    for watched_label, matching_labels in relevant_ids.items():
        matching_ids = [class_id for class_id, _ in matching_labels]
        matching_mask = np.isin(mask, matching_ids)
        count = int(np.count_nonzero(matching_mask))
        watched[watched_label] = {
            'ids': matching_ids,
            'count': count,
            'fraction': count / max(pixel_count, 1),
            'mean_confidence': (
                float(np.mean(confidence[matching_mask])) if count else 0.0),
        }
    return {'pixel_count': pixel_count, 'top': top, 'watched': watched}


def raw_prediction_statistics(
    mask,
    confidence,
    id2label,
    top_k=8,
    watched_labels=(),
    normalized_roi=(0.0, 0.0, 0.0, 0.0),
):
    """Summarize raw model predictions globally and in an optional ROI."""
    mask = np.asarray(mask)
    confidence = np.asarray(confidence)
    if mask.ndim != 2 or confidence.shape != mask.shape:
        raise ValueError(
            f'Prediction/confidence shapes must match 2-D masks; '
            f'got {mask.shape} and {confidence.shape}.')

    result = {
        'global': _region_class_statistics(
            mask, confidence, id2label, top_k, watched_labels),
        'roi': None,
        'roi_pixels': None,
    }
    roi_slices = _normalized_roi_slices(mask.shape, normalized_roi)
    if roi_slices is not None:
        rows, columns, pixel_bounds = roi_slices
        result['roi'] = _region_class_statistics(
            mask[rows, columns],
            confidence[rows, columns],
            id2label,
            top_k,
            watched_labels,
        )
        result['roi_pixels'] = pixel_bounds
    return result


def format_raw_statistics(region_statistics):
    """Format one raw prediction region for concise ROS logging."""
    top = ', '.join(
        f'{entry["id"]}:{entry["label"]}='
        f'{entry["fraction"]:.1%}@{entry["mean_confidence"]:.2f}'
        for entry in region_statistics['top']
    )
    watched = ', '.join(
        f'{label}[{",".join(str(class_id) for class_id in values["ids"]) or "-"}]='
        f'{values["fraction"]:.2%}@{values["mean_confidence"]:.2f}'
        for label, values in region_statistics['watched'].items()
    )
    return f'top=[{top}], watched=[{watched}]'


class SegformerNode(Node):
    def __init__(self):
        super().__init__('segformer_node')
        self.declare_parameter(
            'model_id', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('inference_interval_sec', 0.5)
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('class_mask_topic', '/segformer/class_mask')
        self.declare_parameter(
            'raw_class_mask_topic', '/segformer/raw_class_mask')
        self.declare_parameter('confidence_topic', '/segformer/confidence')
        self.declare_parameter('color_mask_topic', '/segformer/color_mask')
        self.declare_parameter('source_image_topic', '/segformer/source_image')
        self.declare_parameter('log_every_n', 10)
        self.declare_parameter('debug_log_raw_predictions', True)
        self.declare_parameter('debug_raw_top_k', 8)
        self.declare_parameter(
            'debug_watch_labels', ['car', 'truck', 'bus', 'road', 'building'])
        self.declare_parameter(
            'debug_roi_xywh_normalized', [0.0, 0.0, 0.0, 0.0])

        self.model_id = self.get_parameter('model_id').value
        requested_device = self.get_parameter('device').value
        self.use_fp16 = bool(self.get_parameter('use_fp16').value)
        self.interval = max(
            0.0, float(self.get_parameter('inference_interval_sec').value))
        self.confidence_threshold = float(np.clip(
            self.get_parameter('confidence_threshold').value, 0.0, 1.0))
        self.image_topic = self.get_parameter('image_topic').value
        self.image_is_compressed = bool(
            self.get_parameter('image_is_compressed').value)
        self.image_encoding = self.get_parameter('image_encoding').value
        self.class_mask_topic = self.get_parameter('class_mask_topic').value
        self.raw_class_mask_topic = self.get_parameter(
            'raw_class_mask_topic').value
        self.confidence_topic = self.get_parameter('confidence_topic').value
        self.color_mask_topic = self.get_parameter('color_mask_topic').value
        self.source_image_topic = self.get_parameter('source_image_topic').value
        self.log_every_n = max(1, int(self.get_parameter('log_every_n').value))
        self.debug_log_raw_predictions = bool(
            self.get_parameter('debug_log_raw_predictions').value)
        self.debug_raw_top_k = max(
            1, int(self.get_parameter('debug_raw_top_k').value))
        self.debug_watch_labels = list(
            self.get_parameter('debug_watch_labels').value)
        self.debug_roi = list(
            self.get_parameter('debug_roi_xywh_normalized').value)

        try:
            import torch
            import torch.nn.functional as functional
            from PIL import Image as PILImage
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                'SegFormer dependencies are missing. Install transformers, '
                'tokenizers, torch and pillow before starting segformer_node.'
            ) from exc

        self.torch = torch
        self.functional = functional
        self.pil_image_type = PILImage
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn(
                'CUDA requested but unavailable; falling back to CPU.')
            requested_device = 'cpu'
        self.device = requested_device

        self.get_logger().info(f'Loading SegFormer model: {self.model_id}')
        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id)
        self.model.eval()
        self.model.to(self.device)
        if self.device == 'cuda' and self.use_fp16:
            self.model.half()

        self.id2label = {
            int(class_id): label
            for class_id, label in self.model.config.id2label.items()
        }
        self.project_lookup = build_project_lookup(self.id2label)
        self.bridge = CvBridge()
        self.last_inference_ns = None
        self.processed_count = 0

        image_type = CompressedImage if self.image_is_compressed else Image
        self.subscription = self.create_subscription(
            image_type,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.class_mask_pub = self.create_publisher(
            Image, self.class_mask_topic, qos_profile_sensor_data)
        self.raw_class_mask_pub = self.create_publisher(
            Image, self.raw_class_mask_topic, qos_profile_sensor_data)
        self.confidence_pub = self.create_publisher(
            Image, self.confidence_topic, qos_profile_sensor_data)
        # The color mask is visualization-only; Reliable matches RViz defaults.
        self.color_mask_pub = self.create_publisher(
            Image, self.color_mask_topic, 1)
        self.source_image_pub = self.create_publisher(
            Image, self.source_image_topic, qos_profile_sensor_data)

        relevant_labels = find_relevant_model_labels(
            self.id2label, self.debug_watch_labels)
        label_summary = '; '.join(
            f'{label}=['
            + ', '.join(f'{class_id}:{name}' for class_id, name in matches)
            + ']'
            for label, matches in relevant_labels.items()
        )
        self.get_logger().info(
            f'SegFormer original id2label (watched): {label_summary}')
        processor_summary = (
            f'do_resize={getattr(self.processor, "do_resize", None)}, '
            f'size={getattr(self.processor, "size", None)}, '
            f'resample={getattr(self.processor, "resample", None)}, '
            f'do_rescale={getattr(self.processor, "do_rescale", None)}, '
            f'do_normalize={getattr(self.processor, "do_normalize", None)}, '
            f'mean={getattr(self.processor, "image_mean", None)}, '
            f'std={getattr(self.processor, "image_std", None)}')
        self.get_logger().info(
            'SegFormer preprocessing: CvBridge '
            f'{self.image_encoding} -> explicit RGB; {processor_summary}; '
            'logits resized to source resolution with bilinear interpolation '
            'before softmax.')
        self.get_logger().info(
            f'SegFormer ready: topic={self.image_topic}, '
            f'compressed={self.image_is_compressed}, device={self.device}, '
            f'confidence_threshold={self.confidence_threshold:.2f}')

    def image_callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_inference_ns is not None
            and now_ns - self.last_inference_ns < int(self.interval * 1e9)
        ):
            return
        self.last_inference_ns = now_ns

        try:
            if self.image_is_compressed:
                bridge_image = self.bridge.compressed_imgmsg_to_cv2(
                    msg, self.image_encoding)
            else:
                bridge_image = self.bridge.imgmsg_to_cv2(
                    msg, self.image_encoding)
            rgb_image = convert_image_to_rgb(
                bridge_image, self.image_encoding)

            height, width = rgb_image.shape[:2]
            inputs = self.processor(
                images=self.pil_image_type.fromarray(rgb_image, mode='RGB'),
                return_tensors='pt',
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            if self.device == 'cuda' and self.use_fp16:
                inputs = {
                    name: value.half() if value.is_floating_point() else value
                    for name, value in inputs.items()
                }

            start = time.perf_counter()
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits
                logits = self.functional.interpolate(
                    logits,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )
                probabilities = self.torch.softmax(logits.float(), dim=1)
                confidence, prediction = probabilities.max(dim=1)
            duration = time.perf_counter() - start

            raw_mask = prediction[0].cpu().numpy().astype(np.uint8)
            confidence_map = confidence[0].cpu().numpy().astype(np.float32)
            project_mask = self.project_lookup[raw_mask]
            project_mask[confidence_map < self.confidence_threshold] = (
                len(PROJECT_CLASSES) - 1)
            color_mask = PROJECT_COLORS[project_mask]

            source_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            source_msg.header = msg.header
            self.source_image_pub.publish(source_msg)

            class_msg = self.bridge.cv2_to_imgmsg(project_mask, encoding='mono8')
            class_msg.header = msg.header
            self.class_mask_pub.publish(class_msg)

            raw_class_msg = self.bridge.cv2_to_imgmsg(
                raw_mask, encoding='mono8')
            raw_class_msg.header = msg.header
            self.raw_class_mask_pub.publish(raw_class_msg)

            confidence_msg = self.bridge.cv2_to_imgmsg(
                confidence_map, encoding='32FC1')
            confidence_msg.header = msg.header
            self.confidence_pub.publish(confidence_msg)

            color_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding='rgb8')
            color_msg.header = msg.header
            self.color_mask_pub.publish(color_msg)

            self.processed_count += 1
            if self.processed_count % self.log_every_n == 0:
                fractions = np.bincount(
                    project_mask.ravel(), minlength=len(PROJECT_CLASSES))
                fractions = fractions / max(project_mask.size, 1)
                summary = ', '.join(
                    f'{name}={fraction:.1%}'
                    for name, fraction in zip(PROJECT_CLASSES, fractions)
                )
                self.get_logger().info(
                    f'SegFormer frame {self.processed_count}: '
                    f'{duration:.3f}s, '
                    f'mean_confidence={np.mean(confidence_map):.3f}, '
                    f'low_confidence={np.mean(confidence_map < self.confidence_threshold):.1%}, '
                    f'{summary}')
                if self.debug_log_raw_predictions:
                    raw_statistics = raw_prediction_statistics(
                        raw_mask,
                        confidence_map,
                        self.id2label,
                        top_k=self.debug_raw_top_k,
                        watched_labels=self.debug_watch_labels,
                        normalized_roi=self.debug_roi,
                    )
                    self.get_logger().info(
                        f'SegFormer raw model ({self.model_id}) global: '
                        + format_raw_statistics(raw_statistics['global']))
                    if raw_statistics['roi'] is not None:
                        self.get_logger().info(
                            f'SegFormer raw model ({self.model_id}) ROI '
                            f'{raw_statistics["roi_pixels"]}: '
                            + format_raw_statistics(raw_statistics['roi']))
        except Exception as exc:
            self.get_logger().error(f'SegFormer inference failed: {exc}')


def main():
    rclpy.init()
    node = SegformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
