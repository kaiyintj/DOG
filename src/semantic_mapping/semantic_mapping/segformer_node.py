#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


PROJECT_CLASSES = [
    'road',
    'building',
    'tree',
    'person',
    'car',
    'unknown background',
]
PROJECT_COLORS = np.asarray([
    [120, 120, 120],
    [210, 210, 210],
    [0, 160, 0],
    [255, 0, 0],
    [0, 90, 255],
    [50, 50, 50],
], dtype=np.uint8)
LABEL_ALIASES = {
    0: {'road', 'sidewalk', 'path'},
    1: {'building', 'wall', 'house', 'fence', 'door', 'windowpane'},
    2: {'tree', 'plant', 'grass', 'flower'},
    3: {'person'},
    4: {'car', 'truck', 'bus', 'van'},
}


def normalize_label(label):
    return label.lower().replace('-', ' ').replace('_', ' ').strip()


def build_project_lookup(id2label):
    """Map a model label set into the six navigation classes."""
    class_count = max(int(class_id) for class_id in id2label) + 1
    lookup = np.full(class_count, 5, dtype=np.uint8)
    for class_id, original_label in id2label.items():
        label = normalize_label(original_label)
        for project_id, aliases in LABEL_ALIASES.items():
            if label in aliases:
                lookup[int(class_id)] = project_id
                break
    return lookup


class SegformerNode(Node):
    def __init__(self):
        super().__init__('segformer_node')
        self.declare_parameter(
            'model_id', 'nvidia/segformer-b0-finetuned-ade-512-512')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('inference_interval_sec', 0.5)
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('class_mask_topic', '/segformer/class_mask')
        self.declare_parameter('confidence_topic', '/segformer/confidence')
        self.declare_parameter('color_mask_topic', '/segformer/color_mask')
        self.declare_parameter('log_every_n', 10)

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
        self.confidence_topic = self.get_parameter('confidence_topic').value
        self.color_mask_topic = self.get_parameter('color_mask_topic').value
        self.log_every_n = max(1, int(self.get_parameter('log_every_n').value))

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

        id2label = {
            int(class_id): label
            for class_id, label in self.model.config.id2label.items()
        }
        self.project_lookup = build_project_lookup(id2label)
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
        self.confidence_pub = self.create_publisher(
            Image, self.confidence_topic, qos_profile_sensor_data)
        self.color_mask_pub = self.create_publisher(
            Image, self.color_mask_topic, qos_profile_sensor_data)

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
                rgb_image = self.bridge.compressed_imgmsg_to_cv2(
                    msg, self.image_encoding)
            else:
                rgb_image = self.bridge.imgmsg_to_cv2(
                    msg, self.image_encoding)

            height, width = rgb_image.shape[:2]
            inputs = self.processor(
                images=self.pil_image_type.fromarray(rgb_image),
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

            ade_mask = prediction[0].cpu().numpy().astype(np.uint8)
            confidence_map = confidence[0].cpu().numpy().astype(np.float32)
            project_mask = self.project_lookup[ade_mask]
            project_mask[confidence_map < self.confidence_threshold] = 5
            color_mask = PROJECT_COLORS[project_mask]

            class_msg = self.bridge.cv2_to_imgmsg(project_mask, encoding='mono8')
            class_msg.header = msg.header
            self.class_mask_pub.publish(class_msg)

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
                    f'{duration:.3f}s, {summary}')
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
