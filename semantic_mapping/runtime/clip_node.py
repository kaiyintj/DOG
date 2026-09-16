#!/usr/bin/env python3
import threading

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
# ⚠️ 1. 引入了 CompressedImage 用于真实数据集
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import open_clip
import torch

from semantic_mapping.runtime.semantic_schema import (
    DEFAULT_CLASSES,
    convert_image_to_rgb,
    resolve_clip_model_name,
    should_run_inference,
)
from semantic_mapping.runtime.semantic_posterior import float32_array_to_image


class ClipNode(Node):
    def __init__(self):
        super().__init__('clip_node')

        self.declare_parameter('model_name', 'ViT-B-32')
        self.declare_parameter('pretrained', 'openai')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('logit_scale', 0.0)
        self.declare_parameter('text_templates', [
            'a photo of a {}.',
            'a photo of the {}.',
            'a {} in the scene.',
        ])
        self.declare_parameter('vocab', list(DEFAULT_CLASSES))
        self.declare_parameter('grid_rows', 3)
        self.declare_parameter('grid_cols', 4)
        self.declare_parameter('inference_interval_sec', 0.2)
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('clip_frame_topic', '/clip/frame')
        self.declare_parameter('clip_source_image_topic', '/clip/source_image')
        self.declare_parameter('text_query_topic', '/text_query')
        self.declare_parameter('query_feature_topic', '/query_feature')

        self.model_name = self.get_parameter('model_name').value
        self.pretrained = self.get_parameter('pretrained').value
        requested_device = self.get_parameter('device').value
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable; falling back to CPU.')
            requested_device = 'cpu'
        self.device = requested_device
        self.configured_logit_scale = float(self.get_parameter('logit_scale').value)
        self.text_templates = list(self.get_parameter('text_templates').value)
        self.vocab = list(self.get_parameter('vocab').value)
        self.grid_rows = int(self.get_parameter('grid_rows').value)
        self.grid_cols = int(self.get_parameter('grid_cols').value)
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError('grid_rows and grid_cols must both be positive.')
        self.interval = float(self.get_parameter('inference_interval_sec').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.image_is_compressed = bool(self.get_parameter('image_is_compressed').value)
        self.image_encoding = self.get_parameter('image_encoding').value
        self.clip_frame_topic = self.get_parameter('clip_frame_topic').value
        self.clip_source_image_topic = self.get_parameter(
            'clip_source_image_topic').value
        self.text_query_topic = self.get_parameter('text_query_topic').value
        self.query_feature_topic = self.get_parameter('query_feature_topic').value

        resolved_model_name = resolve_clip_model_name(
            self.model_name,
            self.pretrained,
            available_models=open_clip.list_models(),
        )
        if resolved_model_name != self.model_name:
            self.get_logger().info(
                f'CLIP {self.model_name}/{self.pretrained} 使用 QuickGELU 权重，'
                f'改用匹配变体 {resolved_model_name}。')
        self.get_logger().info('正在加载 CLIP 模型...')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            resolved_model_name,
            pretrained=self.pretrained,
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(resolved_model_name)

        self.K = len(self.vocab)
        self.text_feats = self.encode_texts(self.vocab)
        self.model_lock = threading.Lock()
        if self.configured_logit_scale > 0.0:
            self.logit_scale = self.configured_logit_scale
        elif hasattr(self.model, 'logit_scale'):
            self.logit_scale = float(self.model.logit_scale.exp().clamp(max=100.0).item())
        else:
            self.logit_scale = 100.0

        self.bridge = CvBridge()
        self.last_inference_ns = None

        image_msg_type = CompressedImage if self.image_is_compressed else Image
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.query_callback_group = MutuallyExclusiveCallbackGroup()
        self.sub = self.create_subscription(
            image_msg_type,
            self.image_topic,
            self.callback,
            qos_profile_sensor_data,
            callback_group=self.image_callback_group,
        )

        self.pub_clip_frame = self.create_publisher(
            Image, self.clip_frame_topic, qos_profile_sensor_data)
        self.pub_source_image = self.create_publisher(
            Image, self.clip_source_image_topic, qos_profile_sensor_data)
        self.sub_query = self.create_subscription(
            String,
            self.text_query_topic,
            self.query_callback,
            10,
            callback_group=self.query_callback_group,
        )
        self.pub_query_feat = self.create_publisher(
            Float32MultiArray, self.query_feature_topic, 10)

        self.get_logger().info(
            f'CLIP 语义节点已就绪: topic={self.image_topic}, '
            f'compressed={self.image_is_compressed}, grid={self.grid_rows}x{self.grid_cols}, '
            f'classes={self.K}, device={self.device}, logit_scale={self.logit_scale:.1f}')

    def encode_texts(self, texts):
        """Encode text with prompt ensembling for more stable zero-shot scores."""
        encoded_texts = []
        with torch.no_grad():
            for text in texts:
                prompts = [template.format(text) for template in self.text_templates]
                tokens = self.tokenizer(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                feature = features.mean(dim=0)
                feature = feature / feature.norm().clamp(min=1e-12)
                encoded_texts.append(feature)
        return torch.stack(encoded_texts, dim=0)

    def query_callback(self, msg):
        text = msg.data.strip()
        self.get_logger().info(f'收到查询指令: "{text}"')
        vocab_lower = [item.lower() for item in self.vocab]
        if text.lower() in vocab_lower:
            class_index = vocab_lower.index(text.lower())
            text_feat = self.text_feats[class_index:class_index + 1]
        else:
            with self.model_lock:
                text_feat = self.encode_texts([text])

        feat_msg = Float32MultiArray()
        feat_msg.layout.dim = [MultiArrayDimension(
            label=f'query:{text}',
            size=int(text_feat.shape[-1]),
            stride=int(text_feat.shape[-1]),
        )]
        feat_msg.data = text_feat.squeeze(0).cpu().numpy().tolist()
        self.pub_query_feat.publish(feat_msg)

    def callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        due, baseline_ns = should_run_inference(
            now_ns, self.last_inference_ns, self.interval)
        if not due:
            return
        self.last_inference_ns = baseline_ns

        try:
            if self.image_is_compressed:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, self.image_encoding)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
            rgb_img = convert_image_to_rgb(cv_img, self.image_encoding)

            from PIL import Image as PILImage

            H, W = rgb_img.shape[:2]
            patch_h = H // self.grid_rows
            patch_w = W // self.grid_cols
            if patch_h <= 0 or patch_w <= 0:
                self.get_logger().warn(
                    f'图像尺寸过小，无法切成 {self.grid_rows}x{self.grid_cols}: {W}x{H}')
                return

            patches = []
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    y0 = r * patch_h
                    x0 = c * patch_w
                    y1 = H if r == self.grid_rows - 1 else (r + 1) * patch_h
                    x1 = W if c == self.grid_cols - 1 else (c + 1) * patch_w
                    patch = rgb_img[y0:y1, x0:x1]
                    patches.append(PILImage.fromarray(patch))

            with self.model_lock, torch.no_grad():
                batch = torch.stack([self.preprocess(p) for p in patches]).to(self.device)
                img_feats = self.model.encode_image(batch)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                # CLIP is trained with a learned temperature. Omitting this
                # scale makes a small-vocabulary softmax almost uniform.
                logits_grid = (
                    self.logit_scale * (img_feats @ self.text_feats.T)
                ).cpu().numpy()
                feats_grid = img_feats.cpu().numpy()

            clip_frame = np.concatenate(
                [
                    logits_grid.reshape(self.grid_rows, self.grid_cols, self.K),
                    feats_grid.reshape(self.grid_rows, self.grid_cols, -1),
                ],
                axis=2,
            )
            self.pub_clip_frame.publish(
                float32_array_to_image(clip_frame, msg.header))
            source_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='rgb8')
            source_msg.header = msg.header
            self.pub_source_image.publish(source_msg)

        except Exception as e:
            self.get_logger().warn(f'CLIP 推理失败: {e}')


def main():
    rclpy.init()
    node = ClipNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
