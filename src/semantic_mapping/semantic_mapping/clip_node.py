#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# ⚠️ 1. 引入了 CompressedImage 用于真实数据集
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import open_clip
import torch
import numpy as np

class ClipNode(Node):
    def __init__(self):
        super().__init__('clip_node')

        self.declare_parameter('model_name', 'ViT-B-32')
        self.declare_parameter('pretrained', 'openai')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('vocab', ['road', 'building', 'tree', 'person', 'car', 'unknown background'])
        self.declare_parameter('grid_rows', 3)
        self.declare_parameter('grid_cols', 4)
        self.declare_parameter('inference_interval_sec', 0.2)
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('clip_logits_topic', '/clip_logits')
        self.declare_parameter('clip_features_topic', '/clip_features')
        self.declare_parameter('text_query_topic', '/text_query')
        self.declare_parameter('query_feature_topic', '/query_feature')

        self.model_name = self.get_parameter('model_name').value
        self.pretrained = self.get_parameter('pretrained').value
        requested_device = self.get_parameter('device').value
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable; falling back to CPU.')
            requested_device = 'cpu'
        self.device = requested_device
        self.vocab = list(self.get_parameter('vocab').value)
        self.grid_rows = int(self.get_parameter('grid_rows').value)
        self.grid_cols = int(self.get_parameter('grid_cols').value)
        self.interval = float(self.get_parameter('inference_interval_sec').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.image_is_compressed = bool(self.get_parameter('image_is_compressed').value)
        self.image_encoding = self.get_parameter('image_encoding').value
        self.clip_logits_topic = self.get_parameter('clip_logits_topic').value
        self.clip_features_topic = self.get_parameter('clip_features_topic').value
        self.text_query_topic = self.get_parameter('text_query_topic').value
        self.query_feature_topic = self.get_parameter('query_feature_topic').value

        self.get_logger().info('正在加载 CLIP 模型...')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        self.K = len(self.vocab)
        with torch.no_grad():
            text_tokens = self.tokenizer(self.vocab).to(self.device)
            self.text_feats = self.model.encode_text(text_tokens)
            self.text_feats = self.text_feats / self.text_feats.norm(dim=-1, keepdim=True)

        self.bridge = CvBridge()
        self.last_time = self.get_clock().now()

        image_msg_type = CompressedImage if self.image_is_compressed else Image
        self.sub = self.create_subscription(
            image_msg_type,
            self.image_topic,
            self.callback,
            qos_profile_sensor_data,
        )

        self.pub_logits_grid = self.create_publisher(Float32MultiArray, self.clip_logits_topic, 10)
        self.pub_feat_grid = self.create_publisher(Float32MultiArray, self.clip_features_topic, 10)
        self.sub_query = self.create_subscription(String, self.text_query_topic, self.query_callback, 10)
        self.pub_query_feat = self.create_publisher(Float32MultiArray, self.query_feature_topic, 10)

        self.get_logger().info(
            f'CLIP 语义节点已就绪: topic={self.image_topic}, '
            f'compressed={self.image_is_compressed}, grid={self.grid_rows}x{self.grid_cols}, '
            f'classes={self.K}, device={self.device}')

    def query_callback(self, msg):
        text = msg.data
        self.get_logger().info(f'收到查询指令: "{text}"')
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_feat = self.model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        feat_msg = Float32MultiArray()
        feat_msg.data = text_feat.squeeze(0).cpu().numpy().tolist()
        self.pub_query_feat.publish(feat_msg)

    def callback(self, msg):
        now = self.get_clock().now()
        if (now - self.last_time).nanoseconds < self.interval * 1e9:
            return
        self.last_time = now

        try:
            if self.image_is_compressed:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, self.image_encoding)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
            
            from PIL import Image as PILImage

            H, W = cv_img.shape[:2]
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
                    patch = cv_img[y0:y0+patch_h, x0:x0+patch_w]
                    patches.append(PILImage.fromarray(patch))

            with torch.no_grad():
                batch = torch.stack([self.preprocess(p) for p in patches]).to(self.device)
                img_feats = self.model.encode_image(batch)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits_grid = (img_feats @ self.text_feats.T).cpu().numpy()
                feats_grid = img_feats.cpu().numpy()

            logits_msg = Float32MultiArray()
            logits_msg.layout.dim = [
                MultiArrayDimension(label='rows', size=self.grid_rows, stride=self.grid_rows * self.grid_cols * self.K),
                MultiArrayDimension(label='cols', size=self.grid_cols, stride=self.grid_cols * self.K),
                MultiArrayDimension(label='classes', size=self.K, stride=self.K),
            ]
            logits_msg.data = logits_grid.flatten().tolist()
            self.pub_logits_grid.publish(logits_msg)

            feat_dim = feats_grid.shape[1]
            feat_msg = Float32MultiArray()
            feat_msg.layout.dim = [
                MultiArrayDimension(label='rows', size=self.grid_rows, stride=self.grid_rows * self.grid_cols * feat_dim),
                MultiArrayDimension(label='cols', size=self.grid_cols, stride=self.grid_cols * feat_dim),
                MultiArrayDimension(label='feat', size=feat_dim, stride=feat_dim),
            ]
            feat_msg.data = feats_grid.flatten().tolist()
            self.pub_feat_grid.publish(feat_msg)

        except Exception as e:
            self.get_logger().warn(f'CLIP 推理失败: {e}')

def main():
    rclpy.init()
    node = ClipNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
