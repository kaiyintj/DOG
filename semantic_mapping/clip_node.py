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

        self.get_logger().info('正在加载 CLIP 模型...')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model.eval()
        self.device = 'cpu'
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.vocab = ['green grass field', 'grey concrete floor', 'white wall', 'person', 'brown rock', 'black unknown background']
        self.K = len(self.vocab)
        with torch.no_grad():
            text_tokens = self.tokenizer(self.vocab).to(self.device)
            self.text_feats = self.model.encode_text(text_tokens)
            self.text_feats = self.text_feats / self.text_feats.norm(dim=-1, keepdim=True)

        self.grid_rows = 3
        self.grid_cols = 4

        self.bridge = CvBridge()
        self.last_time = self.get_clock().now()
        self.interval = 0.2

        # ====== 🌟 核心修复 1：图像订阅端切换 ======
        # [仿真/Livox专用] 
        self.sub = self.create_subscription(Image, '/d435i/image_raw', self.callback, qos_profile_sensor_data)
        
        # [M2DGR 真实数据集专用] 订阅压缩图像
        #self.sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.callback, qos_profile_sensor_data)
        # ============================================

        # ====== 🌟 核心修复 2：话题名称与 ga_bsvm_node 对齐 ======
        self.pub_logits_grid = self.create_publisher(Float32MultiArray, '/clip_logits', 10)
        self.pub_feat_grid = self.create_publisher(Float32MultiArray, '/clip_features', 10)
        # ============================================

        self.sub_query = self.create_subscription(String, '/text_query', self.query_callback, 10)
        self.pub_query_feat = self.create_publisher(Float32MultiArray, '/query_feature', 10)

        self.get_logger().info(
            f'CLIP 语义节点已就绪 (patch grid {self.grid_rows}x{self.grid_cols})')

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
            # ====== 🌟 核心修复 3：真实世界 JPEG 解压缩 ======
            # [仿真/Livox专用] 
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            
            # [M2DGR 真实数据集专用] 
            #cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
            # ============================================
            
            from PIL import Image as PILImage

            H, W = cv_img.shape[:2]
            patch_h = H // self.grid_rows
            patch_w = W // self.grid_cols
            n_patches = self.grid_rows * self.grid_cols

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