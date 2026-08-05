#!/usr/bin/env python3
"""
CLIP 文本查询脚本：把任意文本编码成 512 维特征，发到 /query_feature。
ga_bsvm_node 收到后会在 voxel_map 里搜余弦相似度最高的体素，并发 /goal_pose 给 Nav2。

用法:
    ros2 run semantic_mapping clip_query "brown rock"
    ros2 run semantic_mapping clip_query "person"
"""
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import open_clip


class ClipQuery(Node):
    def __init__(self, text):
        super().__init__('clip_query')
        self.get_logger().info(f'加载 CLIP 模型... 查询: "{text}"')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model.eval().to(device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy().flatten().astype('float32')

        self.pub = self.create_publisher(Float32MultiArray, '/query_feature', 10)
        msg = Float32MultiArray()
        msg.data = feat_np.tolist()

        self.timer = self.create_timer(0.5, lambda: self._send_once(msg))
        self.sent = False

    def _send_once(self, msg):
        if self.sent:
            rclpy.shutdown()
            return
        self.pub.publish(msg)
        self.get_logger().info(f'查询特征已发布 (dim={len(msg.data)})')
        self.sent = True


def main():
    if len(sys.argv) < 2:
        print('用法: ros2 run semantic_mapping clip_query "<text>"')
        sys.exit(1)
    text = ' '.join(sys.argv[1:])
    rclpy.init()
    node = ClipQuery(text)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
