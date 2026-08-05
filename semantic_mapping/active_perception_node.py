#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
from scipy.spatial import cKDTree

from semantic_mapping.semantic_schema import DEFAULT_CLASSES


def scale_velocity_command(
    linear_x,
    linear_y,
    angular_z,
    scale,
    linear_deadband=0.0,
    angular_deadband=0.0,
    preserve_command_curvature=True,
    scale_angular_velocity=False,
    angular_min_speed_ratio=0.0,
):
    """Scale one planar velocity command without silently changing its arc."""
    scale = float(np.clip(scale, 0.0, 1.0))
    if math.hypot(linear_x, linear_y) < float(linear_deadband):
        scaled_linear_x = 0.0
        scaled_linear_y = 0.0
    else:
        scaled_linear_x = float(linear_x) * scale
        scaled_linear_y = float(linear_y) * scale

    if abs(angular_z) < float(angular_deadband):
        scaled_angular_z = 0.0
    else:
        angular_scale = 1.0
        if preserve_command_curvature:
            angular_scale = scale
        elif scale_angular_velocity:
            angular_scale = max(scale, float(angular_min_speed_ratio))
        scaled_angular_z = float(angular_z) * angular_scale
    return scaled_linear_x, scaled_linear_y, scaled_angular_z


class ActivePerceptionNode(Node):
    def __init__(self):
        super().__init__('active_perception_node')

        # === 核心超参数 ===
        self.declare_parameter('lookahead_dist', 2.0)   # 往前看多远 (米)
        self.declare_parameter('search_radius', 0.5)    # 路径点周围的搜索半径 (米)
        # Maximum categorical entropy for the configured semantic vocabulary.
        self.declare_parameter('h_max', math.log(len(DEFAULT_CLASSES)))
        self.declare_parameter('min_speed_ratio', 0.3)  # 最低降速到 30%
        self.declare_parameter('unknown_speed_ratio', 0.65)
        self.declare_parameter('linear_deadband', 0.03)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('uncertainty_ema_gain', 0.2)
        self.declare_parameter('max_scale_step', 0.08)
        self.declare_parameter('preserve_command_curvature', True)
        self.declare_parameter('scale_angular_velocity', False)
        self.declare_parameter('angular_min_speed_ratio', 0.8)
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('entropy_topic', '/voxel_entropy_data')
        self.declare_parameter('nav_cmd_vel_topic', '/cmd_vel_nav')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('path_entropy_topic', '/path_entropy')
        self.declare_parameter('perception_mode_topic', '/perception_mode')
        self.declare_parameter('speed_scale_topic', '/semantic_speed_scale')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.search_radius = self.get_parameter('search_radius').value
        self.h_max = self.get_parameter('h_max').value
        self.min_speed_ratio = self.get_parameter('min_speed_ratio').value
        self.unknown_speed_ratio = float(
            self.get_parameter('unknown_speed_ratio').value)
        self.linear_deadband = float(
            self.get_parameter('linear_deadband').value)
        self.angular_deadband = float(
            self.get_parameter('angular_deadband').value)
        self.uncertainty_ema_gain = float(
            self.get_parameter('uncertainty_ema_gain').value)
        self.max_scale_step = float(self.get_parameter('max_scale_step').value)
        self.preserve_command_curvature = bool(
            self.get_parameter('preserve_command_curvature').value)
        self.scale_angular_velocity = bool(
            self.get_parameter('scale_angular_velocity').value)
        self.angular_min_speed_ratio = float(
            self.get_parameter('angular_min_speed_ratio').value)
        self.plan_topic = self.get_parameter('plan_topic').value
        self.entropy_topic = self.get_parameter('entropy_topic').value
        self.nav_cmd_vel_topic = self.get_parameter('nav_cmd_vel_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.path_entropy_topic = self.get_parameter('path_entropy_topic').value
        self.perception_mode_topic = self.get_parameter('perception_mode_topic').value
        self.speed_scale_topic = self.get_parameter('speed_scale_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # === TF 监听器 ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === 状态缓存 ===
        self.path_points = None
        self.cloud_tree = None
        self.cloud_entropies = None
        self.filtered_entropy = None
        self.current_speed_scale = None

        # === 订阅器 ===
        self.create_subscription(Path, self.plan_topic, self.plan_cb, 10)
        # 直接订阅 ga_bsvm_node 发的高精度熵数据 (intensity 字段 = float32 香农熵)
        self.create_subscription(PointCloud2, self.entropy_topic, self.entropy_cb, 10)
        # 拦截 Nav2 控制器输出 (启动 Nav2 时把 controller_server 的 cmd_vel remap 到这里)
        self.create_subscription(Twist, self.nav_cmd_vel_topic, self.cmd_vel_cb, 10)

        # === 发布器 ===
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)            # 发给真机/仿真的最终速度
        self.entropy_pub = self.create_publisher(Float32, self.path_entropy_topic, 10) # 监控当前路径的平均熵
        self.mode_pub = self.create_publisher(String, self.perception_mode_topic, 10)  # 状态机：CONFIDENT/EXPLORING
        self.speed_scale_pub = self.create_publisher(
            Float32, self.speed_scale_topic, 10)

        self.get_logger().info(
            "主动感知降速节点已启动: "
            f"{self.nav_cmd_vel_topic} -> {self.cmd_vel_topic}, "
            f"frames={self.odom_frame}->{self.base_frame}")

    def plan_cb(self, msg):
        """缓存 Nav2 刚规划出来的 2D 路径点"""
        pts = []
        for pose in msg.poses:
            pts.append([pose.pose.position.x, pose.pose.position.y])
        self.path_points = np.array(pts) if pts else None

    def entropy_cb(self, msg):
        """直接读取 ga_bsvm_node 发布的 float32 熵 (intensity 字段)"""
        pts = []
        entropies = []
        for p in pc2.read_points(msg, field_names=("x", "y", "intensity"), skip_nans=True):
            pts.append([p[0], p[1]])
            entropies.append(float(p[2]))  # 原汁原味的香农熵, 无精度损失

        if len(pts) > 0:
            self.cloud_tree = cKDTree(np.array(pts))
            self.cloud_entropies = np.array(entropies)

    def cmd_vel_cb(self, msg):
        """核心闭环逻辑：截获速度 -> 计算前方熵 -> 连续公式调制 -> 放行"""
        h_bar = self.h_max  # 默认极度危险（全未知）
        has_entropy_support = False

        if self.path_points is not None and self.cloud_tree is not None:
            try:
                # 1. 找狗的位置 (与 SLAM/ga_bsvm_node 一致用 odom 系)
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                )
                rx, ry = t.transform.translation.x, t.transform.translation.y

                # 2. 截取沿路径累计长度不超过 lookahead_dist 的路径片段
                dists = np.linalg.norm(self.path_points - np.array([rx, ry]), axis=1)
                closest_idx = np.argmin(dists)

                lookahead_pts = [self.path_points[closest_idx]]
                path_length = 0.0
                for i in range(closest_idx, len(self.path_points)):
                    if i > closest_idx:
                        path_length += float(np.linalg.norm(
                            self.path_points[i] - self.path_points[i - 1]))
                    if path_length > self.lookahead_dist:
                        break
                    if i > closest_idx:
                        lookahead_pts.append(self.path_points[i])

                # 3. KD树圈地：查路径周围 0.5 米内的点
                if len(lookahead_pts) > 0:
                    idxs = self.cloud_tree.query_ball_point(lookahead_pts, r=self.search_radius)
                    
                    # 展平结果并去重
                    valid_idxs = list(set([i for sublist in idxs for i in sublist]))
                    
                    if len(valid_idxs) > 0:
                        h_bar = np.mean(self.cloud_entropies[valid_idxs]) # 计算平均香农熵
                        has_entropy_support = True
                    else:
                        h_bar = self.h_max # 前方一片空白（没扫到过），满额警戒！
            except Exception as e:
                # TF 还没建立或地图还没数据时降级到 debug, 不刷屏但能查
                self.get_logger().warn(f'TF/熵查询失败 (启动期常见): {e}')

        gain = float(np.clip(self.uncertainty_ema_gain, 0.0, 1.0))
        if self.filtered_entropy is None:
            self.filtered_entropy = float(h_bar)
        else:
            self.filtered_entropy = (
                (1.0 - gain) * self.filtered_entropy + gain * float(h_bar))

        h_ratio = np.clip(self.filtered_entropy / max(self.h_max, 1e-6), 0.0, 1.0)
        if has_entropy_support:
            target_scale = self.min_speed_ratio + (
                1.0 - self.min_speed_ratio) * (1.0 - h_ratio)
        else:
            target_scale = float(np.clip(
                self.unknown_speed_ratio, self.min_speed_ratio, 1.0))
        if self.current_speed_scale is None:
            self.current_speed_scale = float(target_scale)
        else:
            scale_delta = np.clip(
                target_scale - self.current_speed_scale,
                -self.max_scale_step,
                self.max_scale_step,
            )
            self.current_speed_scale += float(scale_delta)
        alpha = float(np.clip(self.current_speed_scale, self.min_speed_ratio, 1.0))

        # 构造并发布调制后的平滑速度
        mod_cmd = Twist()
        (
            mod_cmd.linear.x,
            mod_cmd.linear.y,
            mod_cmd.angular.z,
        ) = scale_velocity_command(
            msg.linear.x,
            msg.linear.y,
            msg.angular.z,
            alpha,
            linear_deadband=self.linear_deadband,
            angular_deadband=self.angular_deadband,
            preserve_command_curvature=self.preserve_command_curvature,
            scale_angular_velocity=self.scale_angular_velocity,
            angular_min_speed_ratio=self.angular_min_speed_ratio,
        )

        self.cmd_pub.publish(mod_cmd)
        self.entropy_pub.publish(Float32(data=float(self.filtered_entropy)))
        self.speed_scale_pub.publish(Float32(data=alpha))

        # 监控发布 (供 UI 展示和录制视频用)
        if not has_entropy_support:
            mode_str = "EXPLORING"
        else:
            mode_str = "CONFIDENT" if alpha > 0.8 else "CAUTIOUS"
        self.mode_pub.publish(String(
            data=f"{mode_str} (H={self.filtered_entropy:.2f}, scale={alpha:.2f})"))

def main():
    rclpy.init()
    node = ActivePerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
