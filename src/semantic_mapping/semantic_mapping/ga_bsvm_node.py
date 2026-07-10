#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, Header, String
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import numpy as np
import message_filters
import struct
from collections import deque
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

# [仿真/Livox专用] 如果没装livox包这里可能会报错，可以注释掉下面这行
from livox_ros_driver2.msg import CustomMsg

from semantic_mapping.voxel_map import VoxelMap
# ⚠️ 终于加进来的 QoS 协议包！
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy

class GABsvmNode(Node):
    def __init__(self):
        super().__init__('ga_bsvm_node')
        self.bridge = CvBridge()

        self.declare_parameter('pointcloud_topic', '/velodyne_points')
        self.declare_parameter('pointcloud_type', 'pointcloud2')
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('imu_topic', '/handsfree/imu')
        self.declare_parameter('clip_logits_topic', '/clip_logits')
        self.declare_parameter('clip_features_topic', '/clip_features')
        self.declare_parameter('text_query_topic', '/text_query')
        self.declare_parameter('query_feature_topic', '/query_feature')
        self.declare_parameter('semantic_cloud_topic', '/semantic_cloud')
        self.declare_parameter('uncertainty_cloud_topic', '/uncertainty_cloud')
        self.declare_parameter('entropy_topic', '/voxel_entropy_data')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('semantic_cost_map_topic', '/semantic_cost_map')
        self.declare_parameter('query_target_pose_topic', '/query_target_pose')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('grid_rows', 3)
        self.declare_parameter('grid_cols', 4)
        self.declare_parameter('num_classes', 6)
        self.declare_parameter('feat_dim', 512)
        self.declare_parameter('vocab', ['road', 'building', 'tree', 'person', 'car', 'unknown background'])
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('evidence_prior', 1.0)
        self.declare_parameter('evidence_strength', 1.0)
        self.declare_parameter('evidence_decay', 0.995)
        self.declare_parameter('max_frame_evidence', 3.0)
        self.declare_parameter('max_total_evidence', 200.0)
        self.declare_parameter('uncertainty_entropy_weight', 0.7)
        self.declare_parameter('semantic_costs', [0, 100, 100, 100, 100, -1])
        self.declare_parameter('query_min_weight_sum', 2.0)
        self.declare_parameter('query_min_class_prob', 0.25)
        self.declare_parameter('query_min_similarity', 0.18)
        self.declare_parameter('query_feature_weight', 0.7)
        self.declare_parameter('query_class_weight', 0.3)
        self.declare_parameter('query_allow_feature_fallback', True)
        self.declare_parameter('query_approach_radius_m', 1.5)
        self.declare_parameter('query_approach_distance_m', 1.0)
        self.declare_parameter('query_approach_min_distance_m', 0.6)
        self.declare_parameter('query_clearance_radius_m', 0.35)
        self.declare_parameter('query_require_safe_approach', True)
        self.declare_parameter('query_cluster_radius_m', 0.45)
        self.declare_parameter('query_cluster_min_voxels', 3)
        self.declare_parameter('query_cluster_min_evidence', 3.0)
        self.declare_parameter('query_cluster_support_weight', 0.08)
        self.declare_parameter('query_distance_weight', 0.01)
        self.declare_parameter('query_max_candidates', 5000)
        self.declare_parameter(
            'query_class_max_extent_m', [100.0, 100.0, 100.0, 1.5, 5.0, 100.0])
        self.declare_parameter('cost_map_size_m', 50.0)
        self.declare_parameter('costmap_min_confidence', 0.15)
        self.declare_parameter('costmap_robot_clearance_m', 0.4)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.2)
        self.declare_parameter('process_every_n_frames', 5)
        self.declare_parameter('entropy_publish_every_n_processed', 2)
        self.declare_parameter('cloud_publish_every_n_processed', 10)
        self.declare_parameter('imu_window_sec', 0.15)
        self.declare_parameter('motion_angular_scale', 2.0)
        self.declare_parameter('motion_accel_scale', 3.0)
        self.declare_parameter('motion_min_reliability', 0.2)
        self.declare_parameter('imu_gravity', 9.81)
        self.declare_parameter('density_scale', 8.0)
        self.declare_parameter('range_scale_m', 20.0)
        self.declare_parameter('view_edge_penalty', 0.4)
        self.declare_parameter('semantic_confidence_floor', 0.15)
        self.declare_parameter('camera_k', [
            554.25, 0.0, 320.5,
            0.0, 554.25, 240.5,
            0.0, 0.0, 1.0,
        ])
        self.declare_parameter('lidar_to_camera_translation', [0.2, 0.0, -0.03])
        self.declare_parameter('lidar_to_camera_quaternion', [-0.5, 0.5, -0.5, 0.5])

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.pointcloud_type = self.get_parameter('pointcloud_type').value
        self.image_topic = self.get_parameter('image_topic').value
        self.image_is_compressed = bool(self.get_parameter('image_is_compressed').value)
        self.image_encoding = self.get_parameter('image_encoding').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.clip_logits_topic = self.get_parameter('clip_logits_topic').value
        self.clip_features_topic = self.get_parameter('clip_features_topic').value
        self.text_query_topic = self.get_parameter('text_query_topic').value
        self.query_feature_topic = self.get_parameter('query_feature_topic').value
        self.semantic_cloud_topic = self.get_parameter('semantic_cloud_topic').value
        self.uncertainty_cloud_topic = self.get_parameter('uncertainty_cloud_topic').value
        self.entropy_topic = self.get_parameter('entropy_topic').value
        self.goal_pose_topic = self.get_parameter('goal_pose_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.semantic_cost_map_topic = self.get_parameter('semantic_cost_map_topic').value
        self.query_target_pose_topic = self.get_parameter(
            'query_target_pose_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.grid_rows = int(self.get_parameter('grid_rows').value)
        self.grid_cols = int(self.get_parameter('grid_cols').value)
        self.num_classes = int(self.get_parameter('num_classes').value)
        self.feat_dim = int(self.get_parameter('feat_dim').value)
        self.vocab = list(self.get_parameter('vocab').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.evidence_prior = float(self.get_parameter('evidence_prior').value)
        self.evidence_strength = float(self.get_parameter('evidence_strength').value)
        self.evidence_decay = float(self.get_parameter('evidence_decay').value)
        self.max_frame_evidence = float(self.get_parameter('max_frame_evidence').value)
        self.max_total_evidence = float(self.get_parameter('max_total_evidence').value)
        self.uncertainty_entropy_weight = float(
            self.get_parameter('uncertainty_entropy_weight').value)
        self.query_min_weight_sum = float(self.get_parameter('query_min_weight_sum').value)
        self.query_min_class_prob = float(self.get_parameter('query_min_class_prob').value)
        self.query_min_similarity = float(self.get_parameter('query_min_similarity').value)
        self.query_feature_weight = float(self.get_parameter('query_feature_weight').value)
        self.query_class_weight = float(self.get_parameter('query_class_weight').value)
        self.query_allow_feature_fallback = bool(self.get_parameter('query_allow_feature_fallback').value)
        self.query_approach_radius_m = float(self.get_parameter('query_approach_radius_m').value)
        self.query_approach_distance_m = float(
            self.get_parameter('query_approach_distance_m').value)
        self.query_approach_min_distance_m = float(
            self.get_parameter('query_approach_min_distance_m').value)
        self.query_clearance_radius_m = float(
            self.get_parameter('query_clearance_radius_m').value)
        self.query_require_safe_approach = bool(
            self.get_parameter('query_require_safe_approach').value)
        self.query_cluster_radius_m = float(
            self.get_parameter('query_cluster_radius_m').value)
        self.query_cluster_min_voxels = int(
            self.get_parameter('query_cluster_min_voxels').value)
        self.query_cluster_min_evidence = float(
            self.get_parameter('query_cluster_min_evidence').value)
        self.query_cluster_support_weight = float(
            self.get_parameter('query_cluster_support_weight').value)
        self.query_distance_weight = float(
            self.get_parameter('query_distance_weight').value)
        self.query_max_candidates = int(self.get_parameter('query_max_candidates').value)
        self.query_class_max_extent_m = list(
            self.get_parameter('query_class_max_extent_m').value)
        self.cost_map_size_m = float(self.get_parameter('cost_map_size_m').value)
        self.costmap_min_confidence = float(
            self.get_parameter('costmap_min_confidence').value)
        self.costmap_robot_clearance_m = float(
            self.get_parameter('costmap_robot_clearance_m').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.imu_window_sec = float(self.get_parameter('imu_window_sec').value)
        self.motion_angular_scale = float(self.get_parameter('motion_angular_scale').value)
        self.motion_accel_scale = float(self.get_parameter('motion_accel_scale').value)
        self.motion_min_reliability = float(
            self.get_parameter('motion_min_reliability').value)
        self.imu_gravity = float(self.get_parameter('imu_gravity').value)
        self.density_scale = float(self.get_parameter('density_scale').value)
        self.range_scale_m = float(self.get_parameter('range_scale_m').value)
        self.view_edge_penalty = float(self.get_parameter('view_edge_penalty').value)
        self.semantic_confidence_floor = float(
            self.get_parameter('semantic_confidence_floor').value)
        self.process_every_n_frames = max(1, int(self.get_parameter('process_every_n_frames').value))
        self.entropy_publish_every_n_processed = max(
            1,
            int(self.get_parameter('entropy_publish_every_n_processed').value),
        )
        self.cloud_publish_every_n_processed = max(
            1,
            int(self.get_parameter('cloud_publish_every_n_processed').value),
        )
        
        trans = self._float_list_param('lidar_to_camera_translation', 3)
        quat = self._float_list_param('lidar_to_camera_quaternion', 4)
        
        rot_matrix = R.from_quat(quat).as_matrix()
        self.T_lidar2cam = np.eye(4, dtype=np.float64)
        self.T_lidar2cam[:3, :3] = rot_matrix
        self.T_lidar2cam[:3, 3] = trans
        
        self.K = np.array(self._float_list_param('camera_k', 9), dtype=np.float64).reshape(3, 3)

        # === 2. 传感器订阅与同步 ===
        self.imu_buffer = deque(maxlen=1000)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)

        self.clip_logits_sub = self.create_subscription(
            Float32MultiArray,
            self.clip_logits_topic,
            self.logits_callback,
            10,
        )
        self.clip_features_sub = self.create_subscription(
            Float32MultiArray,
            self.clip_features_topic,
            self.features_callback,
            10,
        )

        self.logits_grid = None
        self.feats_grid = None

        if self.pointcloud_type == 'livox_custom':
            pointcloud_msg_type = CustomMsg
        elif self.pointcloud_type == 'pointcloud2':
            pointcloud_msg_type = PointCloud2
        else:
            self.get_logger().warn(
                f'未知 pointcloud_type={self.pointcloud_type}，回退到 pointcloud2。')
            self.pointcloud_type = 'pointcloud2'
            pointcloud_msg_type = PointCloud2

        image_msg_type = CompressedImage if self.image_is_compressed else Image

        self.pc_sub = message_filters.Subscriber(
            self,
            pointcloud_msg_type,
            self.pointcloud_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.img_sub = message_filters.Subscriber(
            self,
            image_msg_type,
            self.image_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, self.img_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.voxel_map = VoxelMap(
            voxel_size=self.voxel_size,
            K=self.num_classes,
            evidence_prior=self.evidence_prior,
            evidence_strength=self.evidence_strength,
            evidence_decay=self.evidence_decay,
            max_frame_evidence=self.max_frame_evidence,
            max_total_evidence=self.max_total_evidence,
            uncertainty_entropy_weight=self.uncertainty_entropy_weight,
        )
        self.get_logger().info(
            f'Dirichlet VoxelMap 已创建，网格精度: {self.voxel_size:.3f}m')
        
        self.pub_semantic = self.create_publisher(PointCloud2, self.semantic_cloud_topic, 1)
        self.pub_uncertainty = self.create_publisher(PointCloud2, self.uncertainty_cloud_topic, 1)
        # 高精度熵数据话题: x/y/z + intensity(float32 香农熵)，供 active_perception_node 直读
        self.pub_entropy_data = self.create_publisher(PointCloud2, self.entropy_topic, 10)
        self.text_query_sub = self.create_subscription(
            String,
            self.text_query_topic,
            self.text_query_cb,
            10,
        )
        self.query_sub = self.create_subscription(
            Float32MultiArray,
            self.query_feature_topic,
            self.query_feature_cb,
            10,
        )
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, 10)
        self.query_target_pub = self.create_publisher(
            PoseStamped, self.query_target_pose_topic, 10)
        
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, map_qos)

        semantic_costs = list(self.get_parameter('semantic_costs').value)
        if len(semantic_costs) != self.num_classes:
            self.get_logger().warn(
                f'semantic_costs 长度 {len(semantic_costs)} 与 num_classes={self.num_classes} 不一致，使用默认规则。')
            if self.num_classes >= 2:
                semantic_costs = [0] + [100] * (self.num_classes - 2) + [-1]
            else:
                semantic_costs = [0] * self.num_classes
        self.semantic_cost_dict = {idx: int(cost) for idx, cost in enumerate(semantic_costs)}

        # 固定尺寸 50m x 50m，避免 StaticLayer 频繁 resizeMap 拖死 Nav2
        self.cost_map_origin_x = -self.cost_map_size_m / 2.0
        self.cost_map_origin_y = -self.cost_map_size_m / 2.0
        self.semantic_cost_pub = self.create_publisher(
            OccupancyGrid,
            self.semantic_cost_map_topic,
            map_qos,
        )
        # 1Hz 独立定时器，与点云回调解耦，避免数据海啸
        self.create_timer(1.0, self.publish_semantic_costmap)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.frame_count = 0

        self.current_goal_key = None # 用于记住目标的网格位置
        self.last_query_text = ''
        self.get_logger().info(
            'GA-BSVM 中枢已启动: '
            f'pointcloud={self.pointcloud_topic} ({self.pointcloud_type}), '
            f'image={self.image_topic} compressed={self.image_is_compressed}, '
            f'imu={self.imu_topic}, frames={self.odom_frame}->{self.base_frame}')

    def _float_list_param(self, name, expected_len):
        values = list(self.get_parameter(name).value)
        if len(values) != expected_len:
            raise ValueError(f'参数 {name} 需要 {expected_len} 个数值，当前为 {len(values)} 个。')
        return [float(v) for v in values]

    def imu_callback(self, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        angular_norm = np.linalg.norm([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        acceleration_norm = np.linalg.norm([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        self.imu_buffer.append((stamp_ns, angular_norm, acceleration_norm))

    # 🗡️ 刺客2修复：加入维度校验报警
    def logits_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        expected = self.grid_rows * self.grid_cols * self.num_classes
        if data.size == expected:
            self.logits_grid = data.reshape(self.grid_rows, self.grid_cols, self.num_classes)
        else:
            self.get_logger().warn(f'⚠️ Logits 维度不匹配被抛弃！收到: {data.size}, 期望: {expected} (请检查查询词数量是否为 {self.num_classes})')

    def features_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        expected = self.grid_rows * self.grid_cols * self.feat_dim
        if data.size == expected:
            self.feats_grid = data.reshape(self.grid_rows, self.grid_cols, self.feat_dim)
        else:
            self.get_logger().warn(f'⚠️ 特征 维度不匹配被抛弃！收到: {data.size}, 期望: {expected}')

    def compute_motion_reliability(self, stamp):
        """Estimate image/point reliability from time-aligned body motion."""
        if not self.imu_buffer:
            return 1.0, 0.0, 0.0

        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        window_ns = max(int(self.imu_window_sec * 1e9), 1)
        samples = [
            sample for sample in self.imu_buffer
            if abs(sample[0] - stamp_ns) <= window_ns
        ]
        if not samples:
            return 1.0, 0.0, 0.0

        angular_rms = float(np.sqrt(np.mean([sample[1] ** 2 for sample in samples])))
        acceleration_deviation = float(np.sqrt(np.mean([
            (sample[2] - self.imu_gravity) ** 2 for sample in samples
        ])))
        angular_scale = max(self.motion_angular_scale, 1e-6)
        acceleration_scale = max(self.motion_accel_scale, 1e-6)
        exponent = -0.5 * (
            (angular_rms / angular_scale) ** 2
            + (acceleration_deviation / acceleration_scale) ** 2
        )
        reliability = float(np.clip(
            np.exp(exponent), self.motion_min_reliability, 1.0))
        return reliability, angular_rms, acceleration_deviation

    def parse_livox_msg(self, msg):
        if len(msg.points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        valid = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points, axis=1) > 0.1)
        return points[valid]

    def parse_pointcloud2_msg(self, msg):
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        points = np.array(points, dtype=np.float32)
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        valid = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points, axis=1) > 0.1)
        return points[valid]

    def project_points(self, points_3d, H, W):
        n = len(points_3d)
        pts_h = np.hstack([points_3d, np.ones((n, 1))])
        pts_cam = (self.T_lidar2cam @ pts_h.T).T
        valid_z = pts_cam[:, 2] > 0.1 
        u = (self.K[0,0] * pts_cam[:,0] / pts_cam[:,2] + self.K[0,2]).astype(int)
        v = (self.K[1,1] * pts_cam[:,1] / pts_cam[:,2] + self.K[1,2]).astype(int)
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_mask = valid_z & valid_uv
        return valid_mask, u, v

    def text_query_cb(self, msg):
        self.last_query_text = msg.data.strip().lower()

    def get_query_class_index(self):
        if not self.last_query_text:
            return None

        vocab_lower = [name.lower() for name in self.vocab]
        if self.last_query_text in vocab_lower:
            return vocab_lower.index(self.last_query_text)

        for idx, name in enumerate(vocab_lower):
            tokens = name.replace('_', ' ').split()
            if self.last_query_text in tokens:
                return idx

        return None

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ], dtype=np.float32)
        except Exception:
            return None

    def select_query_cluster(self, candidates, query_class_idx=None):
        """Select a spatially supported target instead of one maximum voxel."""
        if not candidates:
            return None

        candidates = sorted(
            candidates, key=lambda item: item['score'], reverse=True
        )[:max(self.query_max_candidates, 1)]
        positions = np.asarray([item['pos'][:2] for item in candidates])
        tree = cKDTree(positions)
        neighborhoods = tree.query_ball_point(positions, self.query_cluster_radius_m)
        visited = np.zeros(len(candidates), dtype=bool)
        clusters = []

        for seed in range(len(candidates)):
            if visited[seed]:
                continue
            stack = [seed]
            visited[seed] = True
            indices = []
            while stack:
                current = stack.pop()
                indices.append(current)
                for neighbor in neighborhoods[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            clusters.append(indices)

        robot_position = self.get_robot_position()
        best_cluster = None
        best_utility = -float('inf')
        for indices in clusters:
            cluster = [candidates[index] for index in indices]
            total_evidence = sum(item['evidence'] for item in cluster)
            if len(cluster) < self.query_cluster_min_voxels:
                continue
            if total_evidence < self.query_cluster_min_evidence:
                continue

            cluster_positions = np.asarray([item['pos'] for item in cluster])
            horizontal_extent = float(np.max(
                np.ptp(cluster_positions[:, :2], axis=0)))
            if (
                query_class_idx is not None
                and query_class_idx < len(self.query_class_max_extent_m)
                and horizontal_extent > self.query_class_max_extent_m[query_class_idx]
            ):
                continue

            evidence_weights = np.asarray([
                min(item['evidence'], 10.0) for item in cluster
            ], dtype=np.float64)
            scores = np.asarray([item['score'] for item in cluster])
            semantic_score = 0.8 * float(np.average(scores, weights=evidence_weights))
            semantic_score += 0.2 * float(np.max(scores))
            support = min(np.log1p(total_evidence) / np.log(31.0), 1.0)
            centroid = np.average(
                cluster_positions,
                axis=0,
                weights=evidence_weights,
            )
            distance = 0.0
            if robot_position is not None:
                distance = float(np.linalg.norm(centroid[:2] - robot_position[:2]))
            utility = (
                semantic_score
                + self.query_cluster_support_weight * support
                - self.query_distance_weight * distance
            )
            if utility > best_utility:
                best_utility = utility
                representative = max(cluster, key=lambda item: item['score'])
                best_cluster = {
                    'key': representative['key'],
                    'pos': centroid.astype(np.float32),
                    'score': semantic_score,
                    'similarity': max(item['similarity'] for item in cluster),
                    'class_probability': max(
                        item['class_probability'] for item in cluster),
                    'voxel_count': len(cluster),
                    'evidence': total_evidence,
                    'utility': utility,
                }
        return best_cluster

    def find_approach_goal(self, object_pos, query_class_idx):
        """Find a supported road voxel from which the object can be observed."""
        is_traversable_query = (
            query_class_idx is not None
            and self.semantic_cost_dict.get(query_class_idx, 100) < 100
        )
        if is_traversable_query:
            key = self.voxel_map.get_voxel_indices(object_pos)
            return key, object_pos

        road_candidates = []
        obstacle_positions = []
        for key, voxel in self.voxel_map.voxels.items():
            if voxel['weight_sum'] < self.query_min_weight_sum:
                continue
            probabilities = self.voxel_map.get_probabilities(key)
            class_index = int(np.argmax(probabilities))
            position = voxel['pos']
            if self.semantic_cost_dict.get(class_index, -1) >= 100:
                obstacle_positions.append(position[:2])
            if class_index != 0:
                continue
            distance_to_object = float(np.linalg.norm(position[:2] - object_pos[:2]))
            if not (
                self.query_approach_min_distance_m
                <= distance_to_object
                <= self.query_approach_radius_m
            ):
                continue
            road_candidates.append((key, voxel, distance_to_object))

        obstacle_tree = None
        if obstacle_positions:
            obstacle_tree = cKDTree(np.asarray(obstacle_positions))
        robot_position = self.get_robot_position()
        best = None
        best_cost = float('inf')
        for key, voxel, distance_to_object in road_candidates:
            position = voxel['pos']
            if obstacle_tree is not None:
                nearby_obstacles = obstacle_tree.query_ball_point(
                    position[:2], self.query_clearance_radius_m)
                if nearby_obstacles:
                    continue
            confidence = self.voxel_map.get_confidence(key)
            robot_distance = 0.0
            height_cost = 0.0
            if robot_position is not None:
                robot_distance = float(
                    np.linalg.norm(position[:2] - robot_position[:2]))
                height_cost = abs(float(position[2] - robot_position[2]))
            cost = (
                abs(distance_to_object - self.query_approach_distance_m)
                + 0.02 * robot_distance
                + 0.2 * height_cost
                - 0.1 * confidence
            )
            if cost < best_cost:
                best_cost = cost
                best = (key, position)

        if best is not None:
            self.get_logger().info(
                f'安全接近点: x={best[1][0]:.2f}, y={best[1][1]:.2f}')
            return best
        if self.query_require_safe_approach:
            return None, None

        # Simulation fallback: use a line-of-sight standoff point instead of
        # navigating into the object center. Nav2's geometric layers still
        # perform the final collision check.
        fallback_position = np.array(object_pos, dtype=np.float32)
        if robot_position is not None:
            direction = robot_position[:2] - object_pos[:2]
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-6:
                fallback_position[:2] = (
                    object_pos[:2]
                    + direction / direction_norm * self.query_approach_distance_m
                )
            fallback_position[2] = robot_position[2]
        self.get_logger().warn(
            '没有找到满足语义约束的road体素，使用视线方向接近点；'
            '该回退仅应用于仿真配置。')
        return (
            self.voxel_map.get_voxel_indices(fallback_position),
            fallback_position,
        )

    def query_feature_cb(self, msg):
        query_feat = np.array(msg.data, dtype=np.float32)
        if query_feat.size != self.feat_dim:
            self.get_logger().warn(
                f'查询特征维度错误: {query_feat.size}, 期望 {self.feat_dim}')
            return
        query_norm = np.linalg.norm(query_feat)
        if query_norm <= 1e-6:
            self.get_logger().warn('查询特征范数为零，忽略本次查询。')
            return
        query_feat = query_feat / query_norm

        query_class_idx = self.get_query_class_index()
        accepted_candidates = []
        feature_candidates = []
        candidate_count = 0
        class_rejected_count = 0

        for key, voxel in self.voxel_map.voxels.items():
            if voxel['feature_512'] is None or voxel['weight_sum'] < self.query_min_weight_sum:
                continue

            voxel_feat = voxel['feature_512']
            voxel_norm = np.linalg.norm(voxel_feat)
            if voxel_norm <= 1e-6:
                continue

            candidate_count += 1
            sim = float(np.dot(voxel_feat / voxel_norm, query_feat))
            if sim < self.query_min_similarity:
                continue
            class_prob = 0.0
            score = sim
            candidate = {
                'key': key,
                'pos': voxel['pos'],
                'similarity': sim,
                'class_probability': class_prob,
                'score': score,
                'evidence': voxel['weight_sum'],
            }
            feature_candidates.append(candidate)

            if query_class_idx is not None:
                probs = self.voxel_map.get_probabilities(key)
                class_prob = float(probs[query_class_idx])
                if class_prob < self.query_min_class_prob:
                    class_rejected_count += 1
                    continue
                score = self.query_feature_weight * sim + self.query_class_weight * class_prob
                candidate = dict(candidate)
                candidate['class_probability'] = class_prob
                candidate['score'] = score
            accepted_candidates.append(candidate)

        used_fallback = False
        selected = self.select_query_cluster(
            accepted_candidates, query_class_idx=query_class_idx)
        if selected is None and self.query_allow_feature_fallback:
            used_fallback = True
            selected = self.select_query_cluster(
                feature_candidates, query_class_idx=query_class_idx)

        if selected is not None:
            object_pos = selected['pos']
            target = PoseStamped()
            target.header.frame_id = self.odom_frame
            target.header.stamp = self.get_clock().now().to_msg()
            target.pose.position.x = float(object_pos[0])
            target.pose.position.y = float(object_pos[1])
            target.pose.position.z = float(object_pos[2])
            target.pose.orientation.w = 1.0
            self.query_target_pub.publish(target)
            goal_key, goal_pos = self.find_approach_goal(object_pos, query_class_idx)
            if goal_pos is None:
                self.get_logger().warn(
                    f'找到 query="{self.last_query_text}"，但附近没有满足间距和净空要求的road接近点。')
                return
            self.current_goal_key = goal_key
            if used_fallback:
                self.get_logger().warn(
                    f'query="{self.last_query_text}" 没有体素通过类别概率阈值 '
                    f'({class_rejected_count}/{candidate_count} rejected, '
                    f'min_class_prob={self.query_min_class_prob:.2f})，'
                    '改用 CLIP 特征相似度兜底。')
            self.get_logger().info(
                f'🎯 [导航触发] query="{self.last_query_text}", '
                f'score={selected["score"]:.3f}, sim={selected["similarity"]:.3f}, '
                f'class_prob={selected["class_probability"]:.3f}, '
                f'cluster={selected["voxel_count"]} voxels, evidence={selected["evidence"]:.1f}, '
                f'目标坐标: x={object_pos[0]:.2f}, y={object_pos[1]:.2f}, '
                f'导航坐标: x={goal_pos[0]:.2f}, y={goal_pos[1]:.2f}')
            
            goal = PoseStamped()
            goal.header.frame_id = self.odom_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(goal_pos[0])
            goal.pose.position.y = float(goal_pos[1])
            goal.pose.position.z = float(goal_pos[2])
            yaw = float(np.arctan2(
                object_pos[1] - goal_pos[1], object_pos[0] - goal_pos[0]))
            goal.pose.orientation.z = float(np.sin(yaw / 2.0))
            goal.pose.orientation.w = float(np.cos(yaw / 2.0))
            self.goal_pub.publish(goal)
        else:
            self.get_logger().warn(
                f'🔍 未找到合适目标。query="{self.last_query_text}", '
                f'候选体素={candidate_count}, 类别阈值过滤={class_rejected_count}')

    def sync_callback(self, pc_msg, img_msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        motion_reliability, angular_rms, acceleration_deviation = (
            self.compute_motion_reliability(pc_msg.header.stamp)
        )

        if self.logits_grid is None:
            return

        try:
            if self.image_is_compressed:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, self.image_encoding)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, self.image_encoding)
            H, W = cv_img.shape[:2]
            patch_h = H // self.grid_rows
            patch_w = W // self.grid_cols
            if patch_h <= 0 or patch_w <= 0:
                self.get_logger().warn(
                    f'图像尺寸过小，无法切成 {self.grid_rows}x{self.grid_cols}: {W}x{H}')
                return

            if self.pointcloud_type == 'livox_custom':
                points_3d = self.parse_livox_msg(pc_msg)
            else:
                points_3d = self.parse_pointcloud2_msg(pc_msg)
            
            valid_mask, u, v = self.project_points(points_3d, H, W)
            valid_points = points_3d[valid_mask]
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]
            sensor_ranges = np.linalg.norm(valid_points, axis=1)

            if len(valid_points) == 0:
                return

            try:
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                )
                t_vec = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                rot_mat = R.from_quat(q).as_matrix()
                valid_points = (rot_mat @ valid_points.T).T + t_vec
            except Exception as e:
                # 只有这里去掉了报错提示，避免真实数据集缺 TF 时疯狂刷屏
                return

            grid_r = np.clip(valid_v // patch_h, 0, self.grid_rows - 1)
            grid_c = np.clip(valid_u // patch_w, 0, self.grid_cols - 1)

            per_point_logits = self.logits_grid[grid_r, grid_c]

            per_point_feats = None
            if self.feats_grid is not None:
                per_point_feats = self.feats_grid[grid_r, grid_c]

            tree = cKDTree(valid_points)
            neighbors = tree.query_ball_point(valid_points, r=0.3)
            density = np.asarray([len(neighbor) for neighbor in neighbors])
            density_reliability = 1.0 - np.exp(
                -density / max(self.density_scale, 1e-6))
            range_reliability = np.exp(
                -np.square(sensor_ranges / max(self.range_scale_m, 1e-6)))

            normalized_u = (valid_u - 0.5 * W) / max(0.5 * W, 1.0)
            normalized_v = (valid_v - 0.5 * H) / max(0.5 * H, 1.0)
            radial_distance = np.clip(
                np.sqrt(normalized_u ** 2 + normalized_v ** 2) / np.sqrt(2.0),
                0.0,
                1.0,
            )
            view_reliability = np.clip(
                1.0 - self.view_edge_penalty * radial_distance ** 2,
                0.05,
                1.0,
            )

            point_probabilities = self.voxel_map.softmax(per_point_logits)
            semantic_entropy = -np.sum(
                point_probabilities
                * np.log(np.clip(point_probabilities, 1e-10, 1.0)),
                axis=1,
            )
            semantic_certainty = 1.0 - semantic_entropy / max(
                np.log(self.num_classes), 1e-9)
            semantic_reliability = (
                self.semantic_confidence_floor
                + (1.0 - self.semantic_confidence_floor) * semantic_certainty
            )

            reliability = (
                motion_reliability
                * density_reliability
                * range_reliability
                * view_reliability
                * semantic_reliability
            ).astype(np.float32)

            self.voxel_map.update(
                points=valid_points,
                reliability=reliability,
                logits=per_point_logits,
                features=per_point_feats
            )

            processed_count = self.frame_count // self.process_every_n_frames

            if processed_count % self.entropy_publish_every_n_processed == 0:
                self.publish_entropy_data()

            if processed_count % self.cloud_publish_every_n_processed == 0:
                sem_pts, unc_pts = self.voxel_map.get_visualization_clouds()
                if len(sem_pts) > 0:
                    header = Header()
                    header.stamp = pc_msg.header.stamp
                    header.frame_id = self.odom_frame

                    sem_msg = self.create_cloud_msg(header, sem_pts)
                    self.pub_semantic.publish(sem_msg)

                    unc_msg = self.create_cloud_msg(header, unc_pts)
                    self.pub_uncertainty.publish(unc_msg)

                    self.get_logger().info(
                        f'投影并融合 {len(valid_points)} 个3D点: '
                        f'motion={motion_reliability:.2f}, '
                        f'angular_rms={angular_rms:.2f}rad/s, '
                        f'accel_dev={acceleration_deviation:.2f}m/s^2')

        except Exception as e:
            self.get_logger().error(f"投影过程中出错: {e}")

    def create_cloud_msg(self, header, points_rgb):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points_rgb)
        msg.is_dense = False
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        buffer = bytearray(msg.row_step)
        
        for i, p in enumerate(points_rgb):
            r, g, b = int(p[3]), int(p[4]), int(p[5])
            rgb_int = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            struct.pack_into('ffff', buffer, i * 16, p[0], p[1], p[2], rgb_float)
            
        msg.data = bytes(buffer)
        return msg

    def publish_entropy_data(self):
        """
        发布纯数据熵话题 (xyz + intensity=float32 香农熵)
        用 VoxelMap.get_confidence(key) 反推熵: H = (1 - confidence) * ln(K)
        给 active_perception_node 提供高精度熵源, 替代从颜色反解的低精度方案
        """
        if not self.voxel_map.voxels:
            return

        H_max = np.log(self.voxel_map.K)
        pts = []
        for key, voxel in self.voxel_map.voxels.items():
            # 跳过未被有效观测的体素 (置信度还未建立)
            if voxel['weight_sum'] < 0.1:
                continue
            pos = voxel['pos']
            _, _, uncertainty = self.voxel_map.get_uncertainty(key)
            equivalent_entropy = uncertainty * H_max
            pts.append([
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(equivalent_entropy),
            ])

        if len(pts) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.odom_frame

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg = pc2.create_cloud(header, fields, pts)
        self.pub_entropy_data.publish(msg)

    def publish_semantic_costmap(self):
        """Project the 3D semantic posterior into a fixed 2D Nav2 map."""
        if not self.voxel_map.voxels:
            return

        voxel_size = self.voxel_map.voxel_size
        width = int(self.cost_map_size_m / voxel_size)
        height = int(self.cost_map_size_m / voxel_size)

        # 像素索引偏移：voxel 索引 ix 对应世界 ix*voxel_size，减 origin 再除分辨率
        ox_idx = int(self.cost_map_origin_x / voxel_size)
        oy_idx = int(self.cost_map_origin_y / voxel_size)

        # 初始化全 -1 (未知)
        cost_data = [-1] * (width * height)

        # 把 3D 语义拍扁到 2D 网格
        for key, voxel in self.voxel_map.voxels.items():
            # 跳过未充分观测的体素，避免 CLIP 第一帧瞎猜污染地图
            if voxel['weight_sum'] < 1.0:
                continue

            ix, iy, iz = key
            confidence = self.voxel_map.get_confidence(key)
            if confidence < self.costmap_min_confidence:
                continue
            best_class = int(np.argmax(self.voxel_map.get_probabilities(key)))
            cost_value = int(self.semantic_cost_dict.get(best_class, -1))

            px = ix - ox_idx
            py = iy - oy_idx
            if 0 <= px < width and 0 <= py < height:
                idx = py * width + px
                # 取最大危险值，例如 road 单元里观测到 person/car 时保留高代价。
                if cost_data[idx] < cost_value:
                    cost_data[idx] = cost_value

        # The selected goal is already a verified road voxel. Clear only a
        # small semantic-map neighborhood; geometric obstacle layers remain active.
        if self.current_goal_key is not None:
            gx = self.current_goal_key[0] - ox_idx
            gy = self.current_goal_key[1] - oy_idx
            goal_clearance = max(1, int(np.ceil(0.2 / voxel_size)))
            for dx in range(-goal_clearance, goal_clearance + 1):
                for dy in range(-goal_clearance, goal_clearance + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        cost_data[ny * width + nx] = 0

        robot_position = self.get_robot_position()
        if robot_position is not None:
            rx = int(np.floor(robot_position[0] / voxel_size)) - ox_idx
            ry = int(np.floor(robot_position[1] / voxel_size)) - oy_idx
            robot_clearance = max(
                1, int(np.ceil(self.costmap_robot_clearance_m / voxel_size)))
            for dx in range(-robot_clearance, robot_clearance + 1):
                for dy in range(-robot_clearance, robot_clearance + 1):
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        cost_data[ny * width + nx] = 0

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.info.resolution = float(voxel_size)
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = float(self.cost_map_origin_x)
        msg.info.origin.position.y = float(self.cost_map_origin_y)
        msg.data = cost_data
        self.semantic_cost_pub.publish(msg)

        # 同时发到 /map (TRANSIENT_LOCAL)，供老订阅者使用
        map_msg = OccupancyGrid()
        map_msg.header = msg.header
        map_msg.info = msg.info
        map_msg.data = cost_data
        self.map_pub.publish(map_msg)

def main():
    rclpy.init()
    node = GABsvmNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
