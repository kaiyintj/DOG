#!/usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
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

from semantic_mapping.semantic_schema import (
    DEFAULT_CLASSES,
    DEFAULT_CLASS_COLORS,
    DEFAULT_QUERY_MAX_EXTENTS_M,
    DEFAULT_SEMANTIC_COSTS,
    color_membership_score,
    parse_semantic_query,
)
from semantic_mapping.voxel_map import VoxelMap
# ⚠️ 终于加进来的 QoS 协议包！
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy


def segmentation_to_logits(
    class_ids,
    confidences,
    num_classes,
    unknown_probability_floor=0.5,
):
    """Convert a hard segmentation mask and confidence into class logits."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float32)
    if class_ids.shape != confidences.shape:
        raise ValueError('class_ids and confidences must have the same shape')
    if num_classes <= 0:
        raise ValueError('num_classes must be positive')
    if num_classes == 1:
        return np.zeros(class_ids.shape + (1,), dtype=np.float32)

    unknown_index = num_classes - 1
    valid_ids = (class_ids >= 0) & (class_ids < num_classes)
    safe_ids = np.where(valid_ids, class_ids, unknown_index)
    confidence = np.nan_to_num(
        confidences, nan=0.0, posinf=1.0, neginf=0.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    minimum_target = 1.0 / num_classes + 1e-6
    known_target = np.maximum(confidence, minimum_target)
    unknown_floor = float(np.clip(
        unknown_probability_floor, minimum_target, 1.0))
    unknown_target = np.maximum.reduce([
        confidence,
        1.0 - confidence,
        np.full_like(confidence, unknown_floor),
    ])
    target_probability = np.where(
        safe_ids == unknown_index, unknown_target, known_target)
    target_probability = np.clip(target_probability, minimum_target, 1.0 - 1e-6)

    other_probability = (1.0 - target_probability) / (num_classes - 1)
    probabilities = np.repeat(
        other_probability[..., np.newaxis], num_classes, axis=-1)
    np.put_along_axis(
        probabilities,
        safe_ids[..., np.newaxis],
        target_probability[..., np.newaxis],
        axis=-1,
    )
    return np.log(np.clip(probabilities, 1e-6, 1.0)).astype(np.float32)


def normalize_frame_id(frame_id):
    """Return a tf2-compatible frame id without a leading slash."""
    return str(frame_id or '').strip().lstrip('/')


def resolve_pointcloud_frame(configured_frame, message_frame, fallback_frame):
    """Resolve the coordinate frame in which point coordinates are expressed."""
    return (
        normalize_frame_id(configured_frame)
        or normalize_frame_id(message_frame)
        or normalize_frame_id(fallback_frame)
    )


class GABsvmNode(Node):
    def __init__(self):
        super().__init__('ga_bsvm_node')
        self.bridge = CvBridge()

        self.declare_parameter('pointcloud_topic', '/velodyne_points')
        self.declare_parameter('pointcloud_type', 'pointcloud2')
        self.declare_parameter('pointcloud_frame', '')
        self.declare_parameter('tf_buffer_cache_sec', 30.0)
        self.declare_parameter('tf_lookup_timeout_sec', 0.2)
        self.declare_parameter('tf_retry_queue_size', 5)
        self.declare_parameter('tf_retry_period_sec', 0.05)
        self.declare_parameter('tf_retry_max_age_sec', 2.5)
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('image_is_compressed', True)
        self.declare_parameter('image_encoding', 'rgb8')
        self.declare_parameter('imu_topic', '/handsfree/imu')
        self.declare_parameter('semantic_backend', 'clip')
        self.declare_parameter('clip_logits_topic', '/clip_logits')
        self.declare_parameter('clip_features_topic', '/clip_features')
        self.declare_parameter(
            'segformer_class_mask_topic', '/segformer/class_mask')
        self.declare_parameter(
            'segformer_confidence_topic', '/segformer/confidence')
        self.declare_parameter(
            'segformer_source_image_topic', '/segformer/source_image')
        self.declare_parameter('segformer_unknown_probability_floor', 0.5)
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
        self.declare_parameter('num_classes', len(DEFAULT_CLASSES))
        self.declare_parameter('feat_dim', 512)
        self.declare_parameter('vocab', list(DEFAULT_CLASSES))
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('evidence_prior', 1.0)
        self.declare_parameter('evidence_strength', 1.0)
        self.declare_parameter('evidence_decay', 0.995)
        self.declare_parameter('max_frame_evidence', 3.0)
        self.declare_parameter('max_total_evidence', 200.0)
        self.declare_parameter('uncertainty_entropy_weight', 0.7)
        self.declare_parameter('semantic_costs', list(DEFAULT_SEMANTIC_COSTS))
        self.declare_parameter('query_min_weight_sum', 2.0)
        self.declare_parameter('query_min_class_prob', 0.25)
        self.declare_parameter('query_require_class_argmax', True)
        self.declare_parameter('query_min_similarity', 0.18)
        self.declare_parameter('query_feature_weight', 0.7)
        self.declare_parameter('query_class_weight', 0.3)
        self.declare_parameter('query_color_weight', 0.35)
        self.declare_parameter('query_min_color_score', 0.2)
        self.declare_parameter('query_min_color_support_ratio', 0.3)
        self.declare_parameter('query_allow_feature_fallback', True)
        self.declare_parameter('query_approach_radius_m', 1.5)
        self.declare_parameter('query_approach_distance_m', 1.0)
        self.declare_parameter('query_approach_min_distance_m', 0.6)
        self.declare_parameter('query_clearance_radius_m', 0.35)
        self.declare_parameter('query_require_safe_approach', True)
        self.declare_parameter('query_prefer_robot_side', True)
        self.declare_parameter('query_require_robot_side', False)
        self.declare_parameter('query_prefer_direct_path', True)
        self.declare_parameter('query_same_side_min_cosine', 0.0)
        self.declare_parameter('query_path_clearance_radius_m', 0.35)
        self.declare_parameter('query_robot_distance_weight', 0.1)
        self.declare_parameter(
            'query_require_robot_pose_for_approach', False)
        self.declare_parameter('query_cluster_radius_m', 0.45)
        self.declare_parameter('query_cluster_min_voxels', 3)
        self.declare_parameter('query_cluster_min_evidence', 3.0)
        self.declare_parameter('query_cluster_support_weight', 0.08)
        self.declare_parameter('query_distance_weight', 0.01)
        self.declare_parameter('query_max_candidates', 5000)
        self.declare_parameter(
            'query_class_max_extent_m', list(DEFAULT_QUERY_MAX_EXTENTS_M))
        self.declare_parameter('cost_map_size_m', 50.0)
        self.declare_parameter('costmap_min_confidence', 0.15)
        self.declare_parameter('costmap_robot_clearance_m', 0.4)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.2)
        self.declare_parameter('process_every_n_frames', 5)
        self.declare_parameter('segformer_process_every_n_frames', 1)
        self.declare_parameter('entropy_publish_every_n_processed', 2)
        self.declare_parameter('cloud_publish_every_n_processed', 10)
        self.declare_parameter(
            'segformer_cloud_publish_every_n_processed', 1)
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
        self.pointcloud_frame = normalize_frame_id(
            self.get_parameter('pointcloud_frame').value)
        self.tf_buffer_cache_sec = max(
            1.0, float(self.get_parameter('tf_buffer_cache_sec').value))
        self.tf_lookup_timeout_sec = max(
            0.0, float(self.get_parameter('tf_lookup_timeout_sec').value))
        self.tf_retry_queue_size = max(
            0, int(self.get_parameter('tf_retry_queue_size').value))
        self.tf_retry_period_sec = max(
            0.01, float(self.get_parameter('tf_retry_period_sec').value))
        self.tf_retry_max_age_sec = max(
            0.1, float(self.get_parameter('tf_retry_max_age_sec').value))
        self.image_topic = self.get_parameter('image_topic').value
        self.image_is_compressed = bool(self.get_parameter('image_is_compressed').value)
        self.image_encoding = self.get_parameter('image_encoding').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.semantic_backend = str(
            self.get_parameter('semantic_backend').value).strip().lower()
        if self.semantic_backend not in ('clip', 'segformer'):
            raise ValueError(
                'semantic_backend must be either "clip" or "segformer"')
        self.clip_logits_topic = self.get_parameter('clip_logits_topic').value
        self.clip_features_topic = self.get_parameter('clip_features_topic').value
        self.segformer_class_mask_topic = self.get_parameter(
            'segformer_class_mask_topic').value
        self.segformer_confidence_topic = self.get_parameter(
            'segformer_confidence_topic').value
        self.segformer_source_image_topic = self.get_parameter(
            'segformer_source_image_topic').value
        self.segformer_unknown_probability_floor = float(np.clip(
            self.get_parameter('segformer_unknown_probability_floor').value,
            0.0,
            1.0,
        ))
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
        if len(self.vocab) != self.num_classes:
            raise ValueError(
                f'vocab has {len(self.vocab)} classes but num_classes={self.num_classes}')
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
        self.query_require_class_argmax = bool(
            self.get_parameter('query_require_class_argmax').value)
        self.query_min_similarity = float(self.get_parameter('query_min_similarity').value)
        self.query_feature_weight = float(self.get_parameter('query_feature_weight').value)
        self.query_class_weight = float(self.get_parameter('query_class_weight').value)
        self.query_color_weight = float(np.clip(
            self.get_parameter('query_color_weight').value, 0.0, 1.0))
        self.query_min_color_score = float(np.clip(
            self.get_parameter('query_min_color_score').value, 0.0, 1.0))
        self.query_min_color_support_ratio = float(np.clip(
            self.get_parameter('query_min_color_support_ratio').value,
            0.0,
            1.0,
        ))
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
        self.query_prefer_robot_side = bool(
            self.get_parameter('query_prefer_robot_side').value)
        self.query_require_robot_side = bool(
            self.get_parameter('query_require_robot_side').value)
        self.query_prefer_direct_path = bool(
            self.get_parameter('query_prefer_direct_path').value)
        self.query_same_side_min_cosine = float(np.clip(
            self.get_parameter('query_same_side_min_cosine').value,
            -1.0,
            1.0,
        ))
        self.query_path_clearance_radius_m = max(
            0.0,
            float(self.get_parameter(
                'query_path_clearance_radius_m').value),
        )
        self.query_robot_distance_weight = max(
            0.0,
            float(self.get_parameter('query_robot_distance_weight').value),
        )
        self.query_require_robot_pose_for_approach = bool(
            self.get_parameter(
                'query_require_robot_pose_for_approach').value)
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
        if len(self.query_class_max_extent_m) != self.num_classes:
            self.get_logger().warn(
                'query_class_max_extent_m length does not match num_classes; '
                'class extent filtering will use an unrestricted fallback.')
            self.query_class_max_extent_m = [100.0] * self.num_classes
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
        self.segformer_process_every_n_frames = max(
            1,
            int(self.get_parameter(
                'segformer_process_every_n_frames').value),
        )
        self.frame_stride = (
            self.segformer_process_every_n_frames
            if self.semantic_backend == 'segformer'
            else self.process_every_n_frames
        )
        self.entropy_publish_every_n_processed = max(
            1,
            int(self.get_parameter('entropy_publish_every_n_processed').value),
        )
        self.cloud_publish_every_n_processed = max(
            1,
            int(self.get_parameter('cloud_publish_every_n_processed').value),
        )
        self.segformer_cloud_publish_every_n_processed = max(
            1,
            int(self.get_parameter(
                'segformer_cloud_publish_every_n_processed').value),
        )
        self.cloud_publish_stride = (
            self.segformer_cloud_publish_every_n_processed
            if self.semantic_backend == 'segformer'
            else self.cloud_publish_every_n_processed
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

        self.logits_grid = None
        self.feats_grid = None
        self.clip_logits_sub = None
        self.clip_features_sub = None
        self.img_sub = None
        self.segformer_mask_sub = None
        self.segformer_confidence_sub = None
        self.segformer_source_image_sub = None

        if self.pointcloud_type == 'livox_custom':
            pointcloud_msg_type = CustomMsg
        elif self.pointcloud_type == 'pointcloud2':
            pointcloud_msg_type = PointCloud2
        else:
            self.get_logger().warn(
                f'未知 pointcloud_type={self.pointcloud_type}，回退到 pointcloud2。')
            self.pointcloud_type = 'pointcloud2'
            pointcloud_msg_type = PointCloud2

        self.pc_sub = message_filters.Subscriber(
            self,
            pointcloud_msg_type,
            self.pointcloud_topic,
            qos_profile=qos_profile_sensor_data,
        )
        if self.semantic_backend == 'clip':
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
            image_msg_type = CompressedImage if self.image_is_compressed else Image
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
        else:
            self.segformer_mask_sub = message_filters.Subscriber(
                self,
                Image,
                self.segformer_class_mask_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.segformer_confidence_sub = message_filters.Subscriber(
                self,
                Image,
                self.segformer_confidence_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.segformer_source_image_sub = message_filters.Subscriber(
                self,
                Image,
                self.segformer_source_image_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [
                    self.pc_sub,
                    self.segformer_mask_sub,
                    self.segformer_confidence_sub,
                    self.segformer_source_image_sub,
                ],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop,
            )
            self.ts.registerCallback(self.segformer_sync_callback)
        
        self.voxel_map = VoxelMap(
            voxel_size=self.voxel_size,
            K=self.num_classes,
            evidence_prior=self.evidence_prior,
            evidence_strength=self.evidence_strength,
            evidence_decay=self.evidence_decay,
            max_frame_evidence=self.max_frame_evidence,
            max_total_evidence=self.max_total_evidence,
            uncertainty_entropy_weight=self.uncertainty_entropy_weight,
            class_colors=DEFAULT_CLASS_COLORS,
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
        self.query_sub = None
        if self.semantic_backend == 'clip':
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

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=self.tf_buffer_cache_sec))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending_tf_frames = deque()
        self.tf_queue_drop_count = 0
        self.create_timer(
            self.tf_retry_period_sec,
            self.retry_pending_tf_frames,
        )
        self.consecutive_tf_drops = 0
        self.frame_count = 0

        self.current_goal_key = None # 用于记住目标的网格位置
        self._last_approach_was_safe = False
        self.last_query_text = ''
        self.last_query_class_idx = None
        self.last_query_color = None
        self.get_logger().info(
            'GA-BSVM 中枢已启动: '
            f'backend={self.semantic_backend}, '
            f'pointcloud={self.pointcloud_topic} ({self.pointcloud_type}), '
            f'pointcloud_frame={self.pointcloud_frame or "<message header>"}, '
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
        if not self.last_query_text:
            self.get_logger().warn('收到空查询，忽略。')
            return
        (
            self.last_query_class_idx,
            self.last_query_color,
        ) = parse_semantic_query(self.last_query_text, self.vocab)
        parsed_class = (
            self.vocab[self.last_query_class_idx]
            if self.last_query_class_idx is not None
            else 'unresolved'
        )
        self.get_logger().info(
            f'解析查询: query="{self.last_query_text}", '
            f'class={parsed_class}, color={self.last_query_color or "none"}')
        if self.semantic_backend != 'segformer':
            return

        query_class_idx = self.last_query_class_idx
        if query_class_idx is None:
            self.get_logger().warn(
                f'SegFormer 后端仅支持闭集类别 {self.vocab}，'
                f'无法解析 query="{self.last_query_text}"。')
            return
        if query_class_idx == self.num_classes - 1:
            self.get_logger().warn('unknown background 不能作为导航目标。')
            return
        self.query_class_cb(query_class_idx, self.last_query_color)

    def get_query_class_index(self):
        if not self.last_query_text:
            return None
        query_class_idx, query_color = parse_semantic_query(
            self.last_query_text, self.vocab)
        self.last_query_class_idx = query_class_idx
        self.last_query_color = query_color
        return query_class_idx

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

    def score_query_color(self, voxel, query_color):
        """Return a named-color score, or ``None`` when no color is requested."""
        if query_color is None:
            return None
        return color_membership_score(voxel.get('color_rgb'), query_color)

    def select_query_cluster(
        self,
        candidates,
        query_class_idx=None,
        query_color=None,
    ):
        """Select one spatial instance and apply color at cluster level.

        Forming the semantic instance before checking color prevents a small
        white accessory, such as a headlight on a red vehicle, from becoming a
        standalone ``white car`` target.
        """
        self._last_cluster_selection_diagnostics = {
            'cluster_count': 0,
            'color_rejected_clusters': 0,
            'max_cluster_color_score': 0.0,
            'max_cluster_color_support_ratio': 0.0,
        }
        if not candidates:
            return None

        color_weight = float(getattr(self, 'query_color_weight', 0.35))

        def candidate_rank(item):
            semantic_score = float(item['score'])
            if query_color is None:
                return semantic_score
            return (
                (1.0 - color_weight) * semantic_score
                + color_weight * float(item.get('color_score', 0.0))
            )

        candidates = sorted(
            candidates, key=candidate_rank, reverse=True
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

        self._last_cluster_selection_diagnostics['cluster_count'] = len(clusters)
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
            selection_score = semantic_score
            cluster_color_score = 0.0
            color_support_ratio = 0.0
            mean_color_rgb = None
            if query_color is not None:
                color_scores = np.asarray([
                    float(item.get('color_score', 0.0)) for item in cluster
                ], dtype=np.float64)
                cluster_color_score = float(np.average(
                    color_scores, weights=evidence_weights))
                color_support_ratio = float(np.average(
                    color_scores >= self.query_min_color_score,
                    weights=evidence_weights,
                ))
                diagnostics = self._last_cluster_selection_diagnostics
                diagnostics['max_cluster_color_score'] = max(
                    diagnostics['max_cluster_color_score'],
                    cluster_color_score,
                )
                diagnostics['max_cluster_color_support_ratio'] = max(
                    diagnostics['max_cluster_color_support_ratio'],
                    color_support_ratio,
                )
                if (
                    cluster_color_score < self.query_min_color_score
                    or color_support_ratio
                    < getattr(self, 'query_min_color_support_ratio', 0.3)
                ):
                    diagnostics['color_rejected_clusters'] += 1
                    continue
                selection_score = (
                    (1.0 - color_weight) * semantic_score
                    + color_weight * cluster_color_score
                )

                valid_colors = []
                valid_color_weights = []
                for item, item_weight in zip(cluster, evidence_weights):
                    color_rgb = item.get('color_rgb')
                    if color_rgb is None:
                        continue
                    color_rgb = np.asarray(color_rgb, dtype=np.float64).reshape(-1)
                    if color_rgb.size == 3 and np.all(np.isfinite(color_rgb)):
                        valid_colors.append(color_rgb)
                        valid_color_weights.append(item_weight)
                if valid_colors:
                    mean_color_rgb = np.average(
                        np.asarray(valid_colors),
                        axis=0,
                        weights=np.asarray(valid_color_weights),
                    ).astype(np.float32)

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
                selection_score
                + self.query_cluster_support_weight * support
                - self.query_distance_weight * distance
            )
            if utility > best_utility:
                best_utility = utility
                representative = max(cluster, key=lambda item: item['score'])
                best_cluster = {
                    'key': representative['key'],
                    'pos': centroid.astype(np.float32),
                    'score': selection_score,
                    'semantic_score': semantic_score,
                    'similarity': max(item['similarity'] for item in cluster),
                    'class_probability': max(
                        item['class_probability'] for item in cluster),
                    'color_score': cluster_color_score,
                    'color_support_ratio': color_support_ratio,
                    'color_rgb': mean_color_rgb,
                    'voxel_count': len(cluster),
                    'evidence': total_evidence,
                    'utility': utility,
                }
        return best_cluster

    @staticmethod
    def _segment_has_clearance(
        start,
        end,
        obstacle_tree,
        clearance_radius,
        sample_step,
    ):
        """Return whether a straight segment stays clear of semantic obstacles."""
        if obstacle_tree is None or clearance_radius <= 0.0:
            return True
        start = np.asarray(start, dtype=np.float64)[:2]
        end = np.asarray(end, dtype=np.float64)[:2]
        distance = float(np.linalg.norm(end - start))
        if distance <= 1e-9:
            samples = start.reshape(1, 2)
        else:
            step = max(float(sample_step), 0.02)
            sample_count = max(2, int(np.ceil(distance / step)) + 1)
            fractions = np.linspace(0.0, 1.0, sample_count)[:, np.newaxis]
            samples = start + fractions * (end - start)
        nearest_distances, _ = obstacle_tree.query(samples, k=1)
        return bool(np.all(
            np.asarray(nearest_distances) >= float(clearance_radius)))

    def find_approach_goal(self, object_pos, query_class_idx):
        """Find a supported, robot-side road voxel with a clear approach."""
        self._last_approach_was_safe = False
        is_traversable_query = (
            query_class_idx is not None
            and self.semantic_cost_dict.get(query_class_idx, 100) < 100
        )
        if is_traversable_query:
            key = self.voxel_map.get_voxel_indices(object_pos)
            self._last_approach_was_safe = True
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
        if (
            robot_position is None
            and getattr(
                self,
                'query_require_robot_pose_for_approach',
                False,
            )
        ):
            self.get_logger().warn(
                '无法获取机器人位姿，拒绝选择可能位于物体另一侧的接近点。')
            return None, None

        best = None
        best_rank = None
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
            same_side = True
            direct_path_clear = True
            if robot_position is not None:
                robot_distance = float(
                    np.linalg.norm(position[:2] - robot_position[:2]))
                height_cost = abs(float(position[2] - robot_position[2]))
                robot_direction = robot_position[:2] - object_pos[:2]
                candidate_direction = position[:2] - object_pos[:2]
                direction_norm = float(
                    np.linalg.norm(robot_direction)
                    * np.linalg.norm(candidate_direction)
                )
                if direction_norm > 1e-9:
                    side_cosine = float(np.dot(
                        robot_direction,
                        candidate_direction,
                    ) / direction_norm)
                    same_side = side_cosine >= getattr(
                        self, 'query_same_side_min_cosine', 0.0)
                if (
                    getattr(self, 'query_require_robot_side', False)
                    and not same_side
                ):
                    continue
                path_clearance = float(getattr(
                    self,
                    'query_path_clearance_radius_m',
                    self.query_clearance_radius_m,
                ))
                sample_step = min(
                    max(path_clearance * 0.5, 0.05),
                    max(float(self.voxel_map.voxel_size), 0.05),
                )
                direct_path_clear = self._segment_has_clearance(
                    robot_position[:2],
                    position[:2],
                    obstacle_tree,
                    path_clearance,
                    sample_step,
                )
            cost = (
                abs(distance_to_object - self.query_approach_distance_m)
                + getattr(self, 'query_robot_distance_weight', 0.1)
                * robot_distance
                + 0.2 * height_cost
                - 0.1 * confidence
            )
            preference_rank = (
                int(
                    getattr(self, 'query_prefer_robot_side', True)
                    and not same_side
                ),
                int(
                    getattr(self, 'query_prefer_direct_path', True)
                    and not direct_path_clear
                ),
                cost,
            )
            if best_rank is None or preference_rank < best_rank:
                best_rank = preference_rank
                best = (
                    key,
                    position,
                    same_side,
                    direct_path_clear,
                )

        if best is not None:
            self._last_approach_was_safe = True
            self.get_logger().info(
                f'安全接近点: x={best[1][0]:.2f}, y={best[1][1]:.2f}, '
                f'机器人同侧={best[2]}, 直线路径净空={best[3]}')
            return best[0], best[1]
        if self.query_require_safe_approach:
            return None, None

        # Simulation fallback: use a line-of-sight standoff point instead of
        # navigating into the object center. Nav2's geometric layers still
        # perform the final collision check.
        if robot_position is None:
            self.get_logger().warn(
                '没有安全road接近点且机器人位姿不可用，拒绝回退到物体中心。')
            return None, None
        fallback_position = np.array(object_pos, dtype=np.float32)
        direction = robot_position[:2] - object_pos[:2]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return None, None
        fallback_position[:2] = (
            object_pos[:2]
            + direction / direction_norm * self.query_approach_distance_m
        )
        fallback_position[2] = robot_position[2]
        fallback_clearance = float(getattr(
            self,
            'query_path_clearance_radius_m',
            self.query_clearance_radius_m,
        ))
        if obstacle_tree is not None:
            if obstacle_tree.query_ball_point(
                fallback_position[:2],
                fallback_clearance,
            ):
                self.get_logger().warn(
                    '视线方向回退点与语义障碍冲突，拒绝发布导航目标。')
                return None, None
        self.get_logger().warn(
            '没有找到满足语义约束的road体素，使用视线方向接近点；'
            '该回退仅应用于仿真配置。')
        return (
            self.voxel_map.get_voxel_indices(fallback_position),
            fallback_position,
        )

    def select_class_query_target(self, query_class_idx, query_color=None):
        """Select a target using categorical and optional color evidence."""
        candidates = []
        candidate_count = 0
        class_rejected_count = 0
        color_rejected_count = 0
        color_evaluated_count = 0
        max_color_score = 0.0
        best_color_rgb = None
        max_evidence_probability = 0.0
        max_posterior_probability = 0.0
        dominant_query_voxels = 0
        dominant_counts = np.zeros(self.voxel_map.K, dtype=np.int64)
        require_argmax = getattr(
            self, 'query_require_class_argmax', True)
        for key, voxel in self.voxel_map.voxels.items():
            if voxel['weight_sum'] < self.query_min_weight_sum:
                continue
            candidate_count += 1
            posterior_probabilities = self.voxel_map.get_probabilities(key)
            evidence_probabilities = (
                self.voxel_map.get_evidence_probabilities(key))
            dominant_class = int(np.argmax(evidence_probabilities))
            dominant_counts[dominant_class] += 1
            if dominant_class == query_class_idx:
                dominant_query_voxels += 1
            class_probability = float(
                evidence_probabilities[query_class_idx])
            posterior_probability = float(
                posterior_probabilities[query_class_idx])
            max_evidence_probability = max(
                max_evidence_probability, class_probability)
            max_posterior_probability = max(
                max_posterior_probability, posterior_probability)
            if (
                class_probability < self.query_min_class_prob
                or (require_argmax and dominant_class != query_class_idx)
            ):
                class_rejected_count += 1
                continue
            color_score = self.score_query_color(voxel, query_color)
            if color_score is not None:
                color_evaluated_count += 1
                if color_score > max_color_score:
                    max_color_score = float(color_score)
                    color_rgb = voxel.get('color_rgb')
                    best_color_rgb = (
                        None if color_rgb is None
                        else np.asarray(color_rgb, dtype=np.float32).copy()
                    )
            if (
                color_score is not None
                and color_score < self.query_min_color_score
            ):
                color_rejected_count += 1
            score = class_probability
            candidates.append({
                'key': key,
                'pos': voxel['pos'],
                'similarity': 0.0,
                'class_probability': class_probability,
                'posterior_probability': posterior_probability,
                'color_score': color_score or 0.0,
                'color_rgb': voxel.get('color_rgb'),
                'score': score,
                'evidence': voxel['weight_sum'],
            })
        selected = self.select_query_cluster(
            candidates,
            query_class_idx=query_class_idx,
            query_color=query_color,
        )
        self._last_class_query_diagnostics = {
            'max_evidence_probability': max_evidence_probability,
            'max_posterior_probability': max_posterior_probability,
            'dominant_query_voxels': dominant_query_voxels,
            'passed_class_gate': candidate_count - class_rejected_count,
            'dominant_counts': dominant_counts,
            'query_color': query_color,
            'color_evaluated_count': color_evaluated_count,
            'color_rejected_count': color_rejected_count,
            'max_color_score': max_color_score,
            'best_color_rgb': best_color_rgb,
        }
        return (
            selected,
            candidate_count,
            class_rejected_count,
            color_rejected_count,
        )

    def format_class_query_diagnostics(self):
        diagnostics = getattr(self, '_last_class_query_diagnostics', None)
        if diagnostics is None:
            return ''
        counts = diagnostics['dominant_counts']
        top_indices = np.argsort(counts)[::-1][:3]
        top_classes = ', '.join(
            f'{self.vocab[index]}:{int(counts[index])}'
            for index in top_indices
            if counts[index] > 0
        ) or 'none'
        query_color = diagnostics['query_color']
        if query_color is None:
            color_status = '颜色筛选=未请求'
        elif diagnostics['passed_class_gate'] == 0:
            color_status = '颜色筛选=未执行(类别门控无候选)'
        else:
            best_color_rgb = diagnostics['best_color_rgb']
            if best_color_rgb is not None and np.max(best_color_rgb) <= 1.0:
                best_color_rgb = best_color_rgb * 255.0
            rgb_text = (
                'none'
                if best_color_rgb is None
                else ','.join(
                    str(int(np.clip(round(value), 0, 255)))
                    for value in best_color_rgb
                )
            )
            color_status = (
                f'颜色已评估={diagnostics["color_evaluated_count"]}, '
                f'低颜色分数体素={diagnostics["color_rejected_count"]}, '
                f'最高颜色分数={diagnostics["max_color_score"]:.3f}, '
                f'最佳RGB=[{rgb_text}]'
            )
            cluster_diagnostics = getattr(
                self, '_last_cluster_selection_diagnostics', {})
            color_status += (
                f', 颜色拒绝簇='
                f'{cluster_diagnostics.get("color_rejected_clusters", 0)}, '
                f'最高簇颜色均值='
                f'{cluster_diagnostics.get("max_cluster_color_score", 0.0):.3f}, '
                f'最高簇颜色支持率='
                f'{cluster_diagnostics.get("max_cluster_color_support_ratio", 0.0):.1%}'
            )
        return (
            f'最高观测证据概率={diagnostics["max_evidence_probability"]:.3f}, '
            f'最高原始后验={diagnostics["max_posterior_probability"]:.3f}, '
            f'查询类为主类别体素={diagnostics["dominant_query_voxels"]}, '
            f'通过类别门控={diagnostics["passed_class_gate"]}, '
            f'{color_status}, '
            f'主类别Top3=[{top_classes}]'
        )

    def _publish_query_goal(self, selected, query_class_idx, query_source):
        object_pos = selected['pos']
        target = PoseStamped()
        target.header.frame_id = self.odom_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = float(object_pos[0])
        target.pose.position.y = float(object_pos[1])
        target.pose.position.z = float(object_pos[2])
        target.pose.orientation.w = 1.0
        self.query_target_pub.publish(target)

        goal_key, goal_pos = self.find_approach_goal(
            object_pos, query_class_idx)
        if goal_pos is None:
            self.get_logger().warn(
                f'找到 query="{self.last_query_text}"，但附近没有满足间距和净空要求的road接近点。')
            return False
        self.current_goal_key = (
            goal_key if self._last_approach_was_safe else None)

        self.get_logger().info(
            f'🎯 [导航触发] backend={query_source}, '
            f'query="{self.last_query_text}", score={selected["score"]:.3f}, '
            f'sim={selected["similarity"]:.3f}, '
            f'class_evidence_prob={selected["class_probability"]:.3f}, '
            f'color={self.last_query_color or "none"}, '
            f'color_score={selected.get("color_score", 0.0):.3f}, '
            f'color_support={selected.get("color_support_ratio", 0.0):.1%}, '
            f'cluster={selected["voxel_count"]} voxels, '
            f'evidence={selected["evidence"]:.1f}, '
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
        return True

    def query_class_cb(self, query_class_idx, query_color=None):
        (
            selected,
            candidate_count,
            class_rejected_count,
            color_rejected_count,
        ) = self.select_class_query_target(query_class_idx, query_color)
        if selected is None:
            self.get_logger().warn(
                f'🔍 未找到合适目标。backend=segformer, '
                f'query="{self.last_query_text}", 候选体素={candidate_count}, '
                f'类别阈值过滤={class_rejected_count}, '
                f'{self.format_class_query_diagnostics()}')
            return
        self._publish_query_goal(
            selected, query_class_idx, query_source='segformer')

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
        color_rejected_count = 0

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
            color_score = self.score_query_color(
                voxel, self.last_query_color)
            if (
                color_score is not None
                and color_score < self.query_min_color_score
            ):
                color_rejected_count += 1
            class_prob = 0.0
            score = sim
            candidate = {
                'key': key,
                'pos': voxel['pos'],
                'similarity': sim,
                'class_probability': class_prob,
                'color_score': color_score or 0.0,
                'color_rgb': voxel.get('color_rgb'),
                'score': score,
                'evidence': voxel['weight_sum'],
            }
            feature_candidates.append(candidate)

            if query_class_idx is not None:
                probs = self.voxel_map.get_evidence_probabilities(key)
                class_prob = float(probs[query_class_idx])
                if (
                    class_prob < self.query_min_class_prob
                    or (
                        self.query_require_class_argmax
                        and int(np.argmax(probs)) != query_class_idx
                    )
                ):
                    class_rejected_count += 1
                    continue
                score = (
                    self.query_feature_weight * sim
                    + self.query_class_weight * class_prob
                )
                candidate = dict(candidate)
                candidate['class_probability'] = class_prob
                candidate['score'] = score
            accepted_candidates.append(candidate)

        used_fallback = False
        selected = self.select_query_cluster(
            accepted_candidates,
            query_class_idx=query_class_idx,
            query_color=self.last_query_color,
        )
        if selected is None and self.query_allow_feature_fallback:
            used_fallback = True
            selected = self.select_query_cluster(
                feature_candidates,
                query_class_idx=query_class_idx,
                query_color=self.last_query_color,
            )

        if selected is not None:
            if used_fallback:
                self.get_logger().warn(
                    f'query="{self.last_query_text}" 没有体素通过类别概率阈值 '
                    f'({class_rejected_count}/{candidate_count} rejected, '
                    f'min_class_prob={self.query_min_class_prob:.2f})，'
                    '改用 CLIP 特征相似度兜底。')
            self._publish_query_goal(
                selected, query_class_idx, query_source='clip')
        else:
            self.get_logger().warn(
                f'🔍 未找到合适目标。query="{self.last_query_text}", '
                f'候选体素={candidate_count}, 类别阈值过滤={class_rejected_count}, '
                f'低颜色分数体素={color_rejected_count}')

    def _should_process_frame(self):
        self.frame_count += 1
        return self.frame_count % self.frame_stride == 0

    def sync_callback(self, pc_msg, img_msg):
        """Fuse the legacy CLIP grid with a synchronized point cloud."""
        if not self._should_process_frame():
            return
        if self.logits_grid is None:
            return
        try:
            if self.image_is_compressed:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(
                    img_msg, 'rgb8')
            else:
                cv_img = self.bridge.imgmsg_to_cv2(
                    img_msg, 'rgb8')
            height, width = cv_img.shape[:2]
            patch_h = height // self.grid_rows
            patch_w = width // self.grid_cols
            if patch_h <= 0 or patch_w <= 0:
                self.get_logger().warn(
                    f'图像尺寸过小，无法切成 {self.grid_rows}x{self.grid_cols}: '
                    f'{width}x{height}')
                return

            def semantic_lookup(valid_u, valid_v):
                grid_r = np.clip(
                    valid_v // patch_h, 0, self.grid_rows - 1)
                grid_c = np.clip(
                    valid_u // patch_w, 0, self.grid_cols - 1)
                logits = self.logits_grid[grid_r, grid_c]
                features = None
                if self.feats_grid is not None:
                    features = self.feats_grid[grid_r, grid_c]
                colors = cv_img[valid_v, valid_u, :3]
                return logits, features, colors

            self._process_semantic_frame(
                pc_msg, height, width, semantic_lookup)
        except Exception as exc:
            self.get_logger().error(f'CLIP 投影融合出错: {exc}')

    def segformer_sync_callback(
        self,
        pc_msg,
        class_mask_msg,
        confidence_msg,
        source_image_msg,
    ):
        """Fuse synchronized SegFormer pixels and point-cloud returns."""
        if not self._should_process_frame():
            return
        try:
            class_mask = self.bridge.imgmsg_to_cv2(
                class_mask_msg, desired_encoding='mono8')
            confidence = self.bridge.imgmsg_to_cv2(
                confidence_msg, desired_encoding='32FC1')
            source_image = self.bridge.imgmsg_to_cv2(
                source_image_msg, desired_encoding='rgb8')
            class_mask = np.asarray(class_mask)
            confidence = np.asarray(confidence)
            source_image = np.asarray(source_image)
            if class_mask.ndim == 3:
                class_mask = class_mask[..., 0]
            if confidence.ndim == 3:
                confidence = confidence[..., 0]
            if class_mask.shape != confidence.shape:
                self.get_logger().warn(
                    'SegFormer class_mask 和 confidence 尺寸不一致: '
                    f'{class_mask.shape} != {confidence.shape}')
                return
            height, width = class_mask.shape
            if source_image.shape[:2] != (height, width):
                self.get_logger().warn(
                    'SegFormer source image 尺寸不一致: '
                    f'{source_image.shape[:2]} != {(height, width)}')
                return

            def semantic_lookup(valid_u, valid_v):
                logits = segmentation_to_logits(
                    class_mask[valid_v, valid_u],
                    confidence[valid_v, valid_u],
                    self.num_classes,
                    self.segformer_unknown_probability_floor,
                )
                colors = source_image[valid_v, valid_u, :3]
                return logits, None, colors

            self._process_semantic_frame(
                pc_msg, height, width, semantic_lookup)
        except Exception as exc:
            self.get_logger().error(f'SegFormer 投影融合出错: {exc}')

    def _process_semantic_frame(
        self,
        pc_msg,
        height,
        width,
        semantic_lookup,
    ):
        """Project one synchronized frame and fuse it once historical TF exists."""
        motion_reliability, angular_rms, acceleration_deviation = (
            self.compute_motion_reliability(pc_msg.header.stamp))

        if self.pointcloud_type == 'livox_custom':
            points_3d = self.parse_livox_msg(pc_msg)
        else:
            points_3d = self.parse_pointcloud2_msg(pc_msg)
        if len(points_3d) == 0:
            return
        total_points = len(points_3d)

        valid_mask, u, v = self.project_points(points_3d, height, width)
        valid_points = points_3d[valid_mask]
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        if len(valid_points) == 0:
            return

        sensor_ranges = np.linalg.norm(valid_points, axis=1)
        (
            per_point_logits,
            per_point_feats,
            per_point_colors,
        ) = semantic_lookup(valid_u, valid_v)
        per_point_logits = np.asarray(per_point_logits, dtype=np.float32)
        if per_point_logits.shape != (len(valid_points), self.num_classes):
            raise ValueError(
                'semantic logits shape mismatch: '
                f'{per_point_logits.shape} != '
                f'({len(valid_points)}, {self.num_classes})')

        tree = cKDTree(valid_points)
        neighbors = tree.query_ball_point(valid_points, r=0.3)
        density = np.asarray([len(neighbor) for neighbor in neighbors])
        density_reliability = 1.0 - np.exp(
            -density / max(self.density_scale, 1e-6))
        range_reliability = np.exp(
            -np.square(sensor_ranges / max(self.range_scale_m, 1e-6)))

        normalized_u = (valid_u - 0.5 * width) / max(0.5 * width, 1.0)
        normalized_v = (valid_v - 0.5 * height) / max(0.5 * height, 1.0)
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

        source_frame = resolve_pointcloud_frame(
            self.pointcloud_frame,
            pc_msg.header.frame_id,
            self.base_frame,
        )
        target_frame = normalize_frame_id(self.odom_frame)
        stamp = Time.from_msg(pc_msg.header.stamp)
        frame_data = {
            'points': valid_points,
            'reliability': reliability,
            'logits': per_point_logits,
            'features': per_point_feats,
            'colors': per_point_colors,
            'stamp_msg': pc_msg.header.stamp,
            'total_points': total_points,
            'processed_count': self.frame_count // self.frame_stride,
            'motion_reliability': motion_reliability,
            'angular_rms': angular_rms,
            'acceleration_deviation': acceleration_deviation,
        }

        if source_frame != target_frame and stamp.nanoseconds == 0:
            self.consecutive_tf_drops += 1
            self.get_logger().warn(
                '丢弃无时间戳点云，避免用最新位姿污染语义地图: '
                f'source={source_frame}, target={target_frame}')
            return False

        try:
            transformed_points = self._transform_semantic_points(
                valid_points,
                target_frame,
                source_frame,
                stamp,
            )
        except Exception as exc:
            self._queue_pending_tf_frame(
                frame_data,
                target_frame,
                source_frame,
                stamp,
                exc,
            )
            return False

        self._fuse_projected_semantic_frame(
            frame_data,
            transformed_points,
        )
        if self.consecutive_tf_drops:
            self.get_logger().info(
                '点云历史TF已恢复，等待帧重新进入融合。')
            self.consecutive_tf_drops = 0
        return True

    def _transform_semantic_points(
        self,
        points,
        target_frame,
        source_frame,
        stamp,
    ):
        """Transform points at their sensor timestamp without a latest-TF fallback."""
        if source_frame == target_frame:
            return points
        # Waiting inside a callback prevents a single-threaded executor from
        # receiving the future TF. Missing transforms are retried by a timer.
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=0.0),
        )
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        quaternion = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        rotation = R.from_quat(quaternion).as_matrix()
        return (rotation @ points.T).T + translation

    def _queue_pending_tf_frame(
        self,
        frame_data,
        target_frame,
        source_frame,
        stamp,
        error,
    ):
        """Queue a bounded semantic frame until its historical TF arrives."""
        self.consecutive_tf_drops += 1
        if self.tf_retry_queue_size <= 0:
            if (
                self.consecutive_tf_drops <= 3
                or self.consecutive_tf_drops % 20 == 0
            ):
                self.get_logger().warn(
                    '历史TF不可用且重试队列已禁用，丢弃语义帧: '
                    f'{target_frame}<-{source_frame}, error={error}')
            return

        if len(self.pending_tf_frames) >= self.tf_retry_queue_size:
            self.pending_tf_frames.popleft()
            self.tf_queue_drop_count += 1
        self.pending_tf_frames.append({
            'queued_at_ns': self.get_clock().now().nanoseconds,
            'target_frame': target_frame,
            'source_frame': source_frame,
            'stamp': stamp,
            'frame_data': frame_data,
        })
        if (
            self.consecutive_tf_drops <= 3
            or self.consecutive_tf_drops % 20 == 0
        ):
            stamp_sec = stamp.nanoseconds / 1e9
            self.get_logger().warn(
                '点云时刻的历史TF尚未到达，语义帧已进入有界重试队列: '
                f'{target_frame}<-{source_frame} at {stamp_sec:.6f}s, '
                f'queue={len(self.pending_tf_frames)}/'
                f'{self.tf_retry_queue_size}, error={error}')

    def retry_pending_tf_frames(self):
        """Retry one queued frame per timer tick after TF callbacks can run."""
        if not self.pending_tf_frames:
            return
        pending = self.pending_tf_frames.popleft()
        age_sec = max(
            0.0,
            (self.get_clock().now().nanoseconds - pending['queued_at_ns'])
            / 1e9,
        )
        if age_sec > self.tf_retry_max_age_sec:
            self.tf_queue_drop_count += 1
            self.get_logger().warn(
                '语义帧等待历史TF超时，已安全丢弃: '
                f'age={age_sec:.2f}s, total_dropped='
                f'{self.tf_queue_drop_count}')
            return
        try:
            transformed_points = self._transform_semantic_points(
                pending['frame_data']['points'],
                pending['target_frame'],
                pending['source_frame'],
                pending['stamp'],
            )
        except Exception:
            # Rotate an unavailable historical transform to the back.  A
            # permanently missing old frame must not block newer frames whose
            # transforms have already arrived.
            self.pending_tf_frames.append(pending)
            return

        self._fuse_projected_semantic_frame(
            pending['frame_data'],
            transformed_points,
        )
        if self.consecutive_tf_drops:
            self.get_logger().info(
                '历史TF已到达，排队语义帧已恢复融合: '
                f'remaining={len(self.pending_tf_frames)}')
            self.consecutive_tf_drops = 0

    def _fuse_projected_semantic_frame(self, frame_data, valid_points):
        """Fuse an already projected frame in the odometry coordinate frame."""
        self.voxel_map.update(
            points=valid_points,
            reliability=frame_data['reliability'],
            logits=frame_data['logits'],
            features=frame_data['features'],
            colors=frame_data['colors'],
        )

        processed_count = frame_data['processed_count']
        if processed_count % self.entropy_publish_every_n_processed == 0:
            self.publish_entropy_data()

        if processed_count % self.cloud_publish_stride == 0:
            sem_pts, unc_pts = self.voxel_map.get_visualization_clouds()
            if len(sem_pts) > 0:
                uncertainty_values = np.asarray(
                    [point[3] for point in unc_pts], dtype=np.float32) / 255.0
                mean_uncertainty = float(np.mean(uncertainty_values))
                high_uncertainty_ratio = float(np.mean(
                    uncertainty_values >= 0.75))
                header = Header()
                header.stamp = frame_data['stamp_msg']
                header.frame_id = self.odom_frame
                self.pub_semantic.publish(
                    self.create_cloud_msg(header, sem_pts))
                self.pub_uncertainty.publish(
                    self.create_cloud_msg(header, unc_pts))
                self.get_logger().info(
                    f'投影并融合 {len(valid_points)}/'
                    f'{frame_data["total_points"]} 个3D点 '
                    f'({len(valid_points) / frame_data["total_points"]:.1%}): '
                    f'backend={self.semantic_backend}, '
                    f'uncertainty_mean={mean_uncertainty:.2f}, '
                    f'high_uncertainty={high_uncertainty_ratio:.1%}, '
                    f'motion={frame_data["motion_reliability"]:.2f}, '
                    f'angular_rms={frame_data["angular_rms"]:.2f}rad/s, '
                    f'accel_dev='
                    f'{frame_data["acceleration_deviation"]:.2f}m/s^2')

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
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
