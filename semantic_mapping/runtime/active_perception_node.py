#!/usr/bin/env python3
"""Apply fail-closed semantic-risk speed modulation to navigation commands."""

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import threading
from scipy.spatial import cKDTree

from semantic_mapping.runtime.semantic_profile_ros import load_semantic_contract


def stamp_to_seconds(stamp):
    """Convert a ROS stamp to seconds, rejecting absent/invalid stamps."""
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        return None
    if sec == 0 and nanosec == 0:
        return None
    return sec + nanosec * 1e-9


def sample_age_seconds(now_sec, sample_stamp_sec):
    """Return a finite sample age, or infinity when it cannot be trusted."""
    if sample_stamp_sec is None:
        return math.inf
    try:
        now_sec = float(now_sec)
        sample_stamp_sec = float(sample_stamp_sec)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(now_sec) or not math.isfinite(sample_stamp_sec):
        return math.inf
    return now_sec - sample_stamp_sec


def is_sample_fresh(
    now_sec,
    sample_stamp_sec,
    max_age_sec,
    future_tolerance_sec=0.1,
):
    """Accept only samples inside a bounded past/future time window."""
    try:
        max_age_sec = float(max_age_sec)
        future_tolerance_sec = float(future_tolerance_sec)
    except (TypeError, ValueError):
        return False
    if (not math.isfinite(max_age_sec) or max_age_sec <= 0.0
            or not math.isfinite(future_tolerance_sec)
            or future_tolerance_sec < 0.0):
        return False
    age_sec = sample_age_seconds(now_sec, sample_stamp_sec)
    return -future_tolerance_sec <= age_sec <= max_age_sec


def stale_required_inputs(
    now_sec,
    path_stamp_sec,
    cloud_stamp_sec,
    imu_stamp_sec,
    path_max_age_sec,
    cloud_max_age_sec,
    imu_max_age_sec,
    future_tolerance_sec=0.1,
    require_imu=True,
):
    """Return required inputs that are absent, stale, or future-dated."""
    stale = []
    checks = (
        ('path', path_stamp_sec, path_max_age_sec),
        ('entropy_cloud', cloud_stamp_sec, cloud_max_age_sec),
    )
    for name, stamp_sec, max_age_sec in checks:
        if not is_sample_fresh(
                now_sec, stamp_sec, max_age_sec, future_tolerance_sec):
            stale.append(name)
    if require_imu and not is_sample_fresh(
            now_sec, imu_stamp_sec, imu_max_age_sec, future_tolerance_sec):
        stale.append('imu')
    return tuple(stale)


def command_watchdog_expired(now_sec, last_command_sec, timeout_sec):
    """Fail closed when no recent command receipt exists on a steady clock."""
    try:
        now_sec = float(now_sec)
        timeout_sec = float(timeout_sec)
    except (TypeError, ValueError):
        return True
    if (not math.isfinite(now_sec) or not math.isfinite(timeout_sec)
            or timeout_sec <= 0.0 or last_command_sec is None):
        return True
    try:
        last_command_sec = float(last_command_sec)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(last_command_sec):
        return True
    age_sec = now_sec - last_command_sec
    # A backwards steady-clock jump should never happen; treat it as unsafe.
    return age_sec < 0.0 or age_sec >= timeout_sec


def make_imu_sensor_qos():
    """Return the ROS sensor-data QoS used for the IMU subscription."""
    return qos_profile_sensor_data


def resolve_h_max(configured_h_max, num_classes):
    """Return an explicit entropy bound or derive it from the class count."""
    configured_h_max = float(configured_h_max)
    if math.isfinite(configured_h_max) and configured_h_max > 0.0:
        return configured_h_max

    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError(
            'num_classes must be at least 2 when h_max is automatic')
    return math.log(num_classes)


def is_valid_frame_id(frame_id):
    """Validate a TF frame without silently normalising an invalid name."""
    if not isinstance(frame_id, str) or not frame_id:
        return False
    if frame_id != frame_id.strip() or frame_id.startswith('/'):
        return False
    return not any(character.isspace() for character in frame_id)


def transform_xyz_points(points, translation, rotation):
    """Apply a geometry_msgs-style rigid transform to an N x 3 point array."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3
            or not np.all(np.isfinite(points))):
        raise ValueError('points must be a finite N x 3 array')

    translation = np.asarray(translation, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError('translation must contain three finite values')
    if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
        raise ValueError('rotation must contain four finite quaternion values')

    quaternion_norm = float(np.linalg.norm(rotation))
    if quaternion_norm <= 1e-12:
        raise ValueError('rotation quaternion has zero norm')
    x, y, z, w = rotation / quaternion_norm
    rotation_matrix = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)
    return points @ rotation_matrix.T + translation


def aggregate_uncertainty(values, percentile=90.0, percentile_weight=0.35):
    """Blend mean and upper-tail uncertainty to avoid hiding local hazards."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('at least one finite uncertainty value is required')
    percentile = float(np.clip(percentile, 0.0, 100.0))
    percentile_weight = float(np.clip(percentile_weight, 0.0, 1.0))
    mean_value = float(np.mean(values))
    tail_value = float(np.percentile(values, percentile))
    return ((1.0 - percentile_weight) * mean_value
            + percentile_weight * tail_value)


def rate_limit_scale(
    current_scale,
    target_scale,
    now_sec,
    last_update_sec,
    max_scale_rate_per_sec,
):
    """Rate-limit a scale using elapsed time and handle clock resets."""
    target_scale = float(target_scale)
    now_sec = float(now_sec)
    if not math.isfinite(target_scale) or not math.isfinite(now_sec):
        raise ValueError('target_scale and now_sec must be finite')
    if current_scale is None or last_update_sec is None:
        return target_scale, now_sec

    current_scale = float(current_scale)
    last_update_sec = float(last_update_sec)
    if not math.isfinite(current_scale) or not math.isfinite(last_update_sec):
        return target_scale, now_sec

    dt = now_sec - last_update_sec
    if dt <= 0.0:
        # Simulation time can jump backwards. Keep the current output for this
        # sample and rebase the limiter instead of applying a discontinuity.
        return current_scale, now_sec

    max_rate = float(max_scale_rate_per_sec)
    if not math.isfinite(max_rate) or max_rate <= 0.0:
        return target_scale, now_sec
    max_delta = max_rate * dt
    delta = float(np.clip(target_scale - current_scale, -max_delta, max_delta))
    return current_scale + delta, now_sec


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
        self.declare_parameter('ontology_profile', 'outdoor13')
        self.profile = load_semantic_contract({
            'ontology_profile': self.get_parameter('ontology_profile').value,
        }).profile

        # === 核心超参数 ===
        self.declare_parameter('lookahead_dist', 2.0)   # 往前看多远 (米)
        self.declare_parameter('search_radius', 0.5)    # 路径点周围的搜索半径 (米)
        # h_max <= 0 derives the categorical entropy bound from num_classes.
        self.declare_parameter('num_classes', self.profile.K)
        self.declare_parameter('h_max', 0.0)
        self.declare_parameter('min_speed_ratio', 0.3)  # 最低降速到 30%
        self.declare_parameter('unknown_speed_ratio', 0.65)
        self.declare_parameter('linear_deadband', 0.03)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('uncertainty_ema_gain', 0.2)
        self.declare_parameter('uncertainty_percentile', 90.0)
        self.declare_parameter('uncertainty_percentile_weight', 0.35)
        self.declare_parameter('max_scale_rate_per_sec', 0.0)
        # Deprecated compatibility inputs. They are converted to a per-second
        # rate, so output no longer depends on callback frequency.
        self.declare_parameter('max_scale_step', 0.08)
        self.declare_parameter('legacy_cmd_rate_hz', 10.0)
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
        # Safety freshness limits. Positive defaults fail conservatively when
        # any required stream stops or its source timestamps stop advancing.
        self.declare_parameter('plan_max_age_sec', 2.0)
        self.declare_parameter('entropy_max_age_sec', 2.5)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('imu_max_age_sec', 0.5)
        self.declare_parameter('require_fresh_imu', True)
        self.declare_parameter('freshness_future_tolerance_sec', 0.1)
        # Twist has no Header. Use receipt time on a steady clock, keep only
        # the newest DDS sample, and stop independently after this timeout.
        self.declare_parameter('cmd_vel_timeout_sec', 0.25)
        self.declare_parameter('watchdog_period_sec', 0.05)

        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.search_radius = self.get_parameter('search_radius').value
        self.num_classes = int(self.get_parameter('num_classes').value)
        load_semantic_contract({
            'ontology_profile': self.profile.id,
            'num_classes': self.num_classes,
        })
        configured_h_max = float(self.get_parameter('h_max').value)
        self.h_max = resolve_h_max(configured_h_max, self.num_classes)
        self.min_speed_ratio = self.get_parameter('min_speed_ratio').value
        self.unknown_speed_ratio = float(
            self.get_parameter('unknown_speed_ratio').value)
        self.linear_deadband = float(
            self.get_parameter('linear_deadband').value)
        self.angular_deadband = float(
            self.get_parameter('angular_deadband').value)
        self.uncertainty_ema_gain = float(
            self.get_parameter('uncertainty_ema_gain').value)
        self.uncertainty_percentile = float(
            self.get_parameter('uncertainty_percentile').value)
        self.uncertainty_percentile_weight = float(
            self.get_parameter('uncertainty_percentile_weight').value)
        configured_scale_rate = float(
            self.get_parameter('max_scale_rate_per_sec').value)
        self.max_scale_step = float(self.get_parameter('max_scale_step').value)
        self.legacy_cmd_rate_hz = float(
            self.get_parameter('legacy_cmd_rate_hz').value)
        if configured_scale_rate > 0.0:
            self.max_scale_rate_per_sec = configured_scale_rate
        else:
            self.max_scale_rate_per_sec = max(
                0.0, self.max_scale_step * self.legacy_cmd_rate_hz)
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
        self.path_entropy_topic = self.get_parameter(
            'path_entropy_topic').value
        self.perception_mode_topic = self.get_parameter(
            'perception_mode_topic').value
        self.speed_scale_topic = self.get_parameter('speed_scale_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.plan_max_age_sec = float(
            self.get_parameter('plan_max_age_sec').value)
        self.entropy_max_age_sec = float(
            self.get_parameter('entropy_max_age_sec').value)
        self.imu_topic = self.get_parameter('imu_topic').value
        self.imu_max_age_sec = float(
            self.get_parameter('imu_max_age_sec').value)
        self.require_fresh_imu = bool(
            self.get_parameter('require_fresh_imu').value)
        self.freshness_future_tolerance_sec = float(
            self.get_parameter('freshness_future_tolerance_sec').value)
        self.cmd_vel_timeout_sec = float(
            self.get_parameter('cmd_vel_timeout_sec').value)
        configured_watchdog_period_sec = float(
            self.get_parameter('watchdog_period_sec').value)
        if not is_valid_frame_id(self.odom_frame):
            raise ValueError(f'Invalid odom_frame: {self.odom_frame!r}')
        if not is_valid_frame_id(self.base_frame):
            raise ValueError(f'Invalid base_frame: {self.base_frame!r}')
        positive_limits = {
            'plan_max_age_sec': self.plan_max_age_sec,
            'entropy_max_age_sec': self.entropy_max_age_sec,
            'imu_max_age_sec': self.imu_max_age_sec,
            'cmd_vel_timeout_sec': self.cmd_vel_timeout_sec,
            'watchdog_period_sec': configured_watchdog_period_sec,
        }
        for name, value in positive_limits.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if (not math.isfinite(self.freshness_future_tolerance_sec)
                or self.freshness_future_tolerance_sec < 0.0):
            raise ValueError(
                'freshness_future_tolerance_sec must be finite and '
                'non-negative')
        if self.require_fresh_imu and not self.imu_topic:
            raise ValueError(
                'imu_topic cannot be empty when require_fresh_imu=true')
        self.watchdog_period_sec = min(
            configured_watchdog_period_sec,
            self.cmd_vel_timeout_sec * 0.5,
        )

        # Heavy cloud construction, control, IMU, and the watchdog must not
        # share a mutually-exclusive callback group. A multi-threaded executor
        # below lets the steady-clock watchdog stop the robot even while the
        # Python cloud callback is busy.
        self.path_callback_group = MutuallyExclusiveCallbackGroup()
        self.cloud_callback_group = MutuallyExclusiveCallbackGroup()
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.watchdog_callback_group = MutuallyExclusiveCallbackGroup()
        self.state_lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        # === TF 监听器 ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === 状态缓存 ===
        self.path_points = None
        self.cloud_tree = None
        self.cloud_uncertainties = None
        self.filtered_uncertainty = None
        self.current_speed_scale = None
        self.last_scale_update_sec = None
        self.last_warning_times = {}
        self.path_stamp_sec = None
        self.cloud_stamp_sec = None
        self.imu_stamp_sec = None
        self.last_cmd_receipt_steady_sec = None
        self.command_sequence = 0
        self.watchdog_stop_published = False

        # === 订阅器 ===
        latest_qos = QoSProfile(depth=1)
        self.plan_sub = self.create_subscription(
            Path,
            self.plan_topic,
            self.plan_cb,
            latest_qos,
            callback_group=self.path_callback_group,
        )
        # intensity carries the map's entropy-equivalent/fused uncertainty.
        self.entropy_sub = self.create_subscription(
            PointCloud2,
            self.entropy_topic,
            self.entropy_cb,
            latest_qos,
            callback_group=self.cloud_callback_group,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_cb,
            make_imu_sensor_qos(),
            callback_group=self.imu_callback_group,
        )
        # 拦截 Nav2 控制器输出 (启动 Nav2 时把 controller_server 的 cmd_vel remap 到这里)
        self.cmd_sub = self.create_subscription(
            Twist,
            self.nav_cmd_vel_topic,
            self.cmd_vel_cb,
            latest_qos,
            callback_group=self.control_callback_group,
        )

        # === 发布器 ===
        # Final velocity sent to the simulation or the real robot bridge.
        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, 10)
        # Legacy topic name retained; this publishes fused path uncertainty.
        self.entropy_pub = self.create_publisher(
            Float32, self.path_entropy_topic, 10)
        self.mode_pub = self.create_publisher(
            String, self.perception_mode_topic, 10)
        self.speed_scale_pub = self.create_publisher(
            Float32, self.speed_scale_topic, 10)
        self.watchdog_timer = self.create_timer(
            self.watchdog_period_sec,
            self.command_watchdog_cb,
            callback_group=self.watchdog_callback_group,
            clock=self.steady_clock,
        )

        self.get_logger().info(
            "主动感知降速节点已启动: "
            f"{self.nav_cmd_vel_topic} -> {self.cmd_vel_topic}, "
            f"frames={self.odom_frame}->{self.base_frame}, "
            f"h_max={self.h_max:.5f}, max_scale_rate="
            f"{self.max_scale_rate_per_sec:.3f}/s, "
            f"freshness(path={self.plan_max_age_sec:.2f}s, "
            f"cloud={self.entropy_max_age_sec:.2f}s, "
            f"imu={self.imu_max_age_sec:.2f}s required="
            f"{self.require_fresh_imu}), cmd_watchdog="
            f"{self.cmd_vel_timeout_sec:.3f}s")

    def _warn_throttled(self, key, message, period_sec=5.0):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        last_sec = self.last_warning_times.get(key)
        if (last_sec is None or now_sec < last_sec
                or now_sec - last_sec >= period_sec):
            self.get_logger().warn(message)
            self.last_warning_times[key] = now_sec

    def _ros_now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _steady_now_sec(self):
        return self.steady_clock.now().nanoseconds * 1e-9

    def _clear_path(self):
        with self.state_lock:
            self.path_points = None
            self.path_stamp_sec = None

    def _clear_cloud(self):
        with self.state_lock:
            self.cloud_tree = None
            self.cloud_uncertainties = None
            self.cloud_stamp_sec = None

    @staticmethod
    def _lookup_time(stamp):
        if stamp.sec == 0 and stamp.nanosec == 0:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(stamp)

    def _transform_points(self, points, source_frame, stamp):
        if source_frame == self.odom_frame:
            return np.asarray(points, dtype=np.float64)
        transform = self.tf_buffer.lookup_transform(
            self.odom_frame,
            source_frame,
            self._lookup_time(stamp),
        ).transform
        return transform_xyz_points(
            points,
            [transform.translation.x, transform.translation.y,
             transform.translation.z],
            [transform.rotation.x, transform.rotation.y,
             transform.rotation.z, transform.rotation.w],
        )

    def plan_cb(self, msg):
        """Cache a Nav2 path only after expressing it in odom_frame."""
        source_frame = msg.header.frame_id
        if not is_valid_frame_id(source_frame):
            self._clear_path()
            self._warn_throttled(
                'path_frame',
                'Path has a missing or invalid frame_id; ignoring it')
            return
        stamp_sec = stamp_to_seconds(msg.header.stamp)
        if stamp_sec is None:
            self._clear_path()
            self._warn_throttled(
                'path_stamp',
                'Path has a missing or invalid timestamp; ignoring it')
            return

        points = []
        for pose in msg.poses:
            pose_frame = pose.header.frame_id
            if pose_frame and pose_frame != source_frame:
                self._clear_path()
                self._warn_throttled(
                    'path_mixed_frames',
                    'Path poses use a frame different from its header; '
                    'ignoring it')
                return
            points.append([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ])
        if not points:
            self._clear_path()
            return
        try:
            transformed = self._transform_points(
                points, source_frame, msg.header.stamp)
            if not np.all(np.isfinite(transformed)):
                raise ValueError(
                    'transformed path contains non-finite coordinates')
            with self.state_lock:
                self.path_points = transformed[:, :2]
                self.path_stamp_sec = stamp_sec
        except Exception as exc:
            # Clear the old path: stale coordinates are unsafe after a frame
            # change or a missing transform.
            self._clear_path()
            self._warn_throttled(
                'path_transform',
                f'Cannot transform Path {source_frame!r} to '
                f'{self.odom_frame!r}; ignoring it: {exc}')

    def entropy_cb(self, msg):
        """Cache fused uncertainty points after transforming them to odom."""
        source_frame = msg.header.frame_id
        if not is_valid_frame_id(source_frame):
            self._clear_cloud()
            self._warn_throttled(
                'cloud_frame',
                'Uncertainty cloud has a missing or invalid frame_id; '
                'ignoring it')
            return
        stamp_sec = stamp_to_seconds(msg.header.stamp)
        if stamp_sec is None:
            self._clear_cloud()
            self._warn_throttled(
                'cloud_stamp',
                'Uncertainty cloud has a missing or invalid timestamp; '
                'ignoring it')
            return

        points = []
        uncertainties = []
        try:
            for point in pc2.read_points(
                    msg,
                    field_names=("x", "y", "z", "intensity"),
                    skip_nans=True):
                uncertainty = float(point[3])
                if not math.isfinite(uncertainty):
                    continue
                points.append([point[0], point[1], point[2]])
                uncertainties.append(uncertainty)
            if not points:
                self._clear_cloud()
                return
            transformed = self._transform_points(
                points, source_frame, msg.header.stamp)
            if not np.all(np.isfinite(transformed)):
                raise ValueError(
                    'transformed cloud contains non-finite coordinates')
            cloud_tree = cKDTree(transformed[:, :2])
            cloud_uncertainties = np.asarray(
                uncertainties, dtype=np.float64)
            with self.state_lock:
                self.cloud_tree = cloud_tree
                self.cloud_uncertainties = cloud_uncertainties
                self.cloud_stamp_sec = stamp_sec
        except Exception as exc:
            self._clear_cloud()
            self._warn_throttled(
                'cloud_transform',
                f'Cannot consume uncertainty cloud {source_frame!r} in '
                f'{self.odom_frame!r}; ignoring it: {exc}')

    def imu_cb(self, msg):
        """Record finite, source-stamped IMU samples for the health gate."""
        stamp_sec = stamp_to_seconds(msg.header.stamp)
        motion_values = (
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        )
        if stamp_sec is None or not all(
                math.isfinite(float(value)) for value in motion_values):
            with self.state_lock:
                self.imu_stamp_sec = None
            self._warn_throttled(
                'imu_invalid',
                'IMU has an invalid timestamp or non-finite motion sample; '
                'the velocity gate will remain conservative')
            return
        with self.state_lock:
            self.imu_stamp_sec = stamp_sec

    def cmd_vel_cb(self, msg):
        """Intercept the newest velocity and modulate it using fresh inputs."""
        receipt_steady_sec = self._steady_now_sec()
        with self.command_lock:
            self.command_sequence += 1
            command_sequence = self.command_sequence
            self.last_cmd_receipt_steady_sec = receipt_steady_sec
            self.watchdog_stop_published = False

        command_values = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(float(value)) for value in command_values):
            with self.command_lock:
                if command_sequence == self.command_sequence:
                    self.cmd_pub.publish(Twist())
                    self.watchdog_stop_published = True
            self._warn_throttled(
                'invalid_cmd_vel',
                'Velocity command contains non-finite values; publishing zero')
            self.mode_pub.publish(String(data='STOPPED (invalid cmd_vel)'))
            self.speed_scale_pub.publish(Float32(data=0.0))
            return

        now_sec = self._ros_now_sec()
        with self.state_lock:
            path_points = self.path_points
            cloud_tree = self.cloud_tree
            cloud_uncertainties = self.cloud_uncertainties
            path_stamp_sec = self.path_stamp_sec
            cloud_stamp_sec = self.cloud_stamp_sec
            imu_stamp_sec = self.imu_stamp_sec

        stale_inputs = stale_required_inputs(
            now_sec,
            path_stamp_sec,
            cloud_stamp_sec,
            imu_stamp_sec,
            self.plan_max_age_sec,
            self.entropy_max_age_sec,
            self.imu_max_age_sec,
            self.freshness_future_tolerance_sec,
            self.require_fresh_imu,
        )
        # No mapped support is treated as fully unknown.
        path_uncertainty = self.h_max
        has_uncertainty_support = False

        if (not stale_inputs and path_points is not None
                and cloud_tree is not None):
            try:
                # 1. 找狗的位置 (与 SLAM/ga_bsvm_node 一致用 odom 系)
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                )
                rx, ry = t.transform.translation.x, t.transform.translation.y

                # 2. 截取沿路径累计长度不超过 lookahead_dist 的路径片段
                dists = np.linalg.norm(
                    path_points - np.array([rx, ry]), axis=1)
                closest_idx = np.argmin(dists)

                lookahead_pts = [path_points[closest_idx]]
                path_length = 0.0
                for i in range(closest_idx, len(path_points)):
                    if i > closest_idx:
                        path_length += float(np.linalg.norm(
                            path_points[i] - path_points[i - 1]))
                    if path_length > self.lookahead_dist:
                        break
                    if i > closest_idx:
                        lookahead_pts.append(path_points[i])

                # 3. KD树圈地：查路径周围 0.5 米内的点
                if len(lookahead_pts) > 0:
                    idxs = cloud_tree.query_ball_point(
                        lookahead_pts, r=self.search_radius)

                    # 展平结果并去重
                    valid_idxs = list({
                        index for nearby in idxs for index in nearby
                    })

                    if len(valid_idxs) > 0:
                        path_uncertainty = aggregate_uncertainty(
                            cloud_uncertainties[valid_idxs],
                            percentile=self.uncertainty_percentile,
                            percentile_weight=(
                                self.uncertainty_percentile_weight),
                        )
                        has_uncertainty_support = True
                    else:
                        # No mapped support along the path: full uncertainty.
                        path_uncertainty = self.h_max
            except Exception as e:
                # TF 还没建立或地图还没数据时降级到 debug, 不刷屏但能查
                self._warn_throttled(
                    'uncertainty_query',
                    f'TF/不确定度查询失败 (启动期常见): {e}')

        gain = float(np.clip(self.uncertainty_ema_gain, 0.0, 1.0))
        with self.state_lock:
            if stale_inputs:
                # Stale safety evidence must take effect immediately. Do not
                # let EMA or the upward/downward rate limiter retain optimism.
                self.filtered_uncertainty = float(self.h_max)
                self.current_speed_scale = float(self.min_speed_ratio)
                self.last_scale_update_sec = now_sec
            else:
                if self.filtered_uncertainty is None:
                    self.filtered_uncertainty = float(path_uncertainty)
                else:
                    self.filtered_uncertainty = (
                        (1.0 - gain) * self.filtered_uncertainty
                        + gain * float(path_uncertainty))

                uncertainty_ratio = np.clip(
                    self.filtered_uncertainty / max(self.h_max, 1e-6),
                    0.0,
                    1.0,
                )
                if has_uncertainty_support:
                    target_scale = self.min_speed_ratio + (
                        1.0 - self.min_speed_ratio) * (
                            1.0 - uncertainty_ratio)
                else:
                    target_scale = float(np.clip(
                        self.unknown_speed_ratio,
                        self.min_speed_ratio,
                        1.0,
                    ))
                (
                    self.current_speed_scale,
                    self.last_scale_update_sec,
                ) = rate_limit_scale(
                    self.current_speed_scale,
                    target_scale,
                    now_sec,
                    self.last_scale_update_sec,
                    self.max_scale_rate_per_sec,
                )
            filtered_uncertainty = float(self.filtered_uncertainty)
            alpha = float(np.clip(
                self.current_speed_scale, self.min_speed_ratio, 1.0))

        if stale_inputs:
            ages = {
                'path': sample_age_seconds(now_sec, path_stamp_sec),
                'entropy_cloud': sample_age_seconds(
                    now_sec, cloud_stamp_sec),
                'imu': sample_age_seconds(now_sec, imu_stamp_sec),
            }
            age_text = ', '.join(
                f'{name}={ages[name]:.2f}s' for name in stale_inputs)
            self._warn_throttled(
                'stale_control_inputs',
                'Active-perception input freshness gate is conservative: '
                f'{age_text}')

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

        # Serialize the freshness check with watchdog publication. A command
        # that spent longer than the timeout in this callback can never be
        # published after the watchdog's zero.
        with self.command_lock:
            command_still_current = command_sequence == self.command_sequence
            command_expired = command_watchdog_expired(
                self._steady_now_sec(),
                receipt_steady_sec,
                self.cmd_vel_timeout_sec,
            )
            if not command_still_current or command_expired:
                return
            self.cmd_pub.publish(mod_cmd)

        # Keep the legacy topic name for compatibility; the value represents
        # entropy-equivalent/fused semantic uncertainty.
        self.entropy_pub.publish(
            Float32(data=filtered_uncertainty))
        self.speed_scale_pub.publish(Float32(data=alpha))

        # 监控发布 (供 UI 展示和录制视频用)
        if stale_inputs:
            mode_str = f"STALE[{','.join(stale_inputs)}]"
        elif not has_uncertainty_support:
            mode_str = "EXPLORING"
        else:
            mode_str = "CONFIDENT" if alpha > 0.8 else "CAUTIOUS"
        self.mode_pub.publish(String(
            data=(f"{mode_str} (U={filtered_uncertainty:.2f}, "
                  f"scale={alpha:.2f})")))

    def command_watchdog_cb(self):
        """Publish one zero whenever the input command stream expires."""
        published_stop = False
        now_steady_sec = self._steady_now_sec()
        with self.command_lock:
            if (command_watchdog_expired(
                    now_steady_sec,
                    self.last_cmd_receipt_steady_sec,
                    self.cmd_vel_timeout_sec)
                    and not self.watchdog_stop_published):
                self.cmd_pub.publish(Twist())
                self.watchdog_stop_published = True
                published_stop = True

        if not published_stop:
            return

        now_sec = self._ros_now_sec()
        with self.state_lock:
            self.filtered_uncertainty = float(self.h_max)
            self.current_speed_scale = float(self.min_speed_ratio)
            self.last_scale_update_sec = now_sec
        self.entropy_pub.publish(Float32(data=float(self.h_max)))
        self.speed_scale_pub.publish(Float32(data=0.0))
        self.mode_pub.publish(String(data='STOPPED (cmd_vel timeout)'))
        self._warn_throttled(
            'cmd_vel_timeout',
            'No fresh velocity command; watchdog published zero',
        )


def main():
    rclpy.init()
    node = ActivePerceptionNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
