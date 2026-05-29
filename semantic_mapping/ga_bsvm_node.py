#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, Header
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import numpy as np
import message_filters
import struct
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
        
        # === 1. 外参矩阵 T_lidar2cam ===
        trans = [0.2, 0.0, -0.03]          
        quat = [-0.5, 0.5, -0.5, 0.5]      
        
        rot_matrix = R.from_quat(quat).as_matrix()
        self.T_lidar2cam = np.eye(4, dtype=np.float64)
        self.T_lidar2cam[:3, :3] = rot_matrix
        self.T_lidar2cam[:3, 3] = trans
        
        # 相机内参 K
        self.K = np.array([
            [554.25, 0.0, 320.5],
            [0.0, 554.25, 240.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # === 2. 传感器订阅与同步 ===
        self.imu_buffer = []
        # [仿真/Livox专用] 
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 200)
        #self.imu_sub = self.create_subscription(Imu, '/handsfree/imu', self.imu_callback, 200)  # [M2DGR真实数据集] 底盘 IMU

        # 🗡️ 刺客1修复：把接收话题改回和 clip_node 一致的默认话题名称
        self.clip_logits_sub = self.create_subscription(Float32MultiArray, '/clip_logits', self.logits_callback, 10)
        self.clip_features_sub = self.create_subscription(Float32MultiArray, '/clip_features', self.features_callback, 10)

        self.logits_grid = None
        self.feats_grid = None
        
        # ⚠️ 如果你 clip_node 里改了，这里的行数、列数、分类数量必须完全匹配！
        self.grid_rows = 3
        self.grid_cols = 4
        self.num_classes = 6
        self.feat_dim = 512

        # [仿真/Livox专用] 
        self.pc_sub = message_filters.Subscriber(self, CustomMsg, '/livox/lidar', qos_profile=qos_profile_sensor_data)
        
        #self.pc_sub = message_filters.Subscriber(self, PointCloud2, '/velodyne_points', qos_profile=qos_profile_sensor_data)  # [M2DGR真实数据集] Velodyne点云

        # [仿真/Livox专用] 
        self.img_sub = message_filters.Subscriber(self, Image, '/d435i/image_raw', qos_profile=qos_profile_sensor_data)
        
        #self.img_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed', qos_profile=qos_profile_sensor_data)  # [M2DGR真实数据集] 压缩图像
        
        # 🗡️ 刺客3修复：slop从0.05放大到0.2，容忍真实世界传感器的时间差！
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pc_sub, self.img_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.sync_callback)
        
        self.voxel_map = VoxelMap(voxel_size=0.1, K=6)
        self.get_logger().info('🌍 贝叶斯 VoxelMap 已创建，网格精度: 0.1m')
        
        self.pub_semantic = self.create_publisher(PointCloud2, '/semantic_cloud', 1)
        self.pub_uncertainty = self.create_publisher(PointCloud2, '/uncertainty_cloud', 1)
        # 高精度熵数据话题: x/y/z + intensity(float32 香农熵)，供 active_perception_node 直读
        self.pub_entropy_data = self.create_publisher(PointCloud2, '/voxel_entropy_data', 10)
        self.query_sub = self.create_subscription(Float32MultiArray, '/query_feature', self.query_feature_cb, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # ====== 🌟 核心修复：强制使用 Transient Local QoS 发布地图 ======
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.frame_count = 0

        self.current_goal_key = None # 用于记住目标的网格位置
        self.get_logger().info('✅ GA-BSVM 中枢已启动！(支持仿真与真实数据集)')

    def imu_callback(self, msg):
        w_norm = np.linalg.norm([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_buffer.append(w_norm)
        if len(self.imu_buffer) > 400: 
            self.imu_buffer.pop(0)

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

    def compute_S_time(self):
        if len(self.imu_buffer) < 100:
            return 1.0, 0.0, 0.0 
        data = np.array(self.imu_buffer)
        freqs = np.fft.rfftfreq(len(data), d=1/200.0) 
        fft_mag = np.abs(np.fft.rfft(data - np.mean(data)))
        valid_idx = (freqs > 0.5) & (freqs < 4.0)
        if not np.any(valid_idx):
            f_gait = 1.5
        else:
            f_gait = freqs[valid_idx][np.argmax(fft_mag[valid_idx])]
        t = self.get_clock().now().nanoseconds * 1e-9
        phi = 2 * np.pi * f_gait * t
        S_phi = (1.0 + np.cos(phi)) / 2.0
        w_norm = data[-1]
        tau = 1.5 
        S_imu = 1.0 / (1.0 + w_norm / tau)
        return S_phi * S_imu, f_gait, w_norm

    # [仿真/Livox专用] 解析 Livox CustomMsg 自定义点云
    def parse_livox_msg(self, msg):
        points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        valid = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points, axis=1) > 0.1)
        return points[valid]

    # [M2DGR真实数据集] 解析标准 PointCloud2 (Velodyne) 点云 —— 当前注释停用
    # def parse_velodyne_msg(self, msg):
    #     points = []
    #     for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
    #         points.append([p[0], p[1], p[2]])
    #     points = np.array(points, dtype=np.float32)
    #     if len(points) == 0:
    #         return np.zeros((0, 3), dtype=np.float32)
    #     valid = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points, axis=1) > 0.1)
    #     return points[valid]

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

    def query_feature_cb(self, msg):
        query_feat = np.array(msg.data, dtype=np.float32)
        best_sim = -1.0
        best_pos = None
        best_key = None

        for key, voxel in self.voxel_map.voxels.items():
            if voxel['feature_512'] is None or voxel['weight_sum'] < 2.0:
                continue
            sim = np.dot(voxel['feature_512'], query_feat)
            if sim > best_sim:
                best_sim = sim
                best_pos = voxel['pos']
                best_key = key

        if best_pos is not None:
            self.current_goal_key = best_key # 记住目标所在的网格，方便等会挖空！
            self.get_logger().info(f'🎯 [导航触发] 目标坐标: x={best_pos[0]:.2f}, y={best_pos[1]:.2f}')
            
            goal = PoseStamped()
            goal.header.frame_id = 'odom'  # 与 SLAM 输出坐标系一致, 不再伪造 map
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(best_pos[0])
            goal.pose.position.y = float(best_pos[1])
            goal.pose.position.z = float(best_pos[2])
            goal.pose.orientation.w = 1.0
            self.goal_pub.publish(goal)
        else:
            self.get_logger().warn('🔍 未找到合适目标。')

    def sync_callback(self, pc_msg, img_msg):
        S_time, f_gait, w_norm = self.compute_S_time()

        if self.logits_grid is None:
            return

        try:
            # [仿真/Livox专用] 普通 Image 直接转 cv2
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'rgb8')
            # [M2DGR真实数据集] JPEG 解压 —— 当前注释停用
            # cv_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'rgb8')
            H, W = cv_img.shape[:2]

            # [仿真/Livox专用] Livox CustomMsg 解析
            points_3d = self.parse_livox_msg(pc_msg)
            # [M2DGR真实数据集] Velodyne PointCloud2 解析 —— 当前注释停用
            # points_3d = self.parse_velodyne_msg(pc_msg)
            
            valid_mask, u, v = self.project_points(points_3d, H, W)
            valid_points = points_3d[valid_mask]
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]

            if len(valid_points) == 0:
                return

            try:
                t = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
                t_vec = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                rot_mat = R.from_quat(q).as_matrix()
                valid_points = (rot_mat @ valid_points.T).T + t_vec
            except Exception as e:
                # 只有这里去掉了报错提示，避免真实数据集缺 TF 时疯狂刷屏
                return

            patch_h = H // self.grid_rows
            patch_w = W // self.grid_cols
            grid_r = np.clip(valid_v // patch_h, 0, self.grid_rows - 1)
            grid_c = np.clip(valid_u // patch_w, 0, self.grid_cols - 1)

            per_point_logits = self.logits_grid[grid_r, grid_c]

            per_point_feats = None
            if self.feats_grid is not None:
                per_point_feats = self.feats_grid[grid_r, grid_c]

            tree = cKDTree(valid_points)
            neighbors = tree.query_ball_point(valid_points, r=0.3)
            rho = np.array([len(n) for n in neighbors])
            S_rho = np.clip(rho / 10.0, 0.0, 1.0)

            S_combined = S_time * S_rho

            new_vox, up_vox = self.voxel_map.update(
                points=valid_points,
                S_combined=S_combined,
                logits=per_point_logits,
                features=per_point_feats
            )

            self.frame_count += 1

            # 高频熵话题: 每 2 帧 (~5Hz) 发一次, 给主动降速节点用
            if self.frame_count % 2 == 0:
                self.publish_entropy_data()

            if self.frame_count % 10 == 0:
                sem_pts, unc_pts = self.voxel_map.get_visualization_clouds()
                if len(sem_pts) > 0:
                    header = pc_msg.header
                    header.frame_id = 'odom'

                    sem_msg = self.create_cloud_msg(header, sem_pts)
                    self.pub_semantic.publish(sem_msg)

                    unc_msg = self.create_cloud_msg(header, unc_pts)
                    self.pub_uncertainty.publish(unc_msg)

                    self.publish_2d_nav_map()
                    self.get_logger().info(f'📸 成功投影并更新 {len(valid_points)} 个 3D 点到全局坐标系！')

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
        发布纯数据熵话题 /voxel_entropy_data (xyz + intensity=float32 香农熵)
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
            confidence = self.voxel_map.get_confidence(key)
            entropy = (1.0 - confidence) * H_max  # float32 高精度熵
            pts.append([float(pos[0]), float(pos[1]), float(pos[2]), float(entropy)])

        if len(pts) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom'  # 与 SLAM 输出坐标系一致

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg = pc2.create_cloud(header, fields, pts)
        self.pub_entropy_data.publish(msg)

    def publish_2d_nav_map(self):
        if not self.voxel_map.voxels:
            return

        keys = list(self.voxel_map.voxels.keys())
        min_x = min(k[0] for k in keys)
        max_x = max(k[0] for k in keys)
        min_y = min(k[1] for k in keys)
        max_y = max(k[1] for k in keys)

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = np.full((height, width), -1, dtype=np.int8)

        for key, voxel in self.voxel_map.voxels.items():
            ix, iy, iz = key
            grid_x = ix - min_x
            grid_y = iy - min_y

            if voxel['weight_sum'] < 1.0: 
                continue

            best_idx = np.argmax(voxel['log_odds'])
            if best_idx in [0, 1]:      # 草地/水泥地
                if grid[grid_y, grid_x] != 100:
                    grid[grid_y, grid_x] = 0
            elif best_idx in [2, 3, 4]: # 墙/人/石头
                grid[grid_y, grid_x] = 100

        # ====== 🌟 破局大招：在目标周围“挖洞” ======
        if self.current_goal_key is not None:
            gx = self.current_goal_key[0] - min_x
            gy = self.current_goal_key[1] - min_y
            # 挖一个 9x9 的超级大坑 (半径 0.45 米)
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if 0 <= gy+dy < height and 0 <= gx+dx < width:
                        grid[gy+dy, gx+dx] = 0 # 强行设为 0 (可通行平地)
                        
        # 2. 起点周围也挖大坑，防止机器狗觉得自己的出生点卡在未知空间里！
        rx, ry = int(0 - min_x), int(0 - min_y) # 假设机器狗目前在原点附近
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if 0 <= ry+dy < height and 0 <= rx+dx < width:
                    grid[ry+dy, rx+dx] = 0
        # ============================================

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'  # 与 SLAM 输出坐标系一致 (Nav2 启动时 global_frame 也设为 odom)
        msg.info.resolution = self.voxel_map.voxel_size
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = float(min_x * self.voxel_map.voxel_size)
        msg.info.origin.position.y = float(min_y * self.voxel_map.voxel_size)
        msg.data = grid.flatten().tolist()
        self.map_pub.publish(msg)

def main():
    rclpy.init()
    node = GABsvmNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()