"""ROS adapter that records benchmark evidence, never supplies prediction inputs."""

from collections import deque
from functools import partial
import json
import math
import time
import xml.etree.ElementTree as ET

from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped, Twist
from lifecycle_msgs.srv import GetState
from livox_ros_driver2.msg import CustomMsg
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rcl_interfaces.msg import Log
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock as ClockMessage
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage

from semantic_mapping.gazebo.indoor_benchmark import (
    MAX_POSE_PAIR_SKEW_SEC,
    _pose_matrix,
)
from semantic_mapping.runtime.semantic_posterior import posterior_image_to_array
from semantic_mapping.runtime.semantic_profile import open_profile
from semantic_mapping.runtime.semantic_profile_ros import (
    read_semantic_capability, semantic_capability_qos,
)


def _stamp(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def _pose(message):
    value = message.pose.pose if isinstance(message, Odometry) else message.pose
    return {'stamp': _stamp(message), 'frame': message.header.frame_id.lstrip('/'),
            'position': [value.position.x, value.position.y, value.position.z],
            'orientation': [value.orientation.x, value.orientation.y,
                            value.orientation.z, value.orientation.w]}


class BenchmarkObserver(Node):
    """Own the complete readiness/query observation sequence for one fresh run."""

    def __init__(self, case):
        self.ros_context = Context()
        rclpy.init(args=[], context=self.ros_context)
        super().__init__(
            'indoor_benchmark_observer', context=self.ros_context,
            parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.ros_executor = SingleThreadedExecutor(context=self.ros_context)
        self.ros_executor.add_node(self)
        self.case = case
        self.profile = open_profile(case['ontology_profile'])
        self.recording = {'termination': 'startup_timeout', 'readiness': {'passed': False},
                          'query': {'sent': False, 'ack': None}, 'target_poses': [],
                          'goal_poses': [], 'bridge_events': [], 'counts': {},
                          'control': {}}
        self.last = {}
        self.poses = {'truth': deque(maxlen=3000), 'estimate': deque(maxlen=3000)}
        self.odom_tf_seen = False
        self.urdf_base_is_root = False
        self.capability = None
        self.clock_wall = None
        self.first_estimate_stamp = None
        self.service_status = {}
        self.pending = {}
        streams = {
            'rgb': (Image, '/d435i/image_raw'), 'imu': (Imu, '/imu/data'),
            'lidar': (CustomMsg, '/livox/lidar'), 'registered': (PointCloud2, '/cloud_registered'),
            'posterior': (Image, '/segformer/project_posterior'),
            'semantic': (PointCloud2, '/semantic_cloud'),
            'costmap': (OccupancyGrid, '/semantic_cost_map'),
            'truth': (Odometry, '/odom/ground_truth'), 'estimate': (Odometry, '/Odometry'),
        }
        for name, (kind, topic) in streams.items():
            self.create_subscription(
                kind, topic, partial(self._sensor, name), qos_profile_sensor_data)
        self.create_subscription(ClockMessage, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(String, '/segformer/semantic_capability', self._capability,
                                 semantic_capability_qos())
        self.create_subscription(TFMessage, '/tf', self._tf, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/tf_static', self._tf, semantic_capability_qos())
        self.create_subscription(String, '/robot_description', self._description,
                                 semantic_capability_qos())
        for key, topic in (('target_poses', '/query_target_pose'), ('goal_poses', '/goal_pose')):
            self.create_subscription(PoseStamped, topic, partial(self._target, key), 10)
        self.create_subscription(String, '/nav_goal_bridge/status', self._bridge, 10)
        self.create_subscription(
            Twist, '/cmd_vel', partial(self._command, 'nav_cmd'), 10)
        self.create_subscription(
            Twist, '/cmd_vel_champ', partial(self._command, 'filtered_cmd'), 10)
        self.create_subscription(
            Float32, '/semantic_speed_scale', self._speed_scale, 10)
        self.create_subscription(
            String, '/perception_mode', self._perception_mode, 10)
        self.create_subscription(Log, '/rosout', self._log, 100)
        self.query_publisher = self.create_publisher(String, '/text_query', 10)
        self.navigation = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.service_clients = {
            name: self.create_client(GetState, f'/{name}/get_state') for name in (
                'controller_server', 'planner_server', 'bt_navigator',
                'behavior_server', 'velocity_smoother')}
        self.service_clients['controllers'] = self.create_client(
            ListControllers, '/controller_manager/list_controllers')
        self.create_timer(.2, self._poll_services, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def require_empty_domain(self):
        """Refuse to attach to an existing simulation or robot domain."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.ros_executor.spin_once(timeout_sec=.05)
        # ``ros2 run`` may start its own discovery daemon in this domain.  It
        # is a CLI helper, not a simulator or robot node, so it must not make
        # an otherwise isolated benchmark look occupied.
        others = [
            (name, namespace)
            for name, namespace in self.get_node_names_and_namespaces()
            if name != self.get_name() and not name.startswith('_ros2cli_daemon_')
        ]
        if others:
            raise ValueError(f'ROS domain is not empty: {others}')

    def _on_clock(self, message):
        stamp = message.clock.sec + message.clock.nanosec * 1e-9
        if stamp < self.recording.get('sim_time', 0):
            raise ValueError('Simulation clock moved backwards')
        self.recording['sim_time'] = stamp
        self.clock_wall = time.monotonic()

    def _sensor(self, name, message):
        if name in self.poses:
            value = _pose(message)
            _pose_matrix(value, 'world' if name == 'truth' else 'odom')
            if message.child_frame_id.lstrip('/') != 'base_link':
                raise ValueError('GT and FAST-LIO must both report base_link')
            self.poses[name].append(value)
            if name == 'estimate' and self.first_estimate_stamp is None:
                self.first_estimate_stamp = value['stamp']
        elif name == 'posterior':
            array = posterior_image_to_array(message, expected_classes=self.profile.K)
            if (not np.isfinite(array).all() or np.min(array) < 0
                    or not np.allclose(array.sum(axis=-1), 1, atol=1e-3)):
                raise ValueError('Invalid full semantic posterior')
            fractions = np.bincount(array.argmax(axis=-1).ravel(), minlength=self.profile.K)
            self.recording['posterior'] = {
                'encoding': message.encoding, 'K': self.profile.K,
                'argmax_pixel_fractions': dict(zip(
                    self.profile.classes, (fractions / fractions.sum()).tolist())),
            }
        elif name == 'lidar' and (message.point_num == 0 or not message.points):
            return
        elif name in ('rgb', 'registered', 'semantic', 'costmap') and not message.data:
            return
        self.last[name] = (_stamp(message), time.monotonic())
        counts = self.recording['counts']
        counts[name] = counts.get(name, 0) + 1

    def _capability(self, message):
        self.capability = read_semantic_capability(message, self.profile)
        if self.capability.checkpoint_id != self.case['checkpoint_path']:
            raise ValueError('Loaded checkpoint differs from the pinned case snapshot')
        self.recording['capability'] = json.loads(message.data)

    def _tf(self, message):
        for transform in message.transforms:
            if (transform.header.frame_id.lstrip('/') == 'odom'
                    and transform.child_frame_id.lstrip('/') == 'base_link'):
                self.odom_tf_seen = True

    def _description(self, message):
        root = ET.fromstring(message.data)
        children = {joint.find('child').get('link') for joint in root.findall('joint')}
        self.urdf_base_is_root = root.find('link[@name="base_link"]') is not None and (
            'base_link' not in children)

    def _tf_authority_problem(self):
        # Humble's Python callback discards publisher metadata. In this fixed
        # Go2 chain, exclude robot_state_publisher only when its actual URDF
        # proves base_link is the root (so it cannot publish a transform TO it).
        odom = self.get_publishers_info_by_topic('/Odometry')
        if len(odom) != 1 or not self.odom_tf_seen:
            return 'odom_tf_authority_not_unique'
        source = (odom[0].node_namespace, odom[0].node_name)
        dynamic = self.get_publishers_info_by_topic('/tf')
        static = self.get_publishers_info_by_topic('/tf_static')
        self.recording['tf_authority_graph'] = {
            'odometry_source': source,
            'dynamic_publishers': [
                (publisher.node_namespace, publisher.node_name)
                for publisher in dynamic
            ],
            'static_publishers': [
                (publisher.node_namespace, publisher.node_name)
                for publisher in static
            ],
            'base_link_is_urdf_root': self.urdf_base_is_root,
        }
        primary = 0
        for info in dynamic:
            identity = (info.node_namespace, info.node_name)
            if identity == source:
                primary += 1
            elif info.node_name != 'robot_state_publisher' or not self.urdf_base_is_root:
                return 'unverified_tf_publisher'
        if primary != 1:
            return 'odom_tf_authority_not_unique'
        if any(p.node_name != 'robot_state_publisher' or not self.urdf_base_is_root
               for p in static):
            return 'unverified_static_tf_publisher'
        self.recording['tf_authority'] = {
            'method': 'publisher_graph_and_urdf_root_exclusion', 'odometry_source': source,
            'dynamic_publishers': [(p.node_namespace, p.node_name) for p in dynamic],
            'static_publishers': [(p.node_namespace, p.node_name) for p in static],
            'base_link_is_urdf_root': self.urdf_base_is_root,
        }
        return None

    def _target(self, key, message):
        value = _pose(message)
        _pose_matrix(value, 'odom')
        self.recording[key].append(value)

    def _bridge(self, message):
        self.recording['bridge_events'].append(json.loads(message.data))

    def _command(self, name, message):
        values = (
            float(message.linear.x), float(message.linear.y),
            float(message.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Non-finite {name} sample')
        linear = math.hypot(values[0], values[1])
        angular = abs(values[2])
        stats = self.recording['control'].setdefault(name, {
            'count': 0, 'nonzero_count': 0,
            'max_linear_mps': 0.0, 'max_angular_rps': 0.0,
        })
        stats['count'] += 1
        stats['nonzero_count'] += int(linear > 1e-6 or angular > 1e-6)
        stats['max_linear_mps'] = max(stats['max_linear_mps'], linear)
        stats['max_angular_rps'] = max(stats['max_angular_rps'], angular)
        stats['last'] = {'linear_x': values[0], 'linear_y': values[1],
                         'angular_z': values[2]}

    def _speed_scale(self, message):
        value = float(message.data)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('Invalid semantic speed scale')
        stats = self.recording['control'].setdefault('speed_scale', {
            'count': 0, 'min': value, 'max': value, 'last': value,
        })
        stats['count'] += 1
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['last'] = value

    def _perception_mode(self, message):
        value = message.data.strip()
        if not value:
            raise ValueError('Empty perception mode')
        mode = value.split(' ', 1)[0]
        stats = self.recording['control'].setdefault('perception_mode', {
            'count': 0, 'last': mode, 'counts': {},
        })
        stats['count'] += 1
        stats['last'] = mode
        stats['counts'][mode] = stats['counts'].get(mode, 0) + 1

    def _log(self, message):
        if message.name != 'ga_bsvm_node' or not self.recording['query']['sent']:
            return
        if message.msg.startswith('SegFormer query accepted:'):
            self.recording['query']['ack'] = 'accepted'
        elif message.msg.startswith('SegFormer query rejected ('):
            self.recording['query']['ack'] = message.msg.split('(', 1)[1].split(')', 1)[0]

    def _poll_services(self):
        for name, client in self.service_clients.items():
            future = self.pending.get(name)
            if future is not None:
                if not future.done():
                    continue
                response = future.result()
                del self.pending[name]
                if name == 'controllers':
                    active = {v.name for v in response.controller if v.state == 'active'}
                    ready = {'joint_states_controller', 'joint_group_effort_controller'} <= active
                else:
                    ready = response.current_state.id == 3
                self.service_status[name] = (ready, time.monotonic())
            if client.service_is_ready():
                request = (ListControllers.Request() if name == 'controllers'
                           else GetState.Request())
                self.pending[name] = client.call_async(request)

    def _live_problem(self, require_fresh_service_status=True):
        now = time.monotonic()
        sim = self.recording.get('sim_time', 0)
        if self.clock_wall is None or now - self.clock_wall > 2:
            return 'clock_not_live'
        for name in ('rgb', 'imu', 'lidar', 'registered', 'posterior', 'semantic',
                     'costmap', 'truth', 'estimate'):
            stamp, wall = self.last.get(name, (0, 0))
            # /clock wall freshness proves that simulation is still advancing;
            # compare stamped streams in simulation time.  Their callback wall
            # intervals expand when Gazebo's real-time factor is low.
            max_age = 5.0 if name in ('posterior', 'semantic', 'costmap') else 2.0
            if stamp <= 0 or not -.100001 <= sim - stamp <= max_age:
                # Historical diagnostic only.  The return value represents the
                # current readiness state and may recover on a later poll.
                self.recording['last_freshness_failure'] = {
                    'stream': name, 'age_sim_sec': sim - stamp, 'age_wall_sec': now - wall}
                return f'{name}_not_fresh'
        if self.capability is None:
            return 'capability_not_received'
        for name in self.service_clients:
            ready, wall = self.service_status.get(name, (False, 0))
            if not ready or (require_fresh_service_status and now - wall > 2):
                return f'{name}_not_active'
        tf_problem = self._tf_authority_problem()
        if tf_problem:
            return tf_problem
        writers = self.get_publishers_info_by_topic('/cmd_vel_champ')
        if len(writers) != 1 or writers[0].node_name != 'active_perception_node':
            return 'final_command_writer_not_unique'
        if (not self.navigation.server_is_ready()
                or self.query_publisher.get_subscription_count() != 1):
            return 'query_or_nav2_not_ready'
        return None

    def _stationary_problem(self):
        now = self.recording.get('sim_time', 0)
        if (self.first_estimate_stamp is None
                or now - self.first_estimate_stamp < self.case['warmup_sim_sec']):
            return 'map_warmup'
        summary = {}
        for name, threshold in (('truth', .2), ('estimate', .5)):
            poses = [p for p in self.poses[name] if now - p['stamp'] <= 5]
            if len(poses) < 10 or poses[-1]['stamp'] - poses[0]['stamp'] < 4:
                return 'stationary_window_incomplete'
            displacement = max(math.dist(poses[0]['position'], p['position']) for p in poses)
            summary[name + '_max_displacement_m'] = displacement
            if displacement > threshold:
                return name + '_not_stationary'
        self.recording['stationary'] = summary
        return None

    def _paired_pose(self):
        """
        Return the newest synchronized pair available in the history.

        Gazebo and FAST-LIO publish on different callbacks.  At a single
        callback instant the newest estimate can be ahead of the newest truth
        message, even though an older pair in the bounded histories is valid.
        Do not turn that transient scheduling skew into a navigation failure.
        """
        estimates = list(self.poses['estimate'])
        truths = list(self.poses['truth'])
        for estimate in reversed(estimates):
            truth = min(truths, key=lambda p: abs(p['stamp'] - estimate['stamp']))
            if abs(truth['stamp'] - estimate['stamp']) <= MAX_POSE_PAIR_SKEW_SEC:
                return {'truth': truth, 'estimate': estimate}
        return None

    def observe(self, process):
        """Wait for evidence, publish exactly one query, then observe the full case."""
        start = time.monotonic()
        print('benchmark: waiting for sensors, map, controllers and stationary localization',
              flush=True)
        while time.monotonic() - start < self.case['startup_timeout_wall_sec']:
            self.ros_executor.spin_once(timeout_sec=.05)
            if process.poll() is not None:
                self.recording['termination'] = 'simulation_exited_before_ready'
                return
            problem = self._live_problem() or self._stationary_problem()
            self.recording['readiness']['last_problem'] = problem
            if not problem:
                break
        else:
            return
        if self.recording['target_poses'] or self.recording['goal_poses']:
            self.recording['termination'] = 'unsolicited_target_or_goal'
            return
        self.recording['readiness'].update(passed=True, wall_sec=time.monotonic() - start)
        alignment = self._paired_pose()
        if alignment is None:
            self.recording['termination'] = 'pose_pair_timeout'
            return
        self.recording['alignment'] = alignment
        # Keep a valid fallback for a terminal event that arrives between
        # truth/estimate callbacks; later synchronized pairs replace it.
        self.recording['final_robot'] = alignment
        if self.case['expected'] == 'start_validation':
            self.recording['termination'] = 'start_validation_complete'
            return
        world_from_odom = _pose_matrix(alignment['truth'], 'world') @ np.linalg.inv(
            _pose_matrix(alignment['estimate'], 'odom'))
        query_start, query_stamp = time.monotonic(), self.recording['sim_time']
        query = self.recording['query']
        query.update(sent=True, stamp=query_stamp, text=self.case['query'])
        self.query_publisher.publish(String(data=self.case['query']))
        print(f'benchmark: query sent: {self.case["query"]}', flush=True)
        negative = self.case['expected'] == 'no_target_or_goal'
        duration = (self.case['negative_observation_wall_sec'] if negative
                    else self.case['query_timeout_wall_sec'])
        self.recording['termination'] = 'query_timeout'
        while time.monotonic() - query_start < duration:
            self.ros_executor.spin_once(timeout_sec=.05)
            query.update(observed_wall_sec=time.monotonic() - query_start,
                         observed_sim_sec=self.recording['sim_time'] - query_stamp)
            # A delayed status-service response under simulation load is not
            # evidence that an already-active controller was deactivated.
            # Explicit inactive responses still fail closed.
            problem = self._live_problem(require_fresh_service_status=False)
            if process.poll() is not None or problem:
                self.recording['termination'] = problem or 'simulation_exited'
                break
            if not negative and any(e['event'] in ('SUCCEEDED', 'ABORTED', 'CANCELED')
                                    for e in self.recording['bridge_events']):
                self.recording['termination'] = 'nav_terminal'
                break
            paired = self._paired_pose()
            if paired is None:
                self.recording['pose_pair_skipped'] = (
                    self.recording.get('pose_pair_skipped', 0) + 1)
                continue
            self.recording['final_robot'] = paired
            estimate_world = (world_from_odom @ _pose_matrix(paired['estimate'], 'odom'))[:3, 3]
            error = math.dist(estimate_world, paired['truth']['position'])
            self.recording['max_robot_alignment_error_m'] = max(
                error, self.recording.get('max_robot_alignment_error_m', 0))
            if error > .5:
                self.recording['termination'] = 'localization_diverged_during_query'
                break
        else:
            if negative:
                self.recording['termination'] = 'observation_complete'
        query.update(observed_wall_sec=time.monotonic() - query_start,
                     observed_sim_sec=self.recording['sim_time'] - query_stamp)
        self.recording['wall_sec'] = time.monotonic() - start
        self.recording['trajectory'] = {name: list(values) for name, values in self.poses.items()}

    def close(self):
        """Close this observer without touching any external ROS process."""
        self.ros_executor.shutdown()
        self.navigation.destroy()
        self.destroy_node()
        self.ros_context.shutdown()
