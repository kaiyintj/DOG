"""ROS transport and bounded viewing actions for pre-navigation target confirmation."""

import json
import math
import time

from geometry_msgs.msg import Twist
import numpy as np
from rclpy.clock import Clock, ClockType
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException

from semantic_mapping.runtime.confirmation_visibility import inspect_view
from semantic_mapping.runtime.target_confirmation import ConfirmationSession


class TargetConfirmationAdapter:
    def __init__(self, node, policy, motion_topic):
        self.node, self.policy = node, policy
        self.motion = node.create_publisher(Twist, motion_topic, 10) if motion_topic else None
        self.status = node.create_publisher(String, '/semantic_confirmation/status', 10)
        self.navigation_subscription = node.create_subscription(
            String, '/nav_goal_bridge/status', self._navigation_status, 10)
        self.timer = node.create_timer(
            .05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.session = None
        self.geometry = None
        self.latest_map_sec = 0.0
        self.latest_source_ns = 0
        self.rejected_regions = []
        self.navigation_busy = False
        self.navigation_sequence = 0
        self.navigation_pending = False
        self.motion_owned = False
        self.last_heading = None
        self.turn_used = 0.0
        self.probe_heading = None
        self.latest_view = None
        self.waiting_for_input = False

    @staticmethod
    def candidate_snapshot(candidate):
        names = ('class_support_mean', 'uncertainty_mean', 'evidence', 'observed_age_sec',
                 'fresh_support_ratio', 'known_time_support_ratio', 'recent_class_support',
                 'position_shift_m', 'localization_valid')
        return {name: candidate.get(name) for name in names}

    def note_publication(self, candidate):
        """Record the no-feedback arm without changing its candidate selection."""
        if self.policy.mode != 'none':
            return
        value = {'event': 'GOAL_RELEASED', 'policy': 'none', 'state': 'executing',
                 'reason': 'feedback_disabled', 'sources_used': 0, 'attempts': 0,
                 'confirmation_sim_sec': 0.0, 'confirmation_wall_sec': 0.0,
                 'candidate_position': candidate['pos'].tolist(),
                 'candidate_before': self.candidate_snapshot(candidate)}
        self.status.publish(String(data=json.dumps(value, allow_nan=False)))

    def _navigation_status(self, message):
        payload = json.loads(message.data)
        sequence, event = payload['sequence'], payload['event']
        if event in ('RECEIVED', 'ACCEPTED') and sequence >= self.navigation_sequence:
            self.navigation_sequence, self.navigation_busy = sequence, True
            self.navigation_pending = False
        elif (not self.navigation_pending and sequence == self.navigation_sequence
              and event in ('SUCCEEDED', 'ABORTED', 'CANCELED', 'REJECTED', 'SEND_ERROR')):
            self.navigation_busy = False

    def emit(self, event):
        session = self.session
        value = {'event': event, 'policy': self.policy.mode, 'map_sec': self.latest_map_sec}
        if session is not None:
            value.update(state=session.status, reason=session.reason, attempts=session.attempts,
                         sources_used=session.sources_used,
                         valid_new_sources=session.valid_sources,
                         new_evidence=session.new_evidence,
                         confirmation_sim_sec=max(
                             0.0, self.node.get_clock().now().nanoseconds / 1e9
                             - session.start_map_sec),
                         confirmation_wall_sec=time.monotonic() - session.start_wall_sec,
                         candidate_position=(None if session.candidate is None
                                             else session.anchor.tolist()),
                         observed_position=None if session.position_weight <= 0 else
                         (session.position_mass / session.position_weight).tolist())
            if session.candidate is not None:
                value['candidate_before'] = self.candidate_snapshot(session.candidate)
                value['new_coverage_proxy'] = session.coverage
                value['turn_used_rad'] = self.turn_used
                value['waiting_for_input'] = self.waiting_for_input
                value['view'] = self.latest_view
                value['new_class_distribution'] = (
                    None if session.new_class_mass is None else
                    (session.new_class_mass / session.new_class_mass.sum()).tolist())
        self.status.publish(String(data=json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def stop_motion(self):
        if self.motion is not None and self.motion_owned:
            self.motion.publish(Twist())
        self.motion_owned = False

    def start(self, class_index, color):
        self.stop_motion()
        self.last_heading, self.probe_heading, self.latest_view = None, None, None
        self.turn_used = 0.0
        self.waiting_for_input = False
        self.class_index, self.color = class_index, color
        request_sec = self.node.get_clock().now().nanoseconds / 1e9
        self.session = ConfirmationSession(
            self.policy, class_index, request_sec, time.monotonic(),
            max(self.latest_source_ns, int(round(request_sec * 1e9))),
            unknown_index=self.node.profile.unknown_id)
        self.rejected_regions.clear()
        self.emit('STARTED')
        self.tick()

    def cancel(self, reason):
        self.stop_motion()
        if self.session is not None and self.session.status not in (
                'executing', 'unconfirmed', 'input_invalid'):
            self.session.status, self.session.reason = 'input_invalid', reason
            self.emit('TERMINATED')

    def clock_reset(self):
        self.stop_motion()
        if self.session is not None:
            self.session.status, self.session.reason = 'input_invalid', 'map_clock_restarted'
            self.emit('TERMINATED')
        self.geometry = None
        self.latest_map_sec = self.latest_source_ns = 0

    def on_observation(self, frame, updates):
        self.latest_map_sec = (frame['stamp_msg'].sec + frame['stamp_msg'].nanosec * 1e-9)
        self.latest_source_ns = max(self.latest_source_ns, frame['source_id'][1])
        self.geometry = frame.get('camera_geometry', self.geometry)
        session = self.session
        if session is None or session.status != 'observing':
            return
        view = self.view()
        if view is None:
            return
        contributions = []
        for key, (evidence, probabilities) in updates.items():
            position = self.node.voxel_map.voxels[key]['pos']
            if session.matches_regional_update(key, probabilities, position):
                contributions.append((key, evidence, probabilities, position))
        supported_positions = [value[3] for value in contributions
                               if value[2][self.class_index] >= self.policy.min_class_support]
        regional_view = (inspect_view(
                            supported_positions, self.geometry,
                            voxel_size=self.node.voxel_map.voxel_size)
                         if supported_positions else view)
        self.latest_view = {**view, 'new_regional_surface_ratio':
                            regional_view['surface_supported_ratio']}
        before = session.status
        session.observe(frame['source_id'], self.latest_map_sec, time.monotonic(), contributions,
                        regional_view['visible_support'])
        if session.status != before:
            self.emit('OBSERVATION_DECISION')

    def view(self):
        if self.geometry is None or self.session.candidate is None:
            return None
        positions = [self.node.voxel_map.voxels[key]['pos']
                     for key in self.session.candidate['voxel_keys']
                     if key in self.node.voxel_map.voxels]
        return None if not positions else inspect_view(
            positions, self.geometry, voxel_size=self.node.voxel_map.voxel_size)

    def current_camera_transform(self):
        if self.geometry is None:
            return None
        try:
            matrix = self.node._points_transform_matrix(
                self.node.odom_frame, self.geometry['source_frame'], Time())
        except TransformException:
            return None
        return matrix @ np.linalg.inv(self.geometry['lidar_to_camera'])

    def select_next(self):
        if self.session.expired(self.node.get_clock().now().nanoseconds / 1e9, time.monotonic()):
            self.session.status, self.session.reason = 'unconfirmed', 'cumulative_budget_exhausted'
            return
        if self.session.attempts >= self.policy.max_candidates:
            self.session.status, self.session.reason = 'unconfirmed', 'candidate_budget_exhausted'
            return
        self.node.select_class_query_target(self.class_index, self.color)
        candidates = self.node._last_query_clusters
        candidate = next((value for value in candidates if not any(
            np.linalg.norm(value['pos'] - position) <= self.policy.association_radius_m
            for position in self.rejected_regions)), None)
        if candidate is None:
            self.session.status = 'selecting'
            return
        candidate = dict(candidate)
        candidate['localization_valid'] = (self.current_camera_transform() is not None
                                           and self.node.get_robot_position() is not None)
        self.session.select(candidate, self.node.get_clock().now().nanoseconds / 1e9)
        self.probe_heading = None
        self.latest_view = None
        self.emit('CANDIDATE_SELECTED')

    def tick(self):
        session = self.session
        if session is None or session.status in ('executing', 'unconfirmed', 'input_invalid'):
            return
        current_sec = self.node.get_clock().now().nanoseconds / 1e9
        session.tick(current_sec, time.monotonic())
        if session.status == 'rejected':
            self.stop_motion()
            self.emit('CANDIDATE_REJECTED')
            self.rejected_regions.append(session.anchor.copy())
            session.status = 'selecting'
        if session.status == 'selecting':
            self.select_next()
        if session.status in ('unconfirmed', 'input_invalid'):
            self.stop_motion()
            self.node.query_retry_pending = False
            self.emit('TERMINATED')
            return
        if session.candidate is None:
            return
        if self.latest_map_sec <= 0 or current_sec - self.latest_map_sec > 2.5:
            self.stop_motion()
            # A short fusion/TF gap stops viewing motion, while the original
            # time/source/candidate limits continue. New input must confirm
            # again before a previously ready goal can be released.
            if session.status == 'confirmed':
                session.status = 'observing'
                session.min_valid_sources = max(
                    self.policy.min_new_sources, session.valid_sources + 1)
            session.reason = 'waiting_current_fusion_input'
            if not self.waiting_for_input:
                self.waiting_for_input = True
                self.emit('WAITING_INPUT')
            return
        if self.waiting_for_input:
            self.waiting_for_input = False
            self.emit('INPUT_RESUMED')
        map_from_camera = self.current_camera_transform()
        if map_from_camera is None or self.node.get_robot_position() is None:
            session.status, session.reason = 'input_invalid', 'current_view_tf_unavailable'
            self.stop_motion()
            self.emit('TERMINATED')
            return
        if session.status == 'confirmed':
            self.stop_motion()
            candidate = dict(session.candidate)
            if session.position_weight > 0:
                candidate['pos'] = session.position_mass / session.position_weight
                candidate['class_support_mean'] = (
                    session.new_class_mass[self.class_index] / session.new_class_mass.sum())
            if self.node._publish_query_goal(candidate, self.class_index, 'segformer'):
                session.status = 'executing'
                self.navigation_busy = True
                self.navigation_pending = True
                self.node.query_retry_pending = False
                self.emit('GOAL_RELEASED')
            else:
                session.status, session.reason = 'rejected', 'no_usable_approach'
                self.emit('CANDIDATE_REJECTED')
            return
        if session.status != 'observing':
            return
        view = self.view()
        if view is None:
            return
        forward = map_from_camera[:3, 2]
        heading = math.atan2(forward[1], forward[0])
        delta = session.anchor - map_from_camera[:3, 3]
        bearing = math.atan2(delta[1], delta[0])
        view['bearing_error_rad'] = math.atan2(
            math.sin(bearing - heading), math.cos(bearing - heading))
        if self.last_heading is not None:
            change = math.atan2(math.sin(heading - self.last_heading),
                                math.cos(heading - self.last_heading))
            self.turn_used += abs(change)
        self.last_heading = heading
        # An active navigation action retains command ownership. The pending
        # query may confirm from incoming views but never steers against it.
        if self.navigation_busy or self.motion is None:
            self.motion_owned = False
            return
        error = view['bearing_error_rad']
        if abs(error) <= .08 and session.sources_used >= 2:
            if self.probe_heading is None:
                self.probe_heading = heading + .2
            error = math.atan2(math.sin(self.probe_heading - heading),
                               math.cos(self.probe_heading - heading))
        command = Twist()
        # First collect post-request views before changing the camera bearing.
        if (session.sources_used >= self.policy.min_new_sources
                and abs(error) > .08 and self.turn_used < self.policy.max_turn_rad):
            command.angular.z = math.copysign(.35, error)
        self.motion.publish(command)
        self.motion_owned = True

    def close(self):
        self.stop_motion()
        self.node.destroy_timer(self.timer)
