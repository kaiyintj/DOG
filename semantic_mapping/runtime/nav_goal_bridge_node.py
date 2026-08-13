#!/usr/bin/env python3
"""Forward semantic PoseStamped goals to the Nav2 NavigateToPose action."""

import copy
from collections import deque
from dataclasses import dataclass
import json
import math
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class _PendingGoal:
    """Latest goal that has not yet been accepted by Nav2."""

    sequence: int
    pose: PoseStamped
    attempts: int = 0
    next_attempt_ns: int = 0


@dataclass
class _SendAttempt:
    """An action send request whose response is still outstanding."""

    token: int
    sequence: int
    started_ns: int


@dataclass
class _CancelAttempt:
    """A cancellation request for the current active goal."""

    token: int
    started_ns: int
    reason: str
    accepted_ns: int = 0


@dataclass
class _ActiveGoal:
    """The single Nav2 goal accepted by the action server."""

    sequence: int
    handle: object
    goal_id: str
    accepted_ns: int
    last_feedback_ns: int
    last_feedback_report_ns: int
    result_future: object = None
    cancel_attempt: object = None
    next_cancel_attempt_ns: int = 0
    timeout_triggered: bool = False


class NavGoalBridgeNode(Node):
    """Make semantic navigation independent of an RViz navigation panel."""

    def __init__(self):
        super().__init__('nav_goal_bridge_node')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter(
            'navigate_to_pose_action', '/navigate_to_pose')
        self.declare_parameter(
            'status_topic', '/nav_goal_bridge/status')
        self.declare_parameter('retry_period_sec', 0.5)
        self.declare_parameter('send_response_timeout_sec', 10.0)
        self.declare_parameter('goal_execution_timeout_sec', 300.0)
        self.declare_parameter('cancel_response_timeout_sec', 5.0)
        self.declare_parameter('cancel_result_timeout_sec', 10.0)
        self.declare_parameter('feedback_report_period_sec', 5.0)
        self.declare_parameter('result_history_size', 256)

        self.goal_pose_topic = str(
            self.get_parameter('goal_pose_topic').value)
        self.navigate_to_pose_action = str(
            self.get_parameter('navigate_to_pose_action').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        retry_period_sec = max(
            0.1, float(self.get_parameter('retry_period_sec').value))
        self.retry_period_ns = self._positive_duration_ns(
            retry_period_sec, 0.5)
        self.send_response_timeout_ns = self._positive_duration_ns(
            self.get_parameter('send_response_timeout_sec').value, 10.0)
        self.goal_execution_timeout_ns = self._positive_duration_ns(
            self.get_parameter('goal_execution_timeout_sec').value, 300.0)
        self.cancel_response_timeout_ns = self._positive_duration_ns(
            self.get_parameter('cancel_response_timeout_sec').value, 5.0)
        self.cancel_result_timeout_ns = self._positive_duration_ns(
            self.get_parameter('cancel_result_timeout_sec').value, 10.0)
        self.feedback_report_period_ns = self._positive_duration_ns(
            self.get_parameter('feedback_report_period_sec').value, 5.0)
        self.result_history_size = max(
            1, int(self.get_parameter('result_history_size').value))

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_to_pose_action,
        )
        self.status_publisher = self.create_publisher(
            String, self.status_topic, 10)
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            self.goal_pose_topic,
            self.goal_pose_cb,
            10,
        )
        self.retry_timer = self.create_timer(
            retry_period_sec, self.try_dispatch_pending_goal)

        self._initialize_transaction_state()

        self.get_logger().info(
            f'Nav2 目标桥接已启动: {self.goal_pose_topic} -> '
            f'{self.navigate_to_pose_action}, status={self.status_topic}, '
            f'goal_timeout={self.goal_execution_timeout_ns / 1e9:.1f}s')

    @staticmethod
    def _positive_duration_ns(value, fallback_sec):
        """Convert a positive finite duration in seconds to nanoseconds."""
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = float(fallback_sec)
        if not math.isfinite(seconds) or seconds <= 0.0:
            seconds = float(fallback_sec)
        return max(1, int(seconds * 1_000_000_000))

    def _initialize_transaction_state(self):
        """Initialize mutable action transaction state."""
        self.pending_goal = None
        self.send_attempt = None
        self.active_goal = None
        self.send_in_progress = False
        self.goal_sequence = 0
        self.latest_dispatched_sequence = 0
        self.last_server_warning_ns = None
        self._operation_token = 0
        self._send_timeout_warned_token = None
        self._handled_result_sequences = set()
        self._handled_result_order = deque()

    @staticmethod
    def _is_valid_goal(msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        values = (
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        quaternion_norm = math.sqrt(
            orientation.x ** 2
            + orientation.y ** 2
            + orientation.z ** 2
            + orientation.w ** 2
        )
        return (
            bool(msg.header.frame_id)
            and all(math.isfinite(value) for value in values)
            and quaternion_norm > 1e-6
        )

    @staticmethod
    def _goal_id_text(goal_handle):
        """Return the Nav2 UUID as stable hexadecimal text."""
        try:
            return bytes(goal_handle.goal_id.uuid).hex()
        except (AttributeError, TypeError, ValueError):
            return 'unknown'

    @staticmethod
    def _status_name(status):
        names = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        return names.get(status, f'STATUS_{status}')

    def _now_ns(self):
        """Use a steady clock so paused simulation time cannot stop safety."""
        return time.monotonic_ns()

    def _next_token(self):
        self._operation_token += 1
        return self._operation_token

    def _emit_status(self, sequence, event, detail='', goal_id=''):
        payload = {
            'sequence': int(sequence),
            'event': str(event),
        }
        if detail:
            payload['detail'] = str(detail)
        if goal_id:
            payload['nav2_goal_id'] = str(goal_id)
        message = String()
        message.data = json.dumps(
            payload, ensure_ascii=False, sort_keys=True)
        self.status_publisher.publish(message)

    def _remember_result_sequence(self, sequence):
        """Deduplicate result callbacks without unbounded memory growth."""
        if sequence in self._handled_result_sequences:
            return False
        if len(self._handled_result_order) >= self.result_history_size:
            oldest = self._handled_result_order.popleft()
            self._handled_result_sequences.discard(oldest)
        self._handled_result_order.append(sequence)
        self._handled_result_sequences.add(sequence)
        return True

    def goal_pose_cb(self, msg):
        if not self._is_valid_goal(msg):
            self.get_logger().error(
                '拒绝无效 /goal_pose：frame_id、坐标或四元数不合法。')
            return

        previous_pending = self.pending_goal
        self.goal_sequence += 1
        self.pending_goal = _PendingGoal(
            self.goal_sequence, copy.deepcopy(msg))
        if previous_pending is not None:
            self._emit_status(
                previous_pending.sequence,
                'SUPERSEDED',
                f'replaced_by={self.goal_sequence}',
            )
        self._emit_status(self.goal_sequence, 'RECEIVED')
        self.get_logger().info(
            f'收到语义导航目标 #{self.goal_sequence}: '
            f'frame={msg.header.frame_id}, '
            f'x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}')
        self.try_dispatch_pending_goal()

    def _warn_server_unavailable(self):
        now_ns = self._now_ns()
        if (
            self.last_server_warning_ns is None
            or now_ns - self.last_server_warning_ns >= 5_000_000_000
        ):
            self.get_logger().warn(
                f'Nav2 action {self.navigate_to_pose_action} 尚不可用；'
                '最新目标已缓存。请启动 nav_with_remap.launch.py。')
            self.last_server_warning_ns = now_ns

    def try_dispatch_pending_goal(self):
        """Advance the single-active/single-pending transaction state."""
        now_ns = self._now_ns()
        if self.send_attempt is not None:
            elapsed = now_ns - self.send_attempt.started_ns
            if (
                elapsed >= self.send_response_timeout_ns
                and self._send_timeout_warned_token
                != self.send_attempt.token
            ):
                self._send_timeout_warned_token = self.send_attempt.token
                self.get_logger().error(
                    f'发送 Nav2 目标 #{self.send_attempt.sequence} '
                    '等待响应超时；为避免产生两个活动目标，保持等待，'
                    '不并发重发。')
                self._emit_status(
                    self.send_attempt.sequence,
                    'SEND_RESPONSE_TIMEOUT',
                )
            return

        if self.active_goal is not None:
            self._service_active_goal(now_ns)
            return

        if self.pending_goal is None:
            return
        if now_ns < self.pending_goal.next_attempt_ns:
            return
        if not self.action_client.server_is_ready():
            self._warn_server_unavailable()
            return
        self._start_send_attempt(now_ns)

    def _start_send_attempt(self, now_ns):
        pending = self.pending_goal
        if pending is None:
            return
        pending.attempts += 1
        token = self._next_token()
        self.send_attempt = _SendAttempt(
            token, pending.sequence, now_ns)
        self.send_in_progress = True
        self._send_timeout_warned_token = None

        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(pending.pose)
        self._emit_status(
            pending.sequence,
            'DISPATCHING',
            f'attempt={pending.attempts}',
        )
        try:
            response_future = self.action_client.send_goal_async(
                goal,
                feedback_callback=(
                    lambda feedback, sequence=pending.sequence:
                    self.goal_feedback_cb(feedback, sequence)
                ),
            )
            response_future.add_done_callback(
                lambda future, sequence=pending.sequence, token=token:
                self.goal_response_cb(future, sequence, token))
        except Exception as exc:
            self._handle_send_failure(
                pending.sequence, token, f'{type(exc).__name__}: {exc}')

    def _handle_send_failure(self, sequence, token, detail):
        if (
            self.send_attempt is None
            or self.send_attempt.token != token
        ):
            return
        self.send_attempt = None
        self.send_in_progress = False
        if (
            self.pending_goal is not None
            and self.pending_goal.sequence == sequence
        ):
            self.pending_goal.next_attempt_ns = (
                self._now_ns() + self.retry_period_ns)
        self.get_logger().error(
            f'发送 Nav2 目标 #{sequence} 失败，目标保留等待重试: '
            f'{detail}')
        self._emit_status(sequence, 'SEND_ERROR', detail)

    def goal_response_cb(self, future, sequence, token=None):
        if token is None and self.send_attempt is not None:
            token = self.send_attempt.token
        if (
            self.send_attempt is None
            or token != self.send_attempt.token
            or sequence != self.send_attempt.sequence
        ):
            return

        self.send_attempt = None
        self.send_in_progress = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            if (
                self.pending_goal is not None
                and self.pending_goal.sequence == sequence
            ):
                self.pending_goal.next_attempt_ns = (
                    self._now_ns() + self.retry_period_ns)
            detail = f'{type(exc).__name__}: {exc}'
            self.get_logger().error(
                f'发送 Nav2 目标 #{sequence} 失败，目标保留等待重试: '
                f'{detail}')
            self._emit_status(sequence, 'SEND_ERROR', detail)
            return

        if not goal_handle.accepted:
            if (
                self.pending_goal is not None
                and self.pending_goal.sequence == sequence
            ):
                self.pending_goal.next_attempt_ns = (
                    self._now_ns() + self.retry_period_ns)
            self.get_logger().error(
                f'Nav2 拒绝目标 #{sequence}；保留最新目标等待重试。')
            self._emit_status(sequence, 'REJECTED')
            return

        now_ns = self._now_ns()
        goal_id = self._goal_id_text(goal_handle)
        self.active_goal = _ActiveGoal(
            sequence=sequence,
            handle=goal_handle,
            goal_id=goal_id,
            accepted_ns=now_ns,
            last_feedback_ns=now_ns,
            last_feedback_report_ns=now_ns,
        )
        if (
            self.pending_goal is not None
            and self.pending_goal.sequence == sequence
        ):
            self.pending_goal = None
        self.latest_dispatched_sequence = max(
            self.latest_dispatched_sequence, sequence)
        self.get_logger().info(
            f'Nav2 已接受目标 #{sequence}，goal_id={goal_id}。')
        self._emit_status(sequence, 'ACCEPTED', goal_id=goal_id)
        self._ensure_result_future()
        self.try_dispatch_pending_goal()

    def _ensure_result_future(self):
        active = self.active_goal
        if active is None or active.result_future is not None:
            return
        try:
            result_future = active.handle.get_result_async()
            active.result_future = result_future
            result_future.add_done_callback(
                lambda future, sequence=active.sequence,
                goal_id=active.goal_id:
                self.goal_result_cb(future, sequence, goal_id))
        except Exception as exc:
            self.get_logger().error(
                f'订阅 Nav2 目标 #{active.sequence} 结果失败，将重试: '
                f'{type(exc).__name__}: {exc}')
            active.result_future = None

    def _service_active_goal(self, now_ns):
        active = self.active_goal
        if active is None:
            return
        self._ensure_result_future()

        pending_preemption = (
            self.pending_goal is not None
            and self.pending_goal.sequence != active.sequence
        )
        execution_timed_out = (
            now_ns - active.accepted_ns
            >= self.goal_execution_timeout_ns
        )
        if execution_timed_out and not active.timeout_triggered:
            active.timeout_triggered = True
            self.get_logger().error(
                f'Nav2 目标 #{active.sequence} 执行超过 '
                f'{self.goal_execution_timeout_ns / 1e9:.1f}s，'
                '请求取消。')
            self._emit_status(
                active.sequence,
                'EXECUTION_TIMEOUT',
                goal_id=active.goal_id,
            )

        cancel = active.cancel_attempt
        if cancel is not None:
            if cancel.accepted_ns:
                cancel_timed_out = (
                    now_ns - cancel.accepted_ns
                    >= self.cancel_result_timeout_ns
                )
            else:
                cancel_timed_out = (
                    now_ns - cancel.started_ns
                    >= self.cancel_response_timeout_ns
                )
            if cancel_timed_out:
                self.get_logger().warn(
                    f'Nav2 目标 #{active.sequence} 取消确认超时，'
                    '保留活动目标并重试取消。')
                self._emit_status(
                    active.sequence,
                    'CANCEL_TIMEOUT',
                    goal_id=active.goal_id,
                )
                active.cancel_attempt = None
                active.next_cancel_attempt_ns = now_ns

        needs_cancel = pending_preemption or active.timeout_triggered
        if needs_cancel and active.cancel_attempt is None:
            reason = 'superseded' if pending_preemption else 'timeout'
            self._request_active_cancel(reason, now_ns)

    def _request_active_cancel(self, reason, now_ns=None):
        active = self.active_goal
        if active is None or active.cancel_attempt is not None:
            return
        if now_ns is None:
            now_ns = self._now_ns()
        if now_ns < active.next_cancel_attempt_ns:
            return

        token = self._next_token()
        active.cancel_attempt = _CancelAttempt(token, now_ns, reason)
        self.get_logger().info(
            f'请求取消 Nav2 目标 #{active.sequence}: {reason}。')
        self._emit_status(
            active.sequence,
            'CANCEL_REQUESTED',
            reason,
            active.goal_id,
        )
        try:
            cancel_future = active.handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda future, sequence=active.sequence, token=token:
                self.cancel_response_cb(future, sequence, token))
        except Exception as exc:
            self._handle_cancel_failure(
                active.sequence,
                token,
                f'{type(exc).__name__}: {exc}',
            )

    def _handle_cancel_failure(self, sequence, token, detail):
        active = self.active_goal
        if (
            active is None
            or active.sequence != sequence
            or active.cancel_attempt is None
            or active.cancel_attempt.token != token
        ):
            return
        active.cancel_attempt = None
        active.next_cancel_attempt_ns = self._now_ns() + self.retry_period_ns
        self.get_logger().error(
            f'取消 Nav2 目标 #{sequence} 请求失败，将重试: {detail}')
        self._emit_status(
            sequence, 'CANCEL_ERROR', detail, active.goal_id)

    def cancel_response_cb(self, future, sequence, token):
        active = self.active_goal
        if (
            active is None
            or active.sequence != sequence
            or active.cancel_attempt is None
            or active.cancel_attempt.token != token
        ):
            return
        try:
            response = future.result()
        except Exception as exc:
            self._handle_cancel_failure(
                sequence,
                token,
                f'{type(exc).__name__}: {exc}',
            )
            return

        if response.goals_canceling:
            active.cancel_attempt.accepted_ns = self._now_ns()
            self.get_logger().info(
                f'Nav2 已接受目标 #{sequence} 的取消请求；'
                '等待终态后再发送最新目标。')
            self._emit_status(
                sequence,
                'CANCEL_ACCEPTED',
                active.cancel_attempt.reason,
                active.goal_id,
            )
            return

        active.cancel_attempt = None
        active.next_cancel_attempt_ns = self._now_ns() + self.retry_period_ns
        self.get_logger().warn(
            f'Nav2 未接受目标 #{sequence} 的取消请求；'
            '保持活动目标并等待结果或重试。')
        self._emit_status(
            sequence, 'CANCEL_REJECTED', goal_id=active.goal_id)

    def goal_feedback_cb(self, feedback_message, sequence):
        active = self.active_goal
        if active is None or active.sequence != sequence:
            return
        now_ns = self._now_ns()
        active.last_feedback_ns = now_ns
        if (
            now_ns - active.last_feedback_report_ns
            < self.feedback_report_period_ns
        ):
            return
        active.last_feedback_report_ns = now_ns
        feedback = getattr(feedback_message, 'feedback', feedback_message)
        distance = getattr(feedback, 'distance_remaining', None)
        detail = ''
        if distance is not None and math.isfinite(float(distance)):
            detail = f'distance_remaining={float(distance):.3f}'
        self._emit_status(
            sequence, 'FEEDBACK', detail, active.goal_id)

    def goal_result_cb(self, future, sequence, goal_id=''):
        if sequence in self._handled_result_sequences:
            return
        try:
            status = future.result().status
        except Exception as exc:
            active = self.active_goal
            if active is not None and active.sequence == sequence:
                active.result_future = None
            self.get_logger().error(
                f'读取 Nav2 目标 #{sequence} 结果失败，将重试订阅: '
                f'{type(exc).__name__}: {exc}')
            return

        if not self._remember_result_sequence(sequence):
            return
        status_name = self._status_name(status)
        active = self.active_goal
        is_current = active is not None and active.sequence == sequence
        effective_goal_id = goal_id
        if is_current:
            effective_goal_id = active.goal_id

        log = (
            self.get_logger().info
            if status == GoalStatus.STATUS_SUCCEEDED
            else self.get_logger().warn
        )
        log(
            f'Nav2 目标 #{sequence} 结束: {status_name}, '
            f'goal_id={effective_goal_id or "unknown"}。')
        self._emit_status(
            sequence,
            status_name,
            goal_id=effective_goal_id,
        )

        if not is_current:
            return
        self.active_goal = None
        self.try_dispatch_pending_goal()


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalBridgeNode()
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
