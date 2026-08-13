import json
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

from semantic_mapping.runtime.nav_goal_bridge_node import NavGoalBridgeNode


class FakeFuture:
    def __init__(self, result=None, exception=None, done=False):
        self._result = result
        self._exception = exception
        self._done = done
        self._callbacks = []

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._done:
            callback(self)

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def resolve(self, result=None):
        self._result = result
        self._done = True
        for callback in list(self._callbacks):
            callback(self)

    def fail(self, exception):
        self._exception = exception
        self._done = True
        for callback in list(self._callbacks):
            callback(self)


class FakeGoalHandle:
    def __init__(self, accepted=True, uuid_seed=1):
        self.accepted = accepted
        self.goal_id = SimpleNamespace(
            uuid=[uuid_seed] * 16)
        self.result_future = FakeFuture()
        self.cancel_futures = []
        self.cancel_calls = 0
        self.result_calls = 0

    def get_result_async(self):
        self.result_calls += 1
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        future = FakeFuture()
        self.cancel_futures.append(future)
        return future


class FakeActionClient:
    def __init__(self):
        self.ready = True
        self.outcomes = []
        self.sent_goals = []
        self.feedback_callbacks = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, feedback_callback=None):
        self.sent_goals.append(goal)
        self.feedback_callbacks.append(feedback_callback)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeLogger:
    def __init__(self):
        self.records = []

    def _record(self, level, message):
        self.records.append((level, message))

    def debug(self, message):
        self._record('debug', message)

    def info(self, message):
        self._record('info', message)

    def warn(self, message):
        self._record('warn', message)

    def error(self, message):
        self._record('error', message)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


class FakeClock:
    def __init__(self):
        self.now_ns = 1_000

    def advance(self, nanoseconds):
        self.now_ns += nanoseconds


def make_goal(frame_id='odom', x=1.0, y=2.0, z=0.0, w=1.0):
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = w
    return goal


def make_node():
    node = object.__new__(NavGoalBridgeNode)
    clock = FakeClock()
    client = FakeActionClient()
    logger = FakeLogger()
    publisher = FakePublisher()
    node.action_client = client
    node.status_publisher = publisher
    node.navigate_to_pose_action = '/navigate_to_pose'
    node.retry_period_ns = 100
    node.send_response_timeout_ns = 1_000
    node.goal_execution_timeout_ns = 10_000
    node.cancel_response_timeout_ns = 500
    node.cancel_result_timeout_ns = 800
    node.feedback_report_period_ns = 200
    node.result_history_size = 4
    node.get_logger = lambda: logger
    node._now_ns = lambda: clock.now_ns
    node._initialize_transaction_state()
    return node, clock, client, logger, publisher


def accept_goal(node, client, pose, uuid_seed=1):
    send_future = FakeFuture()
    client.outcomes.append(send_future)
    node.goal_pose_cb(pose)
    handle = FakeGoalHandle(uuid_seed=uuid_seed)
    send_future.resolve(handle)
    return handle


def test_nav_goal_bridge_accepts_finite_pose_with_frame():
    assert NavGoalBridgeNode._is_valid_goal(make_goal())


def test_nav_goal_bridge_rejects_missing_frame_and_zero_quaternion():
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(frame_id=''))
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(w=0.0))


def test_nav_goal_bridge_rejects_non_finite_coordinate():
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(x=float('nan')))


def test_send_exception_keeps_latest_goal_and_retries():
    node, clock, client, _, publisher = make_node()
    client.outcomes.append(RuntimeError('transport unavailable'))

    node.goal_pose_cb(make_goal(x=4.0))

    assert node.pending_goal.sequence == 1
    assert node.pending_goal.pose.pose.position.x == 4.0
    assert node.pending_goal.attempts == 1
    assert node.send_attempt is None
    assert [item['event'] for item in publisher.messages].count(
        'SEND_ERROR') == 1

    retry_future = FakeFuture()
    client.outcomes.append(retry_future)
    node.try_dispatch_pending_goal()
    assert len(client.sent_goals) == 1

    clock.advance(node.retry_period_ns)
    node.try_dispatch_pending_goal()
    assert len(client.sent_goals) == 2
    assert node.pending_goal.sequence == 1

    handle = FakeGoalHandle(uuid_seed=3)
    retry_future.resolve(handle)
    assert node.pending_goal is None
    assert node.active_goal.sequence == 1


def test_async_send_failure_keeps_goal_for_retry():
    node, clock, client, _, _ = make_node()
    send_future = FakeFuture()
    client.outcomes.append(send_future)
    node.goal_pose_cb(make_goal(x=5.0))

    send_future.fail(RuntimeError('action response failed'))

    assert node.pending_goal.sequence == 1
    assert node.send_attempt is None
    replacement_future = FakeFuture()
    client.outcomes.append(replacement_future)
    clock.advance(node.retry_period_ns)
    node.try_dispatch_pending_goal()
    assert len(client.sent_goals) == 2


def test_new_goal_cancels_active_and_waits_for_terminal_result():
    node, _, client, _, _ = make_node()
    first_handle = accept_goal(node, client, make_goal(x=1.0))
    second_send = FakeFuture()
    client.outcomes.append(second_send)

    node.goal_pose_cb(make_goal(x=2.0))

    assert node.active_goal.sequence == 1
    assert node.pending_goal.sequence == 2
    assert first_handle.cancel_calls == 1
    assert len(client.sent_goals) == 1

    first_handle.cancel_futures[0].resolve(
        SimpleNamespace(goals_canceling=[object()]))
    assert node.active_goal.sequence == 1
    assert len(client.sent_goals) == 1

    first_handle.result_future.resolve(
        SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    assert node.active_goal is None
    assert node.pending_goal.sequence == 2
    assert len(client.sent_goals) == 2
    assert client.sent_goals[1].pose.pose.position.x == 2.0


def test_only_latest_pending_goal_is_dispatched_after_preemption():
    node, _, client, _, publisher = make_node()
    first_handle = accept_goal(node, client, make_goal(x=1.0))
    latest_send = FakeFuture()
    client.outcomes.append(latest_send)

    node.goal_pose_cb(make_goal(x=2.0))
    node.goal_pose_cb(make_goal(x=3.0))

    assert node.pending_goal.sequence == 3
    assert first_handle.cancel_calls == 1
    first_handle.cancel_futures[0].resolve(
        SimpleNamespace(goals_canceling=[object()]))
    first_handle.result_future.resolve(
        SimpleNamespace(status=GoalStatus.STATUS_CANCELED))

    sent_positions = [
        goal.pose.pose.position.x for goal in client.sent_goals]
    assert sent_positions == [1.0, 3.0]
    assert any(
        item['event'] == 'SUPERSEDED' and item['sequence'] == 2
        for item in publisher.messages
    )


def test_cancel_error_retains_active_and_pending_then_retries():
    node, clock, client, _, _ = make_node()
    first_handle = accept_goal(node, client, make_goal(x=1.0))
    node.goal_pose_cb(make_goal(x=2.0))
    first_handle.cancel_futures[0].fail(RuntimeError('cancel failed'))

    assert node.active_goal.sequence == 1
    assert node.pending_goal.sequence == 2
    assert node.active_goal.cancel_attempt is None

    clock.advance(node.retry_period_ns)
    node.try_dispatch_pending_goal()
    assert first_handle.cancel_calls == 2


def test_execution_and_cancel_timeouts_never_dispatch_concurrently():
    node, clock, client, _, publisher = make_node()
    first_handle = accept_goal(node, client, make_goal(x=1.0))

    clock.advance(node.goal_execution_timeout_ns)
    node.try_dispatch_pending_goal()
    assert first_handle.cancel_calls == 1
    assert node.active_goal.sequence == 1

    first_handle.cancel_futures[0].resolve(
        SimpleNamespace(goals_canceling=[object()]))
    clock.advance(node.cancel_result_timeout_ns)
    node.try_dispatch_pending_goal()

    assert first_handle.cancel_calls == 2
    assert node.active_goal.sequence == 1
    events = [item['event'] for item in publisher.messages]
    assert events.count('EXECUTION_TIMEOUT') == 1
    assert 'CANCEL_TIMEOUT' in events


def test_send_response_timeout_reports_once_without_unsafe_resend():
    node, clock, client, _, publisher = make_node()
    client.outcomes.append(FakeFuture())
    node.goal_pose_cb(make_goal(x=7.0))

    clock.advance(node.send_response_timeout_ns)
    node.try_dispatch_pending_goal()
    node.try_dispatch_pending_goal()

    assert len(client.sent_goals) == 1
    assert node.pending_goal.sequence == 1
    assert [item['event'] for item in publisher.messages].count(
        'SEND_RESPONSE_TIMEOUT') == 1


def test_duplicate_and_stale_results_do_not_clear_new_active_goal():
    node, _, client, _, publisher = make_node()
    first_handle = accept_goal(node, client, make_goal(x=1.0))
    second_send = FakeFuture()
    client.outcomes.append(second_send)
    node.goal_pose_cb(make_goal(x=2.0))
    first_handle.cancel_futures[0].resolve(
        SimpleNamespace(goals_canceling=[object()]))
    first_handle.result_future.resolve(
        SimpleNamespace(status=GoalStatus.STATUS_CANCELED))

    second_handle = FakeGoalHandle(uuid_seed=2)
    second_send.resolve(second_handle)
    assert node.active_goal.sequence == 2

    duplicate_result = FakeFuture(
        result=SimpleNamespace(status=GoalStatus.STATUS_ABORTED),
        done=True,
    )
    node.goal_result_cb(duplicate_result, 1, 'old-id')

    assert node.active_goal.sequence == 2
    terminal_events = [
        item for item in publisher.messages
        if item['sequence'] == 1
        and item['event'] in {'CANCELED', 'ABORTED', 'SUCCEEDED'}
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0]['event'] == 'CANCELED'


def test_feedback_is_throttled_and_bound_to_active_sequence():
    node, clock, client, _, publisher = make_node()
    accept_goal(node, client, make_goal(x=1.0))
    feedback = SimpleNamespace(
        feedback=SimpleNamespace(distance_remaining=2.5))

    node.goal_feedback_cb(feedback, 99)
    node.goal_feedback_cb(feedback, 1)
    clock.advance(node.feedback_report_period_ns)
    node.goal_feedback_cb(feedback, 1)

    feedback_events = [
        item for item in publisher.messages
        if item['event'] == 'FEEDBACK'
    ]
    assert len(feedback_events) == 1
    assert feedback_events[0]['sequence'] == 1
    assert feedback_events[0]['detail'] == 'distance_remaining=2.500'


def test_result_history_is_bounded():
    node, _, _, _, _ = make_node()
    for sequence in range(1, 7):
        assert node._remember_result_sequence(sequence)

    assert len(node._handled_result_sequences) == 4
    assert list(node._handled_result_order) == [3, 4, 5, 6]
    assert node._remember_result_sequence(1)
