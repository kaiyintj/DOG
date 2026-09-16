from std_msgs.msg import String

from semantic_mapping.runtime.clip_query import SemanticQuery


class CapturingPublisher:
    def __init__(self):
        self.subscription_count = 0
        self.messages = []

    def get_subscription_count(self):
        return self.subscription_count

    def publish(self, message):
        self.messages.append(message)


class CapturingTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class SilentLogger:
    def info(self, unused_message):
        pass


def test_query_waits_for_ga_subscription_before_one_shot_publish():
    node = object.__new__(SemanticQuery)
    node.pub = CapturingPublisher()
    node.timer = CapturingTimer()
    node.sent = False
    node.wait_reported = False
    node.get_logger = lambda: SilentLogger()
    message = String(data='chair')

    node._send_once(message)

    assert node.pub.messages == []
    assert not node.sent
    assert not node.timer.cancelled

    node.pub.subscription_count = 1
    node._send_once(message)
    node._send_once(message)

    assert node.pub.messages == [message]
    assert node.sent
    assert node.timer.cancelled
