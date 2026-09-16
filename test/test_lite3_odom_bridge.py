from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry

from semantic_mapping.runtime.lite3_odom_bridge_node import (
    Lite3OdomBridgeNode,
    OdometryRepairConfig,
    repair_lite3_odometry,
)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeBroadcaster:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


class FakeNow:
    def __init__(self, stamp):
        self.stamp = stamp

    def to_msg(self):
        return self.stamp


class FakeClock:
    def __init__(self, stamp):
        self.stamp = stamp

    def now(self):
        return FakeNow(self.stamp)


def test_repair_uses_receipt_time_and_isolated_frames_preserving_motion():
    raw = Odometry()
    raw.header.stamp.sec = 1728
    raw.header.stamp.nanosec = 919713344
    raw.pose.pose.position.x = 1.25
    raw.pose.pose.position.y = -0.5
    raw.pose.pose.orientation.z = 0.25
    raw.pose.pose.orientation.w = 0.968245837
    raw.twist.twist.linear.x = 0.3
    raw.twist.twist.linear.y = -0.1
    raw.twist.twist.angular.z = 0.2
    receipt_time = Time(sec=1788355530, nanosec=904881000)

    fixed, transform = repair_lite3_odometry(
        raw,
        receipt_time,
        OdometryRepairConfig(),
    )

    assert fixed.header.stamp == receipt_time
    assert fixed.header.frame_id == 'lite3_test_odom'
    assert fixed.child_frame_id == 'lite3_test_base_link'
    assert fixed.pose.pose == raw.pose.pose
    assert fixed.twist.twist == raw.twist.twist
    assert transform.header.stamp == receipt_time
    assert transform.header.frame_id == 'lite3_test_odom'
    assert transform.child_frame_id == 'lite3_test_base_link'
    assert transform.transform.translation.x == 1.25
    assert transform.transform.translation.y == -0.5
    assert transform.transform.rotation == raw.pose.pose.orientation
    assert raw.header.stamp.sec == 1728
    assert raw.header.frame_id == ''
    assert raw.child_frame_id == ''


def test_repair_replaces_vendor_zero_covariance_with_conservative_values():
    raw = Odometry()
    raw.pose.pose.orientation.w = 1.0

    fixed, _ = repair_lite3_odometry(
        raw,
        Time(sec=10),
        OdometryRepairConfig(),
    )

    diagonal = (0, 7, 14, 21, 28, 35)
    assert [fixed.pose.covariance[index] for index in diagonal] == [
        0.05, 0.05, 0.10, 0.05, 0.05, 0.10,
    ]
    assert [fixed.twist.covariance[index] for index in diagonal] == [
        0.05, 0.05, 0.10, 0.10, 0.10, 0.10,
    ]


def test_node_callback_publishes_only_fixed_odom_and_isolated_transform():
    receipt_time = Time(sec=1788355530, nanosec=904881000)
    node = object.__new__(Lite3OdomBridgeNode)
    node.config = OdometryRepairConfig()
    node.fixed_odom_publisher = FakePublisher()
    node.test_tf_broadcaster = FakeBroadcaster()
    node.get_clock = lambda: FakeClock(receipt_time)
    raw = Odometry()
    raw.pose.pose.orientation.w = 1.0

    node.odom_cb(raw)

    assert len(node.fixed_odom_publisher.messages) == 1
    assert len(node.test_tf_broadcaster.transforms) == 1
    assert node.fixed_odom_publisher.messages[0].header.stamp == receipt_time
    assert (
        node.test_tf_broadcaster.transforms[0].child_frame_id
        == 'lite3_test_base_link'
    )
