import math
import pytest
from nav_msgs.msg import Odometry
from semantic_mapping.runtime.planar_odom_velocity import PlanarVelocityEstimator


def pose(t, x=0., y=0., angle=0.):
    m = Odometry()
    m.header.frame_id = 'odom'
    m.child_frame_id = 'base_link'
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = round((t-int(t))*1e9)
    m.pose.pose.position.x = float(x)
    m.pose.pose.position.y = float(y)
    m.pose.pose.orientation.z = math.sin(angle/2)
    m.pose.pose.orientation.w = math.cos(angle/2)
    return m


def test_body_frame_translation_and_preserved_input():
    e = PlanarVelocityEstimator()
    assert e.update(pose(1, angle=math.pi/2)) is None
    m = pose(1.2, y=.04, angle=math.pi/2)
    out = e.update(m)
    assert out.twist.twist.linear.x == pytest.approx(.2)
    assert out.twist.twist.linear.y == pytest.approx(0, abs=1e-10)
    assert m.twist.twist.linear.x == 0
    assert out.header == m.header
    assert out.pose == m.pose


def test_yaw_wrap():
    e = PlanarVelocityEstimator()
    e.update(pose(1, angle=math.pi-.01))
    out = e.update(pose(1.2, angle=-math.pi+.01))
    assert out.twist.twist.angular.z == pytest.approx(.1)


@pytest.mark.parametrize('t', [.1, 1., 3.])
def test_time_reset_or_gap_does_not_create_velocity_spike(t):
    e = PlanarVelocityEstimator()
    e.update(pose(1))
    assert e.update(pose(t, x=10)) is None
