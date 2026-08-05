from geometry_msgs.msg import PoseStamped

from semantic_mapping.nav_goal_bridge_node import NavGoalBridgeNode


def make_goal(frame_id='odom', x=1.0, y=2.0, z=0.0, w=1.0):
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = w
    return goal


def test_nav_goal_bridge_accepts_finite_pose_with_frame():
    assert NavGoalBridgeNode._is_valid_goal(make_goal())


def test_nav_goal_bridge_rejects_missing_frame_and_zero_quaternion():
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(frame_id=''))
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(w=0.0))


def test_nav_goal_bridge_rejects_non_finite_coordinate():
    assert not NavGoalBridgeNode._is_valid_goal(make_goal(x=float('nan')))
