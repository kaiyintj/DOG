"""Visual-ray attribution accepts two-sided faces and declared sensor transforms."""

import numpy as np

from semantic_mapping.offline.static_attribution import declared_projection, first_ray_hit


def test_ray_returns_nearest_forward_surface_including_back_faces():
    triangle = np.array([[1, -1, -1], [1, 1, -1], [1, 0, 1]], dtype=float)
    triangles = np.array([triangle + [1, 0, 0], triangle[::-1]])
    assert first_ray_hit(np.zeros(3), np.array([1, 0, 0]), triangles) == 1
    assert np.isinf(first_ray_hit(np.zeros(3), np.array([-1, 0, 0]), triangles))


def test_projection_uses_inverse_camera_chain_and_forward_lidar_chain(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="fixed"><link name="base_link"/>
      <joint name="camera" type="fixed"><parent link="base_link"/>
        <child link="camera_color_optical_frame"/><origin xyz=".3 0 .14"/></joint>
      <joint name="lidar" type="fixed"><parent link="base_link"/>
        <child link="mid360_link"/><origin xyz=".1 0 .08"/></joint></robot>''')
    matrix = declared_projection(path)
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
    np.testing.assert_allclose(matrix[:3, 3], [-.2, 0, -.06])
