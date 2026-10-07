"""Geometry references retain units, transforms and explicit metric meanings."""

from pathlib import Path

import numpy as np
import pytest

from semantic_mapping.offline.static_geometry import closest_surface, load_dae_triangles


def test_collada_centimeters_and_scene_transform_are_both_applied(tmp_path):
    path = tmp_path / 'mesh.dae'
    path.write_text('''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
      <asset><unit meter="0.01"/><up_axis>Z_UP</up_axis></asset>
      <library_geometries><geometry id="g"><mesh>
        <source id="p"><float_array>0 0 0 100 0 0 0 100 0</float_array>
          <technique_common><accessor stride="3"/></technique_common></source>
        <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
        <triangles><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
      </mesh></geometry></library_geometries>
      <library_visual_scenes><visual_scene id="s"><node>
        <translate>10 0 0</translate><scale>0.1 0.1 0.1</scale>
        <instance_geometry url="#g"/>
      </node></visual_scene></library_visual_scenes>
      <scene><instance_visual_scene url="#s"/></scene></COLLADA>''')
    triangles = load_dae_triangles(path)
    np.testing.assert_allclose(triangles, [[[.1, 0, 0], [.2, 0, 0], [.1, .1, 0]]])


def test_surface_distance_is_not_a_bbox_membership_label():
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    point, distance = closest_surface([.2, .2, 1], triangles)
    np.testing.assert_allclose(point, [.2, .2, 0])
    assert distance == pytest.approx(1)
    point, _ = closest_surface([2, 0, 0], triangles)
    np.testing.assert_allclose(point, [1, 0, 0])


def test_real_chair_mesh_has_meter_scale_bounds():
    asset = Path(__file__).resolve().parents[2] / (
        'unitree-go2-ros2/robots/configs/go2_config/worlds/aws_robomaker/small_house/models/'
        'aws_robomaker_residential_ChairA_01/meshes/aws_ChairA_01_visual.DAE')
    if not asset.exists():
        pytest.skip('Small House assets are an external simulation dependency')
    triangles = load_dae_triangles(asset)
    extent = np.ptp(triangles.reshape(-1, 3), axis=0)
    assert np.all((extent > .2) & (extent < 1.5))
