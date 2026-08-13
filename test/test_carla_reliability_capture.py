import struct

import numpy as np
import pytest

from semantic_mapping.carla.carla_capture_reliability import (
    SEMANTIC_LIDAR_DTYPE,
    _validate_timing,
    channel_indices,
    decode_normal_lidar,
    decode_semantic_lidar,
    discover_reliability_semantic_tags,
    measurement_record,
)


def test_normal_lidar_decoder_preserves_xyz_and_intensity():
    raw = np.asarray([
        [1.0, 2.0, 3.0, 0.25],
        [-4.0, 5.0, 6.0, 0.75],
    ], dtype='<f4').tobytes()

    xyz, intensity = decode_normal_lidar(raw)

    assert xyz.dtype == np.float32
    assert xyz.tolist() == [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]
    assert intensity.tolist() == [0.25, 0.75]


def test_semantic_lidar_decoder_uses_carla_24_byte_layout():
    assert SEMANTIC_LIDAR_DTYPE.itemsize == 24
    raw = struct.pack('<ffffII', 1.0, -2.0, 3.5, 0.8, 70001, 14)

    decoded = decode_semantic_lidar(raw)

    assert np.allclose(decoded['xyz'], [[1.0, -2.0, 3.5]])
    assert np.allclose(decoded['cos_inc_angle'], [0.8])
    assert decoded['object_idx'].tolist() == [70001]
    assert decoded['object_tag'].tolist() == [14]


def test_channel_indices_reconstruct_carla_channel_order():
    class Measurement:
        channels = 3

        @staticmethod
        def get_point_count(channel):
            return (2, 0, 3)[channel]

    channels, counts = channel_indices(Measurement(), 5)

    assert channels.tolist() == [0, 0, 2, 2, 2]
    assert counts.tolist() == [2, 0, 3]


def test_channel_indices_reject_mismatched_payload():
    class Measurement:
        channels = 2

        @staticmethod
        def get_point_count(channel):
            return (2, 2)[channel]

    with pytest.raises(ValueError, match='do not match'):
        channel_indices(Measurement(), 3)


def test_measurement_record_has_evaluator_schema():
    class Vector:
        x = 1.0
        y = 2.0
        z = 3.0

    class Rotation:
        pitch = 4.0
        yaw = 5.0
        roll = 6.0

    class Transform:
        location = Vector()
        rotation = Rotation()

        @staticmethod
        def get_matrix():
            return np.eye(4).tolist()

    class Measurement:
        frame = 42
        timestamp = 2.1
        transform = Transform()

    record = measurement_record(Measurement(), 'lidar/00000042.npz')

    assert record['frame'] == 42
    assert record['timestamp'] == 2.1
    assert record['path'] == 'lidar/00000042.npz'
    assert record['transform_matrix'] == np.eye(4).tolist()


def test_timing_rejects_unrepresentable_sensor_ticks():
    class Args:
        fixed_delta = 0.01
        camera_tick = 0.02
        lidar_tick = 0.05

    _validate_timing(Args())
    Args.camera_tick = 0.015
    with pytest.raises(ValueError, match='integer multiple'):
        _validate_timing(Args())


def test_reliability_tag_discovery_includes_static_classes():
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class CityObjectLabel:
        Roads = EnumValue(1)
        Sidewalks = EnumValue(2)
        Buildings = EnumValue(3)
        Walls = EnumValue(4)
        Fences = EnumValue(5)
        Vegetation = EnumValue(9)
        Pedestrians = EnumValue(12)
        Car = EnumValue(14)
        Truck = EnumValue(15)
        Bus = EnumValue(16)
        Motorcycle = EnumValue(18)
        Bicycle = EnumValue(19)
        Rider = EnumValue(13)

    Carla = type('Carla', (), {'CityObjectLabel': CityObjectLabel})

    tags = discover_reliability_semantic_tags(Carla)

    assert tags['sidewalk'] == [2]
    assert tags['building'] == [3]
    assert tags['wall'] == [4]
    assert tags['fence'] == [5]
    assert tags['tree'] == [9]
    assert tags['vegetation'] == [9]
