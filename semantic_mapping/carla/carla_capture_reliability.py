#!/usr/bin/env python3
"""Capture synchronized CARLA data for point-level reliability evaluation."""

import argparse
import csv
import json
from pathlib import Path
import queue
import random
import time

import numpy as np
from PIL import Image

from semantic_mapping.carla.carla_benchmark import (
    SPAWN_CLASSES,
    decode_carla_bgra,
    decode_carla_instance,
    decode_carla_semantic,
)
from semantic_mapping.carla.carla_capture_benchmark import (
    _select_ego_blueprint,
    _select_ego_transform,
    _spawn_vehicles,
    _spawn_walkers,
    discover_semantic_tags,
)


SEMANTIC_LIDAR_DTYPE = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('cos_inc_angle', '<f4'),
    ('object_idx', '<u4'),
    ('object_tag', '<u4'),
])

WEATHER_FIELDS = (
    'cloudiness',
    'precipitation',
    'precipitation_deposits',
    'wind_intensity',
    'sun_azimuth_angle',
    'sun_altitude_angle',
    'fog_density',
    'fog_distance',
    'fog_falloff',
    'wetness',
    'scattering_intensity',
    'mie_scattering_scale',
    'rayleigh_scattering_scale',
)

RELIABILITY_SEMANTIC_ENUM_CANDIDATES = {
    'road': ('Roads', 'Road'),
    'road_line': ('RoadLines', 'RoadLine'),
    'sidewalk': ('Sidewalks', 'Sidewalk'),
    'building': ('Buildings', 'Building'),
    'wall': ('Walls', 'Wall'),
    'fence': ('Fences', 'Fence'),
    'pole': ('Poles', 'Pole'),
    'traffic_light': ('TrafficLight', 'TrafficLights'),
    'traffic_sign': ('TrafficSigns', 'TrafficSign'),
    'tree': ('Vegetation',),
    'vegetation': ('Vegetation',),
    'terrain': ('Terrain',),
    'sky': ('Sky',),
    'ground': ('Ground',),
    'bridge': ('Bridge',),
    'rail_track': ('RailTrack',),
    'guard_rail': ('GuardRail',),
    'water': ('Water',),
    'static': ('Static',),
    'dynamic': ('Dynamic',),
    'other': ('Other',),
}


def decode_normal_lidar(raw_data):
    """Decode normal CARLA ray-cast LiDAR into xyz and intensity arrays."""
    values = np.frombuffer(raw_data, dtype='<f4')
    if values.size % 4:
        raise ValueError(
            'Normal LiDAR payload must contain four float32 values per point')
    points = values.reshape(-1, 4)
    return points[:, :3].copy(), points[:, 3].copy()


def discover_reliability_semantic_tags(carla):
    """Discover mobile and static CARLA labels used by the reliability GT."""
    result = discover_semantic_tags(carla)
    enum_type = carla.CityObjectLabel
    for canonical_name, candidates in (
        RELIABILITY_SEMANTIC_ENUM_CANDIDATES.items()
    ):
        values = []
        for candidate in candidates:
            if hasattr(enum_type, candidate):
                enum_item = getattr(enum_type, candidate)
                values.append(int(getattr(enum_item, 'value', enum_item)))
        result[canonical_name] = sorted(set(values))
    return result


def decode_semantic_lidar(raw_data):
    """Decode CARLA Semantic LiDAR without changing tags or actor IDs."""
    if len(raw_data) % SEMANTIC_LIDAR_DTYPE.itemsize:
        raise ValueError(
            'Semantic LiDAR payload size is not a whole number of points')
    values = np.frombuffer(raw_data, dtype=SEMANTIC_LIDAR_DTYPE)
    xyz = np.column_stack((values['x'], values['y'], values['z'])).astype(
        np.float32,
        copy=False,
    )
    return {
        'xyz': np.ascontiguousarray(xyz),
        'cos_inc_angle': values['cos_inc_angle'].copy(),
        'object_idx': values['object_idx'].copy(),
        'object_tag': values['object_tag'].copy(),
    }


def channel_indices(measurement, point_count):
    """Reconstruct per-point laser channel from CARLA channel counts."""
    counts = np.asarray([
        int(measurement.get_point_count(channel))
        for channel in range(int(measurement.channels))
    ], dtype=np.int64)
    if np.any(counts < 0) or int(np.sum(counts)) != int(point_count):
        raise ValueError(
            'LiDAR channel counts do not match decoded point count: '
            f'{counts.tolist()} vs {point_count}')
    channels = np.repeat(
        np.arange(counts.size, dtype=np.uint16),
        counts,
    )
    return channels, counts


def transform_to_dict(transform):
    """Serialize a CARLA transform and its local-to-world matrix."""
    return {
        'location_m': {
            'x': float(transform.location.x),
            'y': float(transform.location.y),
            'z': float(transform.location.z),
        },
        'rotation_deg': {
            'pitch': float(transform.rotation.pitch),
            'yaw': float(transform.rotation.yaw),
            'roll': float(transform.rotation.roll),
        },
        'matrix_local_to_world': [
            [float(value) for value in row]
            for row in transform.get_matrix()
        ],
    }


def measurement_record(measurement, path=None):
    """Return common frame, simulation-time and pose metadata."""
    transform = transform_to_dict(measurement.transform)
    result = {
        'frame': int(measurement.frame),
        'timestamp': float(measurement.timestamp),
        'timestamp_sec': float(measurement.timestamp),
        'transform_matrix': transform['matrix_local_to_world'],
        'transform': transform,
    }
    if path is not None:
        result['path'] = str(path)
    return result


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _set_attribute(blueprint, name, value, required=True):
    if not blueprint.has_attribute(name):
        if required:
            raise RuntimeError(
                f'Blueprint {blueprint.id} has no required attribute {name}')
        return
    blueprint.set_attribute(name, str(value))


def _camera_blueprint(library, blueprint_id, args):
    blueprint = library.find(blueprint_id)
    _set_attribute(blueprint, 'image_size_x', args.width)
    _set_attribute(blueprint, 'image_size_y', args.height)
    _set_attribute(blueprint, 'fov', args.fov)
    _set_attribute(blueprint, 'sensor_tick', args.camera_tick)
    return blueprint


def _lidar_blueprint(library, blueprint_id, args, normal_lidar):
    blueprint = library.find(blueprint_id)
    attributes = {
        'channels': args.lidar_channels,
        'range': args.lidar_range,
        'points_per_second': args.lidar_points_per_second,
        'rotation_frequency': args.lidar_rotation_frequency,
        'upper_fov': args.lidar_upper_fov,
        'lower_fov': args.lidar_lower_fov,
        'sensor_tick': args.lidar_tick,
    }
    for name, value in attributes.items():
        _set_attribute(blueprint, name, value)
    if normal_lidar:
        _set_attribute(blueprint, 'noise_stddev', 0.0, required=False)
        _set_attribute(blueprint, 'dropoff_general_rate', 0.0)
        _set_attribute(blueprint, 'dropoff_intensity_limit', 1.0)
        _set_attribute(blueprint, 'dropoff_zero_intensity', 0.0)
        _set_attribute(blueprint, 'noise_seed', args.seed, required=False)
    return blueprint


def _imu_blueprint(library, args):
    blueprint = library.find('sensor.other.imu')
    _set_attribute(blueprint, 'sensor_tick', args.fixed_delta)
    for name in (
        'noise_accel_stddev_x',
        'noise_accel_stddev_y',
        'noise_accel_stddev_z',
        'noise_gyro_stddev_x',
        'noise_gyro_stddev_y',
        'noise_gyro_stddev_z',
        'noise_gyro_bias_x',
        'noise_gyro_bias_y',
        'noise_gyro_bias_z',
    ):
        _set_attribute(blueprint, name, 0.0, required=False)
    _set_attribute(blueprint, 'noise_seed', args.seed, required=False)
    return blueprint


def _blueprint_attributes(blueprint):
    return {
        str(name): str(value)
        for name, value in blueprint.attributes.items()
    }


def _weather_from_name(carla, name):
    requested = str(name).strip()
    candidates = {
        attr.lower(): attr
        for attr in dir(carla.WeatherParameters)
        if not attr.startswith('_')
    }
    resolved = candidates.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f'Unknown CARLA weather preset {requested!r}; available presets: '
            + ', '.join(sorted(candidates.values())))
    return resolved, getattr(carla.WeatherParameters, resolved)


def _weather_to_dict(weather):
    return {
        name: float(getattr(weather, name))
        for name in WEATHER_FIELDS
        if hasattr(weather, name)
    }


def _motion_profile(args):
    return 'stationary' if args.stationary_ego else args.ego_motion_profile


def _apply_ego_motion(carla, ego, profile, args):
    if profile == 'stationary':
        ego.set_target_velocity(carla.Vector3D())
        ego.set_target_angular_velocity(carla.Vector3D())
        ego.apply_control(carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
        ))
    elif profile == 'constant_velocity':
        forward = ego.get_transform().get_forward_vector()
        speed = float(args.ego_speed)
        ego.set_target_velocity(carla.Vector3D(
            x=forward.x * speed,
            y=forward.y * speed,
            z=forward.z * speed,
        ))
    elif profile == 'turning':
        ego.apply_control(carla.VehicleControl(
            throttle=float(args.ego_throttle),
            steer=float(args.ego_steer),
        ))


def _drain_queues(sensor_queues, quiet_timeout=0.02):
    result = {name: [] for name in sensor_queues}
    deadline = time.monotonic() + float(quiet_timeout)
    while True:
        received = False
        for name, sensor_queue in sensor_queues.items():
            while True:
                try:
                    result[name].append(sensor_queue.get_nowait())
                    received = True
                except queue.Empty:
                    break
        if received:
            deadline = time.monotonic() + float(quiet_timeout)
        if time.monotonic() >= deadline:
            return result
        time.sleep(0.001)


def _vector_to_list(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


def _save_camera(output, name, measurement, args):
    frame_name = f'{int(measurement.frame):08d}.png'
    relative = f'{name}/{frame_name}'
    if name == 'rgb':
        values = decode_carla_bgra(
            measurement.raw_data, args.width, args.height)
        Image.fromarray(values, mode='RGB').save(output / relative)
    elif name == 'semantic_camera':
        values = decode_carla_semantic(
            measurement.raw_data, args.width, args.height)
        Image.fromarray(values, mode='L').save(output / relative)
    else:
        values = decode_carla_instance(
            measurement.raw_data, args.width, args.height)
        Image.fromarray(values, mode='I;16').save(output / relative)
    return measurement_record(measurement, relative)


def _save_normal_lidar(output, measurement):
    xyz, intensity = decode_normal_lidar(measurement.raw_data)
    channels, point_counts = channel_indices(measurement, len(xyz))
    relative = f'lidar/{int(measurement.frame):08d}.npz'
    np.savez_compressed(
        output / relative,
        xyz=xyz,
        intensity=intensity,
        channel=channels,
        point_count_by_channel=point_counts,
    )
    record = measurement_record(measurement, relative)
    record.update({
        'point_count': int(len(xyz)),
        'channels': int(measurement.channels),
        'horizontal_angle_rad': float(measurement.horizontal_angle),
        'point_count_by_channel': point_counts.tolist(),
    })
    return record


def _save_semantic_lidar(output, measurement):
    decoded = decode_semantic_lidar(measurement.raw_data)
    channels, point_counts = channel_indices(
        measurement, len(decoded['xyz']))
    relative = f'semantic_lidar/{int(measurement.frame):08d}.npz'
    np.savez_compressed(
        output / relative,
        xyz=decoded['xyz'],
        cos_inc_angle=decoded['cos_inc_angle'],
        object_idx=decoded['object_idx'],
        object_tag=decoded['object_tag'],
        channel=channels,
        point_count_by_channel=point_counts,
    )
    record = measurement_record(measurement, relative)
    record.update({
        'point_count': int(len(decoded['xyz'])),
        'channels': int(measurement.channels),
        'horizontal_angle_rad': float(measurement.horizontal_angle),
        'point_count_by_channel': point_counts.tolist(),
    })
    return record


def _imu_record(measurement):
    result = measurement_record(measurement)
    result.update({
        'accelerometer_mps2': _vector_to_list(measurement.accelerometer),
        'gyroscope_radps': _vector_to_list(measurement.gyroscope),
        'compass_rad': float(measurement.compass),
    })
    return result


def _ego_record_from_snapshot(snapshot):
    """Serialize the ego state from the exact post-tick world snapshot."""
    return {
        'transform': transform_to_dict(snapshot.get_transform()),
        'velocity_mps': _vector_to_list(snapshot.get_velocity()),
        'angular_velocity_degps': _vector_to_list(
            snapshot.get_angular_velocity()),
        'acceleration_mps2': _vector_to_list(snapshot.get_acceleration()),
    }


def _validate_timing(args):
    for name in ('fixed_delta', 'camera_tick', 'lidar_tick'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('camera_tick', 'lidar_tick'):
        ratio = getattr(args, name) / args.fixed_delta
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-6):
            raise ValueError(
                f'{name} must be an integer multiple of fixed_delta')
    if args.camera_tick > args.lidar_tick:
        raise ValueError('camera_tick must not exceed lidar_tick')
    positive_fields = (
        'lidar_channels',
        'lidar_range',
        'lidar_points_per_second',
        'lidar_rotation_frequency',
    )
    for name in positive_fields:
        if not hasattr(args, name):
            continue
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
    if (
        hasattr(args, 'lidar_lower_fov')
        and hasattr(args, 'lidar_upper_fov')
        and not args.lidar_lower_fov < args.lidar_upper_fov
    ):
        raise ValueError('lidar_lower_fov must be below lidar_upper_fov')


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Capture a CARLA point-level reliability dataset.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--traffic-manager-port', type=int, default=8000)
    parser.add_argument('--town', '--map', dest='town', default='Town05')
    parser.add_argument('--weather', default='ClearNoon')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-frames', '--frames', dest='num_frames',
                        type=int, default=300)
    parser.add_argument('--warmup-frames', type=int, default=40)
    parser.add_argument(
        '--output-dir', '--output', dest='output_dir',
        default=(
            'carla_benchmark_data/reliability_'
            + time.strftime('%Y%m%d_%H%M%S')),
    )
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--sensor-timeout', type=float, default=10.0)
    parser.add_argument('--fixed-delta', type=float, default=0.01)
    parser.add_argument('--camera-tick', type=float, default=0.01)
    parser.add_argument('--lidar-tick', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--lidar-channels', type=int, default=64)
    parser.add_argument('--lidar-range', type=float, default=50.0)
    parser.add_argument('--lidar-points-per-second', type=int, default=600000)
    parser.add_argument('--lidar-rotation-frequency', type=float, default=20.0)
    parser.add_argument('--lidar-upper-fov', type=float, default=10.0)
    parser.add_argument('--lidar-lower-fov', type=float, default=-30.0)
    parser.add_argument('--vehicles', type=int, default=24)
    parser.add_argument('--walkers', type=int, default=12)
    parser.add_argument(
        '--vehicle-classes', nargs='+', choices=SPAWN_CLASSES,
        default=list(SPAWN_CLASSES))
    parser.add_argument(
        '--moving-targets', action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable or disable AI motion for spawned target actors.')
    parser.add_argument('--stationary-ego', action='store_true')
    parser.add_argument(
        '--ego-motion-profile',
        choices=('stationary', 'autopilot', 'constant_velocity', 'turning'),
        default='autopilot',
    )
    parser.add_argument('--ego-speed', type=float, default=5.0)
    parser.add_argument('--ego-throttle', type=float, default=0.45)
    parser.add_argument('--ego-steer', type=float, default=0.35)
    return parser


def run_capture(args):
    """Run the deterministic CARLA reliability capture."""
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            'The CARLA Python API is not importable. Install the matching '
            'CARLA 0.9.16 wheel first.') from exc

    _validate_timing(args)
    if args.num_frames <= 0:
        raise ValueError('num_frames must be positive')
    if args.width <= 0 or args.height <= 0:
        raise ValueError('image dimensions must be positive')
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f'Output directory is not empty: {output}. Use a new run name.')
    for name in (
        'rgb',
        'semantic_camera',
        'instance_camera',
        'lidar',
        'semantic_lidar',
        'imu',
        'metadata',
    ):
        (output / name).mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output.parent / '.carla_cache'
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_files_base_folder(str(cache_dir))
    client.set_timeout(float(args.sensor_timeout))
    world = client.load_world(args.town) if args.town else client.get_world()
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    spawned_actors = []
    controllers = []
    sensors = []

    imu_csv_path = output / 'imu' / 'imu.csv'
    imu_file = imu_csv_path.open('w', newline='', encoding='utf-8')
    imu_writer = csv.writer(imu_file)
    imu_writer.writerow([
        'frame', 'timestamp_sec',
        'accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2',
        'gyro_x_radps', 'gyro_y_radps', 'gyro_z_radps',
        'compass_rad',
    ])

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta)
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(args.seed))
        weather_name, weather = _weather_from_name(carla, args.weather)
        world.set_weather(weather)
        random.seed(args.seed)

        library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('Selected CARLA map has no vehicle spawn points')
        ego_transform = _select_ego_transform(spawn_points, args.seed)
        ego_blueprint = _select_ego_blueprint(library)
        if ego_blueprint.has_attribute('role_name'):
            ego_blueprint.set_attribute('role_name', 'hero')
        ego = world.try_spawn_actor(ego_blueprint, ego_transform)
        if ego is None:
            raise RuntimeError('Could not spawn the reliability ego vehicle')
        spawned_actors.append(ego)

        vehicles, descriptions, blueprint_classes, targets = _spawn_vehicles(
            world,
            library,
            spawn_points,
            ego_transform,
            args.vehicles,
            args.seed,
            args.vehicle_classes,
        )
        spawned_actors.extend(vehicles)
        walkers, controllers = _spawn_walkers(
            world,
            library,
            ego_transform,
            args.walkers,
            args.seed,
        )
        spawned_actors.extend(walkers)
        spawned_actors.extend(controllers)

        world.tick()
        if args.moving_targets:
            for controller in controllers:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(1.2 + random.random() * 0.6)
            for vehicle in vehicles:
                vehicle.set_autopilot(True, args.traffic_manager_port)
        else:
            for vehicle in vehicles:
                vehicle.set_target_velocity(carla.Vector3D())
                vehicle.set_target_angular_velocity(carla.Vector3D())
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                ))

        profile = _motion_profile(args)
        if profile == 'autopilot':
            ego.set_autopilot(True, args.traffic_manager_port)

        camera_transform = carla.Transform(
            carla.Location(x=1.3, z=1.65),
            carla.Rotation(pitch=-5.0),
        )
        lidar_transform = carla.Transform(carla.Location(x=1.3, z=1.80))
        imu_transform = carla.Transform()
        blueprints = {
            'rgb': _camera_blueprint(
                library, 'sensor.camera.rgb', args),
            'semantic_camera': _camera_blueprint(
                library, 'sensor.camera.semantic_segmentation', args),
            'instance_camera': _camera_blueprint(
                library, 'sensor.camera.instance_segmentation', args),
            'lidar': _lidar_blueprint(
                library, 'sensor.lidar.ray_cast', args, True),
            'semantic_lidar': _lidar_blueprint(
                library, 'sensor.lidar.ray_cast_semantic', args, False),
            'imu': _imu_blueprint(library, args),
        }
        attachment_transforms = {
            'rgb': camera_transform,
            'semantic_camera': camera_transform,
            'instance_camera': camera_transform,
            'lidar': lidar_transform,
            'semantic_lidar': lidar_transform,
            'imu': imu_transform,
        }
        sensor_queues = {}
        for name, blueprint in blueprints.items():
            sensor = world.spawn_actor(
                blueprint,
                attachment_transforms[name],
                attach_to=ego,
            )
            sensor_queue = queue.Queue()
            sensor.listen(sensor_queue.put)
            sensors.append(sensor)
            sensor_queues[name] = sensor_queue

        sensor_manifest = {
            name: {
                'blueprint_id': str(blueprint.id),
                'attributes': _blueprint_attributes(blueprint),
                'attachment_transform_ego': transform_to_dict(
                    attachment_transforms[name]),
            }
            for name, blueprint in blueprints.items()
        }
        manifest = {
            'format': 'carla_reliability_v1',
            'format_version': 1,
            'benchmark': 'carla_reliability',
            'carla_server_version': str(client.get_server_version()),
            'carla_client_version': str(client.get_client_version()),
            'town': str(world.get_map().name),
            'weather_preset': weather_name,
            'weather': _weather_to_dict(world.get_weather()),
            'seed': int(args.seed),
            'synchronous_mode': True,
            'fixed_delta_seconds': float(args.fixed_delta),
            'coordinate_convention': (
                'CARLA sensor-local/world coordinates: x forward, y right, '
                'z up; transforms are local-to-world'),
            'normal_lidar_ground_truth_policy': (
                'Normal and semantic LiDAR are independent sensors. Their '
                'array indices are not asserted to correspond; evaluator '
                'must perform validated ray/spatial matching.'),
            'ego_motion_profile': profile,
            'moving_targets': bool(args.moving_targets),
            'semantic_tag_ids': discover_reliability_semantic_tags(carla),
            'vehicle_blueprint_classes': blueprint_classes,
            'vehicle_class_targets': targets,
            'vehicle_actors': descriptions,
            'actor_id_map': {
                str(item['actor_id']): {
                    'type_id': item['type_id'],
                    'kind': 'vehicle',
                    'benchmark_class': item['benchmark_class'],
                }
                for item in descriptions
            },
            'sensors': sensor_manifest,
            'rgb_index': 'rgb_index.json',
            'imu_csv': 'imu/imu.csv',
            'frames': [],
        }
        manifest['actor_id_map'].update({
            str(walker.id): {
                'type_id': str(walker.type_id),
                'kind': 'walker',
                'benchmark_class': 'person',
            }
            for walker in walkers
        })
        _write_json(output / 'manifest.json', manifest)

        for _ in range(max(0, int(args.warmup_frames))):
            _apply_ego_motion(carla, ego, profile, args)
            world.tick()
            _drain_queues(sensor_queues)

        camera_records = {
            'rgb': {},
            'semantic_camera': {},
            'instance_camera': {},
        }
        rgb_index = []
        imu_records = {}
        ego_records = {}
        pending_lidar = {}
        pending_semantic_lidar = {}
        saved = 0
        lidar_stride = max(1, int(round(args.lidar_tick / args.fixed_delta)))
        tick_limit = int(args.num_frames) * lidar_stride * 4 + 100

        for _ in range(tick_limit):
            if saved >= int(args.num_frames):
                break
            _apply_ego_motion(carla, ego, profile, args)
            world_frame = int(world.tick())
            ego_snapshot = world.get_snapshot().find(ego.id)
            if ego_snapshot is None:
                raise RuntimeError(
                    f'Ego actor is absent from world snapshot {world_frame}')
            ego_records[world_frame] = _ego_record_from_snapshot(ego_snapshot)
            batches = _drain_queues(sensor_queues)

            for name in camera_records:
                for measurement in batches[name]:
                    record = _save_camera(output, name, measurement, args)
                    camera_records[name][int(measurement.frame)] = record
                    if name == 'rgb':
                        rgb_index.append(record)

            for measurement in batches['imu']:
                record = _imu_record(measurement)
                frame = int(measurement.frame)
                imu_records[frame] = record
                accel = record['accelerometer_mps2']
                gyro = record['gyroscope_radps']
                imu_writer.writerow([
                    frame,
                    record['timestamp_sec'],
                    *accel,
                    *gyro,
                    record['compass_rad'],
                ])

            for measurement in batches['lidar']:
                pending_lidar[int(measurement.frame)] = measurement
            for measurement in batches['semantic_lidar']:
                pending_semantic_lidar[int(measurement.frame)] = measurement

            eligible = sorted(
                set(pending_lidar).intersection(pending_semantic_lidar)
                .intersection(camera_records['rgb'])
                .intersection(camera_records['semantic_camera'])
                .intersection(camera_records['instance_camera'])
                .intersection(imu_records)
                .intersection(ego_records)
            )
            for frame in eligible:
                if saved >= int(args.num_frames):
                    break
                normal = pending_lidar.pop(frame)
                semantic_lidar = pending_semantic_lidar.pop(frame)
                timestamps = [
                    float(normal.timestamp),
                    float(semantic_lidar.timestamp),
                    camera_records['rgb'][frame]['timestamp_sec'],
                    camera_records['semantic_camera'][frame]['timestamp_sec'],
                    camera_records['instance_camera'][frame]['timestamp_sec'],
                    imu_records[frame]['timestamp_sec'],
                ]
                if max(timestamps) - min(timestamps) > 1e-6:
                    raise RuntimeError(
                        f'Same-frame sensor timestamps differ at frame {frame}: '
                        f'{timestamps}')
                normal_record = _save_normal_lidar(output, normal)
                semantic_record = _save_semantic_lidar(
                    output, semantic_lidar)
                frame_metadata = {
                    'frame': int(frame),
                    'timestamp': float(normal.timestamp),
                    'capture_index': int(saved),
                    'world_frame': int(frame),
                    'simulation_timestamp_sec': float(normal.timestamp),
                    'rgb': camera_records['rgb'][frame],
                    'semantic_camera': camera_records[
                        'semantic_camera'][frame],
                    'instance_camera': camera_records[
                        'instance_camera'][frame],
                    'lidar': normal_record,
                    'semantic_lidar': semantic_record,
                    'imu': imu_records[frame],
                    'ego': ego_records[frame],
                    'synchronization': {
                        'all_sensor_frames_equal': True,
                        'max_timestamp_delta_sec': float(
                            max(timestamps) - min(timestamps)),
                        'last_world_tick_seen': world_frame,
                    },
                }
                relative = f'metadata/{frame:08d}.json'
                _write_json(output / relative, frame_metadata)
                manifest['frames'].append(relative)
                saved += 1
                if saved % 10 == 0:
                    print(
                        f'Captured {saved}/{args.num_frames} reliability '
                        f'frames; CARLA frame={frame}, '
                        f'normal_points={normal_record["point_count"]}, '
                        f'semantic_points={semantic_record["point_count"]}',
                        flush=True,
                    )

            retention_ticks = max(
                100,
                int(np.ceil(1.0 / float(args.fixed_delta))),
            )
            for records in (
                *camera_records.values(),
                imu_records,
                ego_records,
            ):
                stale = [
                    key for key in records
                    if key < world_frame - retention_ticks
                    and key not in pending_lidar
                    and key not in pending_semantic_lidar
                ]
                for key in stale:
                    records.pop(key, None)

        if saved != int(args.num_frames):
            raise RuntimeError(
                f'Captured only {saved}/{args.num_frames} complete frames; '
                'check sensor ticks and CARLA callback timing')
        imu_file.flush()
        _write_json(output / 'rgb_index.json', rgb_index)
        manifest['saved_frame_count'] = int(saved)
        manifest['rgb_frame_count'] = int(len(rgb_index))
        _write_json(output / 'manifest.json', manifest)
        print(f'CARLA reliability dataset saved to {output}', flush=True)
        return 0
    finally:
        imu_file.close()
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        actor_ids = []
        for actor in reversed(spawned_actors + sensors):
            try:
                actor_ids.append(int(actor.id))
            except RuntimeError:
                pass
        if actor_ids:
            commands = [
                carla.command.DestroyActor(actor_id)
                for actor_id in dict.fromkeys(actor_ids)
            ]
            try:
                client.apply_batch_sync(commands, True)
            except RuntimeError:
                pass
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.set_weather(original_weather)
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def main(argv=None):
    """Run the CARLA reliability capture console program."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_capture(args)
    except Exception as exc:
        print(f'CARLA reliability capture failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
