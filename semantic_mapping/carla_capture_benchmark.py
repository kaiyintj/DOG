#!/usr/bin/env python3
"""Capture synchronized CARLA RGB, semantic and instance ground truth."""

import argparse
import json
from pathlib import Path
import queue
import random
import sys
import time

import numpy as np
from PIL import Image

from semantic_mapping.carla_benchmark import (
    CARLA_ENUM_CANDIDATES,
    SPAWN_CLASSES,
    TWO_WHEEL_CLASSES,
    actor_instance_id,
    bbox_from_mask,
    canonical_color_from_rgb,
    decode_carla_bgra,
    decode_carla_instance,
    decode_carla_semantic,
    parse_carla_color,
)


COLOR_PALETTE = (
    ('red', '220,20,20'),
    ('blue', '25,70,220'),
    ('yellow', '235,195,20'),
    ('green', '25,150,55'),
    ('white', '230,230,230'),
    ('black', '20,20,20'),
    ('gray', '110,110,110'),
)


def _enum_value(enum_item):
    value = getattr(enum_item, 'value', enum_item)
    return int(value)


def discover_semantic_tags(carla):
    """Discover semantic IDs across old and current CARLA releases."""
    result = {}
    enum_type = carla.CityObjectLabel
    for canonical_name, candidates in CARLA_ENUM_CANDIDATES.items():
        values = []
        for candidate in candidates:
            if hasattr(enum_type, candidate):
                values.append(_enum_value(getattr(enum_type, candidate)))
        result[canonical_name] = sorted(set(values))
    if not result['car'] and result['vehicle']:
        result['car'] = list(result['vehicle'])
    return result


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _sensor_blueprint(library, blueprint_id, width, height, fov):
    blueprint = library.find(blueprint_id)
    blueprint.set_attribute('image_size_x', str(width))
    blueprint.set_attribute('image_size_y', str(height))
    blueprint.set_attribute('fov', str(fov))
    blueprint.set_attribute('sensor_tick', '0.0')
    return blueprint


def _get_frame(sensor_queue, target_frame, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            data = sensor_queue.get(timeout=remaining)
        except queue.Empty:
            break
        if data.frame == target_frame:
            return data
        if data.frame > target_frame:
            raise RuntimeError(
                f'Sensor skipped frame {target_frame}; next frame={data.frame}.')
    raise TimeoutError(f'Timed out waiting for sensor frame {target_frame}.')


def _distance_squared(transform, origin):
    delta = transform.location - origin.location
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z


def _forward_spawn_candidates(
    spawn_points,
    origin,
    min_distance=8.0,
    max_distance=80.0,
    half_angle_deg=40.0,
):
    """Return spawn points lying inside the camera-facing road cone."""
    forward = origin.get_forward_vector()
    tangent_limit = np.tan(np.deg2rad(float(half_angle_deg)))
    candidates = []
    for transform in spawn_points:
        delta = transform.location - origin.location
        distance_squared = (
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
        )
        if not min_distance ** 2 <= distance_squared <= max_distance ** 2:
            continue
        longitudinal = delta.x * forward.x + delta.y * forward.y
        lateral = abs(forward.x * delta.y - forward.y * delta.x)
        if longitudinal <= 0.0 or lateral > longitudinal * tangent_limit:
            continue
        candidates.append(transform)
    return sorted(
        candidates,
        key=lambda transform: _distance_squared(transform, origin),
    )


def _select_ego_transform(spawn_points, seed):
    """Choose a deterministic camera origin with many forward spawn points."""
    ordered = list(spawn_points)
    random.Random(seed).shuffle(ordered)
    return max(
        ordered,
        key=lambda transform: len(
            _forward_spawn_candidates(spawn_points, transform)
        ),
    )


def _select_ego_blueprint(library):
    preferred = (
        'vehicle.tesla.model3',
        'vehicle.lincoln.mkz_2020',
        'vehicle.audi.tt',
    )
    for blueprint_id in preferred:
        try:
            return library.find(blueprint_id)
        except (IndexError, RuntimeError):
            continue
    candidates = list(library.filter('vehicle.*'))
    if not candidates:
        raise RuntimeError('CARLA map has no vehicle blueprints.')
    return candidates[0]


def _blueprint_wheels(blueprint):
    """Return the wheel count of a vehicle blueprint, or ``None`` if absent."""
    if blueprint.has_attribute('number_of_wheels'):
        return int(blueprint.get_attribute('number_of_wheels').as_int())
    return None


def _benchmark_vehicle_class(blueprint):
    """Infer a stable benchmark class from a vehicle blueprint.

    Two-wheelers are decided first from ``base_type`` and wheel count, because
    a motorcycle id can otherwise contain a substring that looks like a car.
    """
    type_id = str(blueprint.id).lower()
    base_type = ''
    if blueprint.has_attribute('base_type'):
        base_type = str(blueprint.get_attribute('base_type')).lower()
    text = f'{base_type} {type_id}'
    if 'bicycle' in text or _blueprint_wheels(blueprint) == 2 and any(
        token in text for token in ('bike', 'omafiets', 'century', 'gazelle')
    ):
        return 'bicycle'
    if 'motorcycle' in text or 'motorbike' in text:
        return 'motorcycle'
    if any(token in text for token in ('bus', 'coach', 'fusorosa')):
        return 'bus'
    if any(token in text for token in (
        'truck',
        'lorry',
        'firetruck',
        'carlacola',
        'sprinter',
    )):
        return 'truck'
    return 'car'


def _vehicle_blueprints(library):
    grouped = {name: [] for name in SPAWN_CLASSES}
    for blueprint in library.filter('vehicle.*'):
        if not blueprint.has_attribute('color'):
            continue
        wheels = _blueprint_wheels(blueprint)
        benchmark_class = _benchmark_vehicle_class(blueprint)
        is_two_wheeler = benchmark_class in TWO_WHEEL_CLASSES
        if wheels is not None:
            if is_two_wheeler and wheels != 2:
                continue
            if not is_two_wheeler and wheels != 4:
                continue
        grouped[benchmark_class].append(blueprint)
    for values in grouped.values():
        values.sort(key=lambda blueprint: blueprint.id)

    return grouped, {
        class_name: [blueprint.id for blueprint in values]
        for class_name, values in grouped.items()
    }


def _balanced_vehicle_schedule(count, class_names):
    """Build a deterministic round-robin class schedule."""
    names = list(dict.fromkeys(class_names))
    if int(count) <= 0 or not names:
        return []
    return [names[index % len(names)] for index in range(int(count))]


def _spawn_vehicles(
    world,
    library,
    spawn_points,
    ego_transform,
    count,
    seed,
    class_names=SPAWN_CLASSES,
):
    grouped, blueprint_classes = _vehicle_blueprints(library)
    available_classes = [
        name for name in class_names if grouped.get(name)
    ]
    if not available_classes:
        raise RuntimeError('No color-configurable four-wheel vehicles found.')

    nearby = _forward_spawn_candidates(spawn_points, ego_transform)
    requested_schedule = _balanced_vehicle_schedule(count, class_names)
    schedule = [
        name for name in requested_schedule if grouped.get(name)
    ]

    actors = []
    descriptions = []
    transform_index = 0
    class_attempts = {name: 0 for name in available_classes}
    for desired_class in schedule:
        class_blueprints = grouped[desired_class]
        actor = None
        selected_blueprint = None
        selected_color = None
        while transform_index < len(nearby) and actor is None:
            transform = nearby[transform_index]
            transform_index += 1
            for _ in range(len(class_blueprints)):
                blueprint_index = (
                    class_attempts[desired_class]
                    + seed
                ) % len(class_blueprints)
                blueprint = class_blueprints[blueprint_index]
                color_name, color_rgb = COLOR_PALETTE[
                    (len(actors) + seed) % len(COLOR_PALETTE)
                ]
                blueprint.set_attribute('color', color_rgb)
                if blueprint.has_attribute('role_name'):
                    blueprint.set_attribute(
                        'role_name',
                        f'benchmark_{desired_class}_{len(actors)}',
                    )
                class_attempts[desired_class] += 1
                actor = world.try_spawn_actor(blueprint, transform)
                if actor is not None:
                    selected_blueprint = blueprint
                    selected_color = (color_name, color_rgb)
                    break
        if actor is None:
            break
        color_name, color_rgb = selected_color
        actors.append(actor)
        descriptions.append({
            'actor_id': int(actor.id),
            'type_id': str(actor.type_id),
            'benchmark_class': _benchmark_vehicle_class(
                selected_blueprint),
            'requested_color_name': color_name,
            'requested_color_rgb': [
                int(value) for value in parse_carla_color(color_rgb)
            ],
        })
    target_counts = {
        name: requested_schedule.count(name)
        for name in dict.fromkeys(class_names)
    }
    return actors, descriptions, blueprint_classes, target_counts


def _spawn_walkers(
    world,
    library,
    origin_transform,
    count,
    seed,
    min_distance=6.0,
    max_distance=35.0,
    half_angle_deg=38.0,
):
    random_generator = random.Random(seed + 1009)
    walker_blueprints = list(library.filter('walker.pedestrian.*'))
    controller_blueprint = library.find('controller.ai.walker')
    forward = origin_transform.get_forward_vector()
    tangent_limit = np.tan(np.deg2rad(float(half_angle_deg)))
    walkers = []
    controllers = []
    attempts = 0
    while len(walkers) < count and attempts < max(100, count * 100):
        attempts += 1
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        delta = location - origin_transform.location
        distance = np.hypot(delta.x, delta.y)
        if distance < min_distance or distance > max_distance:
            continue
        longitudinal = delta.x * forward.x + delta.y * forward.y
        lateral = abs(forward.x * delta.y - forward.y * delta.x)
        if longitudinal <= 0.0 or lateral > longitudinal * tangent_limit:
            continue
        blueprint = random_generator.choice(walker_blueprints)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        walker = world.try_spawn_actor(
            blueprint,
            type(origin_transform)(location=location),
        )
        if walker is None:
            continue
        controller = world.try_spawn_actor(
            controller_blueprint,
            type(origin_transform)(),
            attach_to=walker,
        )
        walkers.append(walker)
        if controller is not None:
            controllers.append(controller)
    return walkers, controllers


def _visible_instances(
    actors,
    actor_descriptions,
    instance_mask,
    semantic_mask,
    accepted_tags,
    min_pixels,
):
    descriptions_by_id = {
        int(item['actor_id']): item for item in actor_descriptions
    }
    result = []
    for actor in actors:
        instance_id = actor_instance_id(actor.id)
        object_mask = instance_mask == instance_id
        accepted_mask = object_mask & np.isin(
            semantic_mask, list(accepted_tags))
        pixel_count = int(np.count_nonzero(accepted_mask))
        if pixel_count < min_pixels:
            continue
        bbox = bbox_from_mask(object_mask)
        if bbox is None:
            continue
        tag_values, tag_counts = np.unique(
            semantic_mask[object_mask], return_counts=True)
        semantic_tag = int(tag_values[int(np.argmax(tag_counts))])
        item = {
            'actor_id': int(actor.id),
            'type_id': str(actor.type_id),
            'bbox_xyxy': [int(value) for value in bbox],
            'instance_id': int(instance_id),
            'visible_pixels': int(np.count_nonzero(object_mask)),
            'semantic_tag': semantic_tag,
            'instance_match_method': 'actor_id',
        }
        item.update(descriptions_by_id.get(int(actor.id), {}))
        result.append(item)
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Capture CARLA RGB plus exact semantic and instance ground truth '
            'for CLIP/SegFormer image-recognition evaluation.'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--traffic-manager-port', type=int, default=8000)
    parser.add_argument('--map', default='Town05')
    parser.add_argument(
        '--output',
        default=(
            'carla_benchmark_data/run_'
            + time.strftime('%Y%m%d_%H%M%S')),
    )
    parser.add_argument(
        '--cache-dir',
        default='',
        help=(
            'CARLA client map-cache directory. The default is '
            '<output-parent>/.carla_cache so the cache never depends on the '
            'shell working directory.'),
    )
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--warmup-frames', type=int, default=40)
    parser.add_argument('--vehicles', type=int, default=24)
    parser.add_argument(
        '--vehicle-classes',
        nargs='+',
        choices=SPAWN_CLASSES,
        default=list(SPAWN_CLASSES),
        help=(
            'Vehicle classes to balance in round-robin order. The default '
            'captures car, truck, bus, bicycle and motorcycle with nearly '
            'equal target counts. Bicycles and motorcycles spawn a rider.'),
    )
    parser.add_argument('--walkers', type=int, default=18)
    parser.add_argument('--walker-min-distance', type=float, default=6.0)
    parser.add_argument('--walker-max-distance', type=float, default=35.0)
    parser.add_argument('--walker-half-angle', type=float, default=38.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument(
        '--grid-rows',
        type=int,
        default=4,
        help='CLIP grid rows recorded in the dataset manifest.',
    )
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=6,
        help='CLIP grid columns recorded in the dataset manifest.',
    )
    parser.add_argument('--fixed-delta', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-instance-pixels', type=int, default=40)
    parser.add_argument('--sensor-timeout', type=float, default=10.0)
    parser.add_argument(
        '--stationary-ego',
        action='store_true',
        help='Keep the camera vehicle parked instead of using autopilot.')
    return parser


def run_capture(args):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            'The CARLA Python API is not importable. Install the wheel from '
            'CARLA/PythonAPI/carla/dist or add its egg to PYTHONPATH.'
        ) from exc

    if args.grid_rows <= 0 or args.grid_cols <= 0:
        raise ValueError('grid_rows and grid_cols must both be positive.')

    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f'Output directory is not empty: {output}. Use a new run name.')
    for name in ('rgb', 'semantic', 'instance', 'metadata'):
        (output / name).mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output.parent / '.carla_cache'
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_files_base_folder(str(cache_dir))
    client.set_timeout(args.sensor_timeout)
    world = client.load_world(args.map) if args.map else client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    spawned_actors = []
    walker_controllers = []
    sensors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('Selected CARLA map has no vehicle spawn points.')
        ego_transform = _select_ego_transform(spawn_points, args.seed)
        ego_blueprint = _select_ego_blueprint(library)
        if ego_blueprint.has_attribute('role_name'):
            ego_blueprint.set_attribute('role_name', 'hero')
        ego = world.try_spawn_actor(ego_blueprint, ego_transform)
        if ego is None:
            raise RuntimeError('Could not spawn the benchmark camera vehicle.')
        spawned_actors.append(ego)

        (
            vehicles,
            vehicle_descriptions,
            vehicle_blueprint_classes,
            vehicle_class_targets,
        ) = (
            _spawn_vehicles(
                world,
                library,
                spawn_points,
                ego_transform,
                args.vehicles,
                args.seed,
                args.vehicle_classes,
            )
        )
        spawned_actors.extend(vehicles)

        walkers, walker_controllers = _spawn_walkers(
            world,
            library,
            ego_transform,
            args.walkers,
            args.seed,
            args.walker_min_distance,
            args.walker_max_distance,
            args.walker_half_angle,
        )
        spawned_actors.extend(walkers)
        spawned_actors.extend(walker_controllers)

        world.tick()
        for controller in walker_controllers:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(1.2 + random.random() * 0.6)

        if not args.stationary_ego:
            ego.set_autopilot(True, args.traffic_manager_port)
        for vehicle in vehicles:
            vehicle.set_autopilot(True, args.traffic_manager_port)

        camera_transform = carla.Transform(
            carla.Location(x=1.3, z=1.65),
            carla.Rotation(pitch=-5.0),
        )
        sensor_specs = {
            'rgb': 'sensor.camera.rgb',
            'semantic': 'sensor.camera.semantic_segmentation',
            'instance': 'sensor.camera.instance_segmentation',
        }
        sensor_queues = {}
        for name, blueprint_id in sensor_specs.items():
            blueprint = _sensor_blueprint(
                library, blueprint_id, args.width, args.height, args.fov)
            sensor = world.spawn_actor(
                blueprint, camera_transform, attach_to=ego)
            sensor_queue = queue.Queue()
            sensor.listen(sensor_queue.put)
            sensors.append(sensor)
            sensor_queues[name] = sensor_queue

        semantic_tag_ids = discover_semantic_tags(carla)
        manifest = {
            'format_version': 2,
            'instance_id_encoding': 'actor_id=G+(B<<8)',
            'carla_version': str(client.get_server_version()),
            'client_version': str(client.get_client_version()),
            'map': str(world.get_map().name),
            'seed': int(args.seed),
            'image': {
                'width': int(args.width),
                'height': int(args.height),
                'fov_deg': float(args.fov),
            },
            'semantic_tag_ids': semantic_tag_ids,
            'vehicle_blueprint_classes': vehicle_blueprint_classes,
            'requested_vehicle_classes': list(args.vehicle_classes),
            'vehicle_class_targets': vehicle_class_targets,
            'grid_recommendation': {
                'rows': int(args.grid_rows),
                'cols': int(args.grid_cols),
            },
            'vehicle_actors': vehicle_descriptions,
            'spawned_vehicle_count': len(vehicles),
            'spawned_vehicle_class_counts': {
                name: sum(
                    item['benchmark_class'] == name
                    for item in vehicle_descriptions
                )
                for name in SPAWN_CLASSES
            },
            'spawned_walker_count': len(walkers),
            'walker_sampling': {
                'min_distance_m': float(args.walker_min_distance),
                'max_distance_m': float(args.walker_max_distance),
                'half_angle_deg': float(args.walker_half_angle),
            },
            'frames': [],
        }
        _write_json(output / 'manifest.json', manifest)

        for _ in range(args.warmup_frames):
            world.tick()

        vehicle_tags = set()
        for key in ('vehicle', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'):
            vehicle_tags.update(semantic_tag_ids.get(key, ()))
        person_tags = set(semantic_tag_ids.get('person', ()))

        saved = 0
        visible_vehicle_observations = 0
        visible_people_observations = 0
        visible_vehicle_observations_by_class = {
            name: 0 for name in SPAWN_CLASSES
        }
        for capture_index in range(args.frames):
            frame = world.tick()
            rgb_data = _get_frame(
                sensor_queues['rgb'], frame, args.sensor_timeout)
            semantic_data = _get_frame(
                sensor_queues['semantic'], frame, args.sensor_timeout)
            instance_data = _get_frame(
                sensor_queues['instance'], frame, args.sensor_timeout)
            if capture_index % max(1, args.save_every) != 0:
                continue

            rgb = decode_carla_bgra(
                rgb_data.raw_data, args.width, args.height)
            semantic = decode_carla_semantic(
                semantic_data.raw_data, args.width, args.height)
            instance = decode_carla_instance(
                instance_data.raw_data, args.width, args.height)
            frame_name = f'{frame:08d}'
            Image.fromarray(rgb, mode='RGB').save(
                output / 'rgb' / f'{frame_name}.png')
            Image.fromarray(semantic, mode='L').save(
                output / 'semantic' / f'{frame_name}.png')
            Image.fromarray(instance, mode='I;16').save(
                output / 'instance' / f'{frame_name}.png')

            visible_vehicles = _visible_instances(
                vehicles,
                vehicle_descriptions,
                instance,
                semantic,
                vehicle_tags,
                args.min_instance_pixels,
            )
            visible_people = _visible_instances(
                walkers,
                [],
                instance,
                semantic,
                person_tags,
                args.min_instance_pixels,
            )
            for item in visible_vehicles:
                if 'requested_color_rgb' in item:
                    canonical, scores = canonical_color_from_rgb(
                        item['requested_color_rgb'])
                    item['canonical_color_name'] = canonical
                    item['canonical_color_scores'] = scores

            frame_metadata = {
                'frame': int(frame),
                'timestamp': float(rgb_data.timestamp),
                'rgb': f'rgb/{frame_name}.png',
                'semantic': f'semantic/{frame_name}.png',
                'instance': f'instance/{frame_name}.png',
                'visible_vehicles': visible_vehicles,
                'visible_people': visible_people,
            }
            _write_json(
                output / 'metadata' / f'{frame_name}.json',
                frame_metadata,
            )
            manifest['frames'].append(
                f'metadata/{frame_name}.json')
            saved += 1
            visible_vehicle_observations += len(visible_vehicles)
            for item in visible_vehicles:
                class_name = item.get('benchmark_class')
                if class_name in visible_vehicle_observations_by_class:
                    visible_vehicle_observations_by_class[class_name] += 1
            visible_people_observations += len(visible_people)
            if saved % 10 == 0:
                print(
                    f'Captured {saved} frames; current CARLA frame={frame}, '
                    f'vehicles={len(visible_vehicles)}, '
                    f'people={len(visible_people)}',
                    flush=True,
                )

        manifest['saved_frame_count'] = saved
        manifest['visible_vehicle_observations'] = (
            visible_vehicle_observations
        )
        manifest['visible_people_observations'] = visible_people_observations
        manifest['visible_vehicle_observations_by_class'] = (
            visible_vehicle_observations_by_class
        )
        required_classes = [
            name
            for name, target in vehicle_class_targets.items()
            if target > 0
        ]
        missing_vehicle_classes = [
            name
            for name in required_classes
            if manifest['spawned_vehicle_class_counts'].get(name, 0) == 0
            or visible_vehicle_observations_by_class.get(name, 0) == 0
        ]
        manifest['missing_vehicle_classes'] = missing_vehicle_classes
        manifest['valid_for_vehicle_benchmark'] = (
            args.vehicles == 0 or not missing_vehicle_classes
        )
        _write_json(output / 'manifest.json', manifest)
        if args.vehicles > 0 and visible_vehicle_observations == 0:
            raise RuntimeError(
                'Capture produced no visible vehicle instances. The files '
                'were retained for diagnosis but are not a valid vehicle '
                'recognition benchmark.')
        if missing_vehicle_classes:
            print(
                'WARNING: no usable observations for requested classes: '
                + ', '.join(missing_vehicle_classes),
                file=sys.stderr,
            )
        print(
            'Vehicle observations by class: '
            + ', '.join(
                f'{name}={visible_vehicle_observations_by_class[name]}'
                for name in SPAWN_CLASSES
            ),
            flush=True,
        )
        print(f'CARLA benchmark dataset saved to {output}')
        return 0
    finally:
        for controller in walker_controllers:
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
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_capture(args)
    except Exception as exc:
        print(f'CARLA capture failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
