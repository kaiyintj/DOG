import numpy as np

from semantic_mapping.carla_benchmark import (
    actor_instance_id,
    augment_grid_truth_with_instances,
    bbox_from_mask,
    build_project_lookup,
    byte_swap_instance_ids,
    camera_intrinsics,
    classify_instance_color,
    confusion_matrix,
    decode_carla_bgra,
    decode_carla_instance,
    decode_carla_semantic,
    grid_cell_truth,
    make_coarse_masks,
    metrics_from_confusion,
)
from semantic_mapping.carla_capture_benchmark import (
    _balanced_vehicle_schedule,
    _forward_spawn_candidates,
    _spawn_vehicles,
    _spawn_walkers,
)
from semantic_mapping.semantic_schema import DEFAULT_CLASSES


def test_carla_bgra_semantic_and_instance_decoding():
    # CARLA raw order is B, G, R, A. Actor ID is G + (B << 8).
    raw = bytes([
        0x34, 0x12, 83, 255,
        0x78, 0x56, 6, 255,
    ])

    rgb = decode_carla_bgra(raw, width=2, height=1)
    semantic = decode_carla_semantic(raw, width=2, height=1)
    instance = decode_carla_instance(raw, width=2, height=1)

    assert rgb.tolist() == [[[83, 0x12, 0x34], [6, 0x56, 0x78]]]
    assert semantic.tolist() == [[83, 6]]
    assert instance.tolist() == [[0x3412, 0x7856]]


def test_actor_instance_id_and_legacy_byte_swap():
    legacy = np.asarray([[0x1900, 0x2C01]], dtype=np.uint16)

    repaired = byte_swap_instance_ids(legacy)

    assert actor_instance_id(65536 + 25) == 25
    assert repaired.tolist() == [[25, 300]]


def test_bbox_from_mask_is_tight_and_xyxy():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 3:7] = True

    assert bbox_from_mask(mask) == (3, 2, 7, 5)


def test_balanced_vehicle_schedule_covers_all_requested_classes():
    schedule = _balanced_vehicle_schedule(
        8, ('car', 'truck', 'bus'))

    assert schedule == [
        'car', 'truck', 'bus',
        'car', 'truck', 'bus',
        'car', 'truck',
    ]


def test_balanced_vehicle_schedule_covers_two_wheel_classes():
    schedule = _balanced_vehicle_schedule(
        7, ('car', 'truck', 'bus', 'bicycle', 'motorcycle'))

    assert schedule == [
        'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'car', 'truck'
    ]


def test_vehicle_spawn_uses_balanced_requested_classes():
    class Attribute:
        def __init__(self, value):
            self.value = value

        def as_int(self):
            return int(self.value)

        def __str__(self):
            return str(self.value)

    class Blueprint:
        def __init__(self, blueprint_id, base_type):
            self.id = blueprint_id
            self.attributes = {
                'base_type': Attribute(base_type),
                'color': Attribute('0,0,0'),
                'number_of_wheels': Attribute(4),
                'role_name': Attribute(''),
            }

        def has_attribute(self, name):
            return name in self.attributes

        def get_attribute(self, name):
            return self.attributes[name]

        def set_attribute(self, name, value):
            self.attributes[name] = Attribute(value)

    class Library:
        def __init__(self):
            self.blueprints = [
                Blueprint('vehicle.test.car', 'car'),
                Blueprint('vehicle.test.truck', 'truck'),
                Blueprint('vehicle.test.bus', 'bus'),
            ]

        def filter(self, pattern):
            return self.blueprints

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return Location(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

    class Transform:
        def __init__(self, x=0.0, y=0.0):
            self.location = Location(x, y)

        def get_forward_vector(self):
            return Location(x=1.0)

    class Actor:
        def __init__(self, actor_id, blueprint):
            self.id = actor_id
            self.type_id = blueprint.id

    class World:
        def __init__(self):
            self.next_id = 10

        def try_spawn_actor(self, blueprint, transform):
            actor = Actor(self.next_id, blueprint)
            self.next_id += 1
            return actor

    actors, descriptions, _, targets = _spawn_vehicles(
        World(),
        Library(),
        [Transform(float(x), 0.0) for x in range(10, 80, 10)],
        Transform(),
        count=6,
        seed=42,
        class_names=('car', 'truck', 'bus'),
    )

    assert len(actors) == 6
    assert targets == {'car': 2, 'truck': 2, 'bus': 2}
    assert [
        item['benchmark_class'] for item in descriptions
    ] == ['car', 'truck', 'bus', 'car', 'truck', 'bus']


def test_compound_ade_label_maps_to_project_car():
    lookup = build_project_lookup({
        0: 'wall',
        1: 'car, auto, automobile, machine, motorcar',
        2: 'truck',
        3: 'sky',
    })

    assert lookup.tolist() == [1, 4, 5, len(DEFAULT_CLASSES) - 1]


def test_generic_vehicle_tag_does_not_create_three_fine_classes():
    tags = np.full((4, 4), 10, dtype=np.uint8)
    truth = grid_cell_truth(
        tags,
        {
            'road': [7],
            'person': [4],
            'vehicle': [10],
            'car': [],
            'truck': [],
            'bus': [],
        },
        grid_rows=1,
        grid_cols=1,
        min_fraction=0.02,
    )

    assert truth[0]['labels'] == ['vehicle']
    assert truth[0]['fractions']['car'] == 0.0
    assert truth[0]['fractions']['truck'] == 0.0
    assert truth[0]['fractions']['bus'] == 0.0


def test_instance_metadata_adds_only_the_exact_vehicle_class():
    instance = np.zeros((4, 8), dtype=np.uint16)
    instance[:, 4:] = 42
    truth = [
        {'row': 0, 'column': 0, 'labels': [], 'fractions': {}},
        {'row': 0, 'column': 1, 'labels': [], 'fractions': {}},
    ]

    augmented = augment_grid_truth_with_instances(
        truth,
        instance,
        [{'instance_id': 42, 'benchmark_class': 'truck'}],
        grid_rows=1,
        grid_cols=2,
        min_pixels=2,
    )

    assert augmented[0]['labels'] == []
    assert augmented[1]['labels'] == ['truck', 'vehicle']


def test_instance_metadata_adds_exact_two_wheel_class():
    instance = np.full((4, 4), 77, dtype=np.uint16)
    truth = [
        {'row': 0, 'column': 0, 'labels': [], 'fractions': {}},
    ]

    augmented = augment_grid_truth_with_instances(
        truth,
        instance,
        [{'instance_id': 77, 'benchmark_class': 'bicycle'}],
        grid_rows=1,
        grid_cols=1,
        min_pixels=2,
    )

    assert augmented[0]['labels'] == ['bicycle', 'vehicle']


def test_grid_truth_preserves_two_wheel_and_rider_tags():
    tags = np.asarray([[19, 18, 13]], dtype=np.uint8)

    truth = grid_cell_truth(
        tags,
        {
            'bicycle': [19],
            'motorcycle': [18],
            'rider': [13],
        },
        grid_rows=1,
        grid_cols=3,
        min_fraction=0.5,
    )

    assert truth[0]['labels'] == ['bicycle']
    assert truth[1]['labels'] == ['motorcycle']
    assert truth[2]['labels'] == ['person']


def test_segformer_coarse_vehicle_metrics_are_computable():
    class_index = {
        name: index for index, name in enumerate(DEFAULT_CLASSES)
    }
    carla_tags = np.asarray([[7, 4], [10, 0]], dtype=np.uint8)
    project_mask = np.asarray([
        [class_index['road'], class_index['person']],
        [class_index['truck'], class_index['unknown background']],
    ], dtype=np.uint8)
    truth, prediction = make_coarse_masks(
        carla_tags,
        project_mask,
        {'road': [7], 'person': [4], 'vehicle': [10]},
    )
    matrix = confusion_matrix(truth, prediction, class_count=4)
    metrics = metrics_from_confusion(
        matrix, ('background', 'road', 'person', 'vehicle'))

    assert np.array_equal(truth, prediction)
    assert metrics['road']['iou'] == 1.0
    assert metrics['person']['iou'] == 1.0
    assert metrics['vehicle']['iou'] == 1.0


def test_two_wheel_and_rider_are_in_coarse_navigation_metrics():
    class_index = {
        name: index for index, name in enumerate(DEFAULT_CLASSES)
    }
    carla_tags = np.asarray([[19, 18], [13, 0]], dtype=np.uint8)
    project_mask = np.asarray([
        [class_index['bicycle'], class_index['motorcycle']],
        [class_index['person'], class_index['unknown background']],
    ], dtype=np.uint8)

    truth, prediction = make_coarse_masks(
        carla_tags,
        project_mask,
        {'bicycle': [19], 'motorcycle': [18], 'rider': [13]},
    )

    assert np.array_equal(truth, prediction)
    assert truth.tolist() == [[3, 3], [2, 0]]


def test_blue_vehicle_body_is_classified_as_blue():
    pixels = np.tile(
        np.asarray([[25, 70, 220]], dtype=np.uint8),
        (100, 1),
    )
    pixels[:20] = np.asarray([15, 15, 15], dtype=np.uint8)

    color, scores = classify_instance_color(pixels)

    assert color == 'blue'
    assert scores['blue'] > scores['red']


def test_camera_intrinsics_have_centered_principal_point():
    matrix = camera_intrinsics(640, 480, 90.0)

    assert np.isclose(matrix[0, 0], 320.0)
    assert np.isclose(matrix[1, 1], 320.0)
    assert np.isclose(matrix[0, 2], 320.0)
    assert np.isclose(matrix[1, 2], 240.0)


def test_walker_spawn_uses_origin_transform_location():
    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return Location(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

    class Transform:
        def __init__(self, location=None):
            self.location = location or Location()

        def get_forward_vector(self):
            return Location(x=1.0)

    class Blueprint:
        def has_attribute(self, name):
            return name == 'is_invincible'

        def set_attribute(self, name, value):
            pass

    class Library:
        def filter(self, pattern):
            return [Blueprint()]

        def find(self, blueprint_id):
            return Blueprint()

    class World:
        def get_random_location_from_navigation(self):
            return Location(x=10.0)

        def try_spawn_actor(self, blueprint, transform, attach_to=None):
            assert isinstance(transform, Transform)
            return object()

    walkers, controllers = _spawn_walkers(
        World(),
        Library(),
        Transform(Location()),
        count=1,
        seed=42,
    )

    assert len(walkers) == 1
    assert len(controllers) == 1


def test_forward_spawn_candidates_reject_side_and_rear_points():
    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return Location(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
            )

    class Transform:
        def __init__(self, x, y):
            self.location = Location(x, y)

        def get_forward_vector(self):
            return Location(x=1.0)

    origin = Transform(0.0, 0.0)
    ahead = Transform(20.0, 3.0)
    side = Transform(3.0, 20.0)
    rear = Transform(-20.0, 0.0)

    candidates = _forward_spawn_candidates(
        [ahead, side, rear],
        origin,
    )

    assert candidates == [ahead]
