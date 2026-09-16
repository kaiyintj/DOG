"""Semantic meaning shared by segmentation, queries and navigation."""

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Optional


class NavigationRole(str, Enum):
    """Navigation behavior, independent of whether a class is queryable."""

    TRAVERSABLE = 'traversable'
    STRUCTURAL_OBSTACLE = 'structural_obstacle'
    DYNAMIC_OBSTACLE = 'dynamic_obstacle'
    OBJECT_GOAL = 'object_goal'
    TRANSITION = 'transition'
    UNKNOWN = 'unknown'


class QueryStatus(str, Enum):
    """Admission result, distinct from later target evidence selection."""

    ACCEPTED = 'accepted'
    UNSUPPORTED = 'unsupported'
    NONQUERYABLE = 'nonqueryable'
    UNRESOLVED = 'unresolved'
    NOT_READY = 'not_ready'


@dataclass(frozen=True)
class QueryDecision:
    """Parsed query and the reason it may or may not seek a target."""

    status: QueryStatus
    class_id: Optional[int]
    class_name: Optional[str]
    color: Optional[str]

    @property
    def accepted(self):
        """Return whether target selection is permitted for this query."""
        return self.status == QueryStatus.ACCEPTED


@dataclass(frozen=True)
class ClassSpec:
    """Meaning of one class in an ordered project ontology."""

    name: str
    navigation_role: NavigationRole
    queryable: bool
    color: tuple
    semantic_cost: int
    max_extent_m: float
    raw_aliases: tuple = ()
    query_aliases: tuple = ()


@dataclass(frozen=True)
class SemanticProfile:
    """Immutable class order and navigation roles for one map lifetime."""

    id: str
    class_specs: tuple
    raw_lookup: Optional[tuple] = None

    @property
    def classes(self):
        """Return project classes in posterior channel order."""
        return tuple(spec.name for spec in self.class_specs)

    @property
    def K(self):
        """Return the full posterior dimension, including unknown."""
        return len(self.class_specs)

    @property
    def unknown_id(self):
        """Return the channel reserved for unmatched checkpoint labels."""
        return self.K - 1

    @property
    def colors(self):
        """Return one RGB color per posterior channel."""
        return tuple(spec.color for spec in self.class_specs)

    @property
    def semantic_costs(self):
        """Return costs while keeping unknown separate from free space."""
        return tuple(spec.semantic_cost for spec in self.class_specs)

    @property
    def query_max_extents(self):
        """Return default target-cluster extent limits in metres."""
        return tuple(spec.max_extent_m for spec in self.class_specs)

    def _role_ids(self, role):
        return frozenset(
            index for index, spec in enumerate(self.class_specs)
            if spec.navigation_role == role)

    @property
    def traversable_ids(self):
        """Return channels which may support navigation approach points."""
        return self._role_ids(NavigationRole.TRAVERSABLE)

    @property
    def dynamic_ids(self):
        """Return channels subject to the existing dynamic voxel lifetime."""
        return self._role_ids(NavigationRole.DYNAMIC_OBSTACLE)

    @property
    def queryable_ids(self):
        """Return classes allowed as goals, before checkpoint capability."""
        return frozenset(
            index for index, spec in enumerate(self.class_specs)
            if spec.queryable)

    @property
    def supported_ids(self):
        """Return channels backed by actual checkpoint output labels."""
        return frozenset(self.raw_lookup or ()) - {self.unknown_id}

    def resolve_query(self, text):
        """Admit a supported closed-set query or return a normal refusal."""
        class_id, color = parse_semantic_query(
            text, self.classes,
            {spec.name: spec.query_aliases for spec in self.class_specs},
        )
        class_name = None if class_id is None else self.classes[class_id]
        if class_id is None:
            status = QueryStatus.UNRESOLVED
        elif class_id not in self.queryable_ids:
            status = QueryStatus.NONQUERYABLE
        elif self.raw_lookup is None:
            status = QueryStatus.NOT_READY
        elif class_id not in self.supported_ids:
            status = QueryStatus.UNSUPPORTED
        else:
            status = QueryStatus.ACCEPTED
        return QueryDecision(status, class_id, class_name, color)

    def project_probabilities(self, raw_probabilities):
        """Sum the full checkpoint posterior before any hard decision."""
        if self.raw_lookup is None:
            raise ValueError('Checkpoint labels are required for projection')
        lookup = raw_probabilities.new_tensor(self.raw_lookup).long()
        return aggregate_project_probability_tensor(
            raw_probabilities, lookup, self.K)


def _bind_checkpoint(profile, checkpoint_labels):
    if checkpoint_labels is None:
        return profile
    if not checkpoint_labels:
        raise ValueError('Checkpoint id2label must not be empty')
    lookup = [profile.unknown_id] * (
        max(int(index) for index in checkpoint_labels) + 1)
    for raw_id, label in checkpoint_labels.items():
        if int(raw_id) < 0:
            raise ValueError('Checkpoint class IDs must be non-negative')
        terms = split_model_label(label)
        for project_id, spec in enumerate(profile.class_specs):
            aliases = {normalize_label(alias) for alias in spec.raw_aliases}
            aliases.add(normalize_label(spec.name))
            if terms.intersection(aliases):
                lookup[int(raw_id)] = project_id
                break
    return replace(profile, raw_lookup=tuple(lookup))


# Ordered class records are the single source of ontology metadata.
_PROFILE_SPECS = {
    'outdoor13': (
        ClassSpec(
            "road", NavigationRole.TRAVERSABLE, True,
            (120, 120, 120), 0, 100.0,
            ("path", "road", "sidewalk"),
            (
                "path",
                "road",
                "sidewalk",
                "street",
                "路面",
                "道路",
            )),
        ClassSpec(
            "building", NavigationRole.STRUCTURAL_OBSTACLE, True,
            (210, 210, 210), 100, 100.0,
            (
                "building",
                "door",
                "fence",
                "house",
                "wall",
                "windowpane",
            ),
            (
                "building",
                "house",
                "wall",
                "墙",
                "建筑",
                "房屋",
            )),
        ClassSpec(
            "tree", NavigationRole.STRUCTURAL_OBSTACLE, True,
            (0, 160, 0), 100, 100.0,
            (
                "flower",
                "grass",
                "plant",
                "tree",
                "vegetation",
            ),
            (
                "plant",
                "tree",
                "vegetation",
                "树",
                "树木",
                "植物",
            )),
        ClassSpec(
            "person", NavigationRole.DYNAMIC_OBSTACLE, True,
            (255, 0, 0), 100, 1.5,
            ("person", "rider"),
            (
                "human",
                "pedestrian",
                "people",
                "person",
                "人",
                "行人",
            )),
        ClassSpec(
            "car", NavigationRole.DYNAMIC_OBSTACLE, True,
            (0, 90, 255), 100, 5.0,
            (
                "auto",
                "automobile",
                "car",
                "motorcar",
                "van",
            ),
            (
                "automobile",
                "car",
                "cars",
                "van",
                "vehicle",
                "汽车",
                "车辆",
                "轿车",
            )),
        ClassSpec(
            "truck", NavigationRole.DYNAMIC_OBSTACLE, True,
            (255, 140, 0), 100, 9.0,
            ("truck",),
            (
                "lorry",
                "truck",
                "trucks",
                "卡车",
                "货车",
            )),
        ClassSpec(
            "bus", NavigationRole.DYNAMIC_OBSTACLE, True,
            (150, 70, 200), 100, 14.0,
            ("bus",),
            (
                "bus",
                "buses",
                "coach",
                "公交车",
                "巴士",
            )),
        ClassSpec(
            "bicycle", NavigationRole.DYNAMIC_OBSTACLE, True,
            (0, 220, 220), 100, 3.0,
            ("bicycle",),
            (
                "bicycle",
                "bicycles",
                "bike",
                "bikes",
                "cycle",
                "单车",
                "自行车",
            )),
        ClassSpec(
            "electric_bicycle", NavigationRole.DYNAMIC_OBSTACLE, True,
            (70, 170, 255), 100, 3.5,
            (
                "e-bike",
                "ebike",
                "electric bicycle",
                "electric bike",
                "electric_bicycle",
            ),
            (
                "e-bike",
                "e-bikes",
                "ebike",
                "ebikes",
                "electric bicycle",
                "electric bicycles",
                "electric bike",
                "electric bikes",
                "电动单车",
                "电动自行车",
                "电动车",
                "电助力单车",
                "电助力自行车",
            )),
        ClassSpec(
            "motorcycle", NavigationRole.DYNAMIC_OBSTACLE, True,
            (255, 0, 180), 100, 3.5,
            ("minibike", "motorbike", "motorcycle"),
            (
                "electric motorcycle",
                "minibike",
                "motorbike",
                "motorcycle",
                "motorcycles",
                "scooter",
                "摩托车",
                "电动摩托车",
                "电摩",
                "踏板车",
            )),
        ClassSpec(
            "chair", NavigationRole.OBJECT_GOAL, True,
            (160, 100, 40), 100, 2.0,
            (
                "armchair",
                "chair",
                "seat",
                "swivel chair",
            ),
            (
                "armchair",
                "chair",
                "chairs",
                "seat",
                "座椅",
                "椅子",
            )),
        ClassSpec(
            "bench", NavigationRole.OBJECT_GOAL, True,
            (255, 220, 0), 100, 4.0,
            ("bench",),
            (
                "bench",
                "benches",
                "长凳",
                "长椅",
            )),
        ClassSpec(
            "unknown background", NavigationRole.UNKNOWN, False,
            (50, 50, 50), -1, 100.0,
            (),
            ("unknown background",)),
    ),
    'indoor7': (
        ClassSpec(
            "floor", NavigationRole.TRAVERSABLE, False,
            (120, 120, 120), 0, 100.0,
            ("floor",),
            ("floor", "地板", "地面")),
        ClassSpec(
            "wall", NavigationRole.STRUCTURAL_OBSTACLE, False,
            (210, 210, 210), 100, 100.0,
            ("wall",),
            (
                "wall",
                "walls",
                "墙",
                "墙壁",
            )),
        ClassSpec(
            "door", NavigationRole.TRANSITION, False,
            (140, 90, 180), 100, 3.0,
            ("door", "screen door"),
            ("door", "doors", "门")),
        ClassSpec(
            "chair", NavigationRole.OBJECT_GOAL, True,
            (160, 100, 40), 100, 2.0,
            ("chair", "armchair", "swivel chair"),
            (
                "armchair",
                "chair",
                "chairs",
                "seat",
                "座椅",
                "椅子",
            )),
        ClassSpec(
            "table", NavigationRole.OBJECT_GOAL, True,
            (255, 170, 60), 100, 4.0,
            ("table", "coffee table"),
            (
                "table",
                "tables",
                "coffee table",
                "桌子",
                "餐桌",
            )),
        ClassSpec(
            "shelf", NavigationRole.STRUCTURAL_OBSTACLE, True,
            (50, 170, 220), 100, 10.0,
            ("shelf", "bookcase"),
            (
                "shelf",
                "shelves",
                "bookshelf",
                "bookshelves",
                "book shelf",
                "bookcase",
                "storage rack",
                "书架",
                "货架",
            )),
        ClassSpec(
            "bed", NavigationRole.OBJECT_GOAL, True,
            (200, 80, 150), 100, 4.0,
            ("bed",),
            (
                "bed",
                "beds",
                "床",
                "病床",
            )),
        ClassSpec(
            "unknown background", NavigationRole.UNKNOWN, False,
            (50, 50, 50), -1, 100.0,
            (),
            ("unknown background",)),
    ),
}


def open_profile(profile_id, checkpoint_labels=None):
    """Open a fixed ontology without loading a model or touching ROS."""
    if profile_id not in _PROFILE_SPECS:
        raise ValueError(f'Unknown ontology_profile: {profile_id!r}')
    return _bind_checkpoint(
        SemanticProfile(profile_id, _PROFILE_SPECS[profile_id]),
        checkpoint_labels,
    )


# Real legacy callers retain these exports, derived from the same records.
_OUTDOOR = open_profile('outdoor13')
DEFAULT_CLASSES = _OUTDOOR.classes
DEFAULT_CLASS_COLORS = _OUTDOOR.colors
DEFAULT_SEMANTIC_COSTS = _OUTDOOR.semantic_costs
DEFAULT_QUERY_MAX_EXTENTS_M = _OUTDOOR.query_max_extents
SEGFORMER_LABEL_ALIASES = {
    index: set(spec.raw_aliases)
    for index, spec in enumerate(_OUTDOOR.class_specs)
    if spec.navigation_role != NavigationRole.UNKNOWN
}
QUERY_CLASS_ALIASES = {
    spec.name: set(spec.query_aliases) for spec in _OUTDOOR.class_specs
}

COLOR_ALIASES = {
    'red': {'red', '红', '红色'},
    'orange': {'orange', '橙', '橙色'},
    'yellow': {'yellow', '黄', '黄色'},
    'green': {'green', '绿', '绿色'},
    'blue': {'blue', '蓝', '蓝色'},
    'purple': {'purple', 'violet', '紫', '紫色'},
    'brown': {'brown', '棕', '棕色', '褐色'},
    'black': {'black', '黑', '黑色'},
    'white': {'white', '白', '白色'},
    'gray': {'gray', 'grey', '灰', '灰色'},
}


def normalize_label(label):
    """Normalize model labels and text queries for matching."""
    return str(label).lower().replace('-', ' ').replace('_', ' ').strip()


def _alias_matches(alias, normalized_text, tokens):
    normalized_alias = normalize_label(alias)
    if not normalized_alias:
        return False
    if re.fullmatch(r'[a-z0-9 ]+', normalized_alias):
        if ' ' in normalized_alias:
            return normalized_alias in normalized_text
        return normalized_alias in tokens
    return normalized_alias in normalized_text


def parse_semantic_query(
    query, vocab=DEFAULT_CLASSES, class_aliases=QUERY_CLASS_ALIASES,
):
    """Return ``(class_index, color_name)`` parsed from a text query."""
    normalized = normalize_label(query)
    tokens = set(re.findall(r'[a-z0-9]+', normalized))

    color_name = None
    for canonical_color, aliases in COLOR_ALIASES.items():
        if any(_alias_matches(alias, normalized, tokens) for alias in aliases):
            color_name = canonical_color
            break

    class_index = None
    best_alias_score = None
    vocab_lower = [normalize_label(name) for name in vocab]
    for index, class_name in enumerate(vocab_lower):
        raw_class_name = str(vocab[index]).lower().strip()
        aliases = set(class_aliases.get(raw_class_name, ()))
        aliases.update(class_aliases.get(class_name, ()))
        aliases.add(class_name)
        for alias in aliases:
            if not _alias_matches(alias, normalized, tokens):
                continue
            normalized_alias = normalize_label(alias)
            # Prefer the most specific phrase across classes. Otherwise the
            # shorter "bicycle" alias consumes "electric bicycle" first.
            score = (
                len(normalized_alias),
                len(normalized_alias.split()),
                -index,
            )
            if best_alias_score is None or score > best_alias_score:
                best_alias_score = score
                class_index = index

    return class_index, color_name


def split_model_label(label):
    """Return a model label and each comma/slash-separated synonym."""
    terms = {normalize_label(label)}
    terms.update(
        normalize_label(part)
        for part in re.split(r'[,;/|]+', str(label))
    )
    return {term for term in terms if term}


def aggregate_project_probability_tensor(
    raw_probabilities,
    project_lookup,
    num_project_classes,
):
    """Sum a BxCxHxW tensor into project classes without hard decisions."""
    if raw_probabilities.ndim != 4:
        raise ValueError('raw probability tensor must have shape BxCxHxW')
    if project_lookup.ndim != 1:
        raise ValueError('project lookup tensor must be one-dimensional')
    if raw_probabilities.shape[1] != project_lookup.shape[0]:
        raise ValueError('project lookup must cover every raw class')
    if int(num_project_classes) <= 0:
        raise ValueError('num_project_classes must be positive')
    projected = raw_probabilities.new_zeros(
        (
            raw_probabilities.shape[0],
            int(num_project_classes),
            raw_probabilities.shape[2],
            raw_probabilities.shape[3],
        )
    )
    projected.index_add_(1, project_lookup, raw_probabilities)
    return projected / projected.sum(dim=1, keepdim=True).clamp_min(1e-12)
