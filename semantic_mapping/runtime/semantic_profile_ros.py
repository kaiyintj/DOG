"""Adapt ROS startup configuration to one semantic profile contract."""

from dataclasses import dataclass
import json

from semantic_mapping.runtime.semantic_profile import SemanticProfile, open_profile


@dataclass(frozen=True)
class SemanticContract:
    """Validated startup selection consumed by existing ROS nodes."""

    profile: SemanticProfile
    semantic_costs: tuple


@dataclass(frozen=True)
class SemanticCapability:
    """The actual frontend checkpoint and its bound project meaning."""

    checkpoint_id: str
    profile: SemanticProfile


def load_semantic_contract(parameters):
    """Select a profile and reject conflicting legacy class parameters."""
    profile = open_profile(parameters.get('ontology_profile', 'outdoor13'))
    if parameters.get('num_classes', profile.K) != profile.K:
        raise ValueError(
            f'num_classes must be {profile.K} for {profile.id}')
    if tuple(parameters.get('vocab', profile.classes)) != profile.classes:
        raise ValueError(f'vocab must match the ordered classes of {profile.id}')
    costs = tuple(parameters.get('semantic_costs', profile.semantic_costs))
    if len(costs) != profile.K:
        raise ValueError('semantic_costs length must match num_classes')
    for index, cost in enumerate(costs):
        if index == profile.unknown_id:
            valid = cost == -1
        elif index in profile.traversable_ids:
            valid = 0 <= cost < 100
        else:
            valid = cost == 100
        if not valid:
            raise ValueError(
                f'semantic_costs contradicts navigation role: '
                f'{profile.classes[index]}={cost}')
    return SemanticContract(profile, costs)


def make_semantic_capability(profile, checkpoint_id, checkpoint_labels):
    """Describe the actual loaded frontend without inventing class support."""
    from std_msgs.msg import String

    return String(data=json.dumps({
        'profile': profile.id,
        'classes': profile.classes,
        'checkpoint': str(checkpoint_id),
        'id2label': {
            str(index): str(label)
            for index, label in checkpoint_labels.items()
        },
    }, sort_keys=True))


def read_semantic_capability(message, expected_profile):
    """Bind actual labels only after matching the receiver's class order."""
    try:
        payload = json.loads(message.data)
        if payload['profile'] != expected_profile.id:
            raise ValueError('SegFormer capability profile does not match GA')
        if tuple(payload['classes']) != expected_profile.classes:
            raise ValueError('SegFormer capability class order does not match GA')
        profile = open_profile(expected_profile.id, payload['id2label'])
        return SemanticCapability(str(payload['checkpoint']), profile)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid SegFormer capability: {exc}') from exc


def semantic_capability_qos():
    """Keep startup capability available to a later-starting receiver."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
