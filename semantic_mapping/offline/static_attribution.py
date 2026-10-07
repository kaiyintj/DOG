"""Partial visual-geometry attribution using the actual atomic fusion trace."""

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from semantic_mapping.gazebo.indoor_benchmark import _pose_matrix
from semantic_mapping.offline.static_geometry import (
    closest_surface, load_target_references, pose_matrix,
)
from semantic_mapping.runtime.semantic_profile import open_profile


def first_ray_hit(origin, direction, triangles):
    """Return the closest positive two-sided triangle intersection in meters."""
    a = triangles[:, 0]
    ab, ac = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(direction, ac)
    determinant = np.einsum('ij,ij->i', ab, h)
    inverse = np.zeros_like(determinant)
    nonparallel = np.abs(determinant) > 1e-10
    inverse[nonparallel] = 1 / determinant[nonparallel]
    offset = origin - a
    u = inverse * np.einsum('ij,ij->i', offset, h)
    q = np.cross(offset, ab)
    v = inverse * (q @ direction)
    distance = inverse * np.einsum('ij,ij->i', ac, q)
    valid = nonparallel & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > .02)
    return float(np.min(distance[valid])) if np.any(valid) else float('inf')


def declared_projection(path):
    root = ET.parse(path).getroot()
    joints = {joint.find('child').get('link'): joint for joint in root.findall('joint')}

    def base_from_link(link):
        chain = []
        while link in joints:
            joint = joints[link]
            origin = joint.find('origin')
            xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
            rpy = origin.get('rpy', '0 0 0') if origin is not None else '0 0 0'
            chain.append(pose_matrix(np.fromstring(xyz + ' ' + rpy, sep=' ')))
            link = joint.find('parent').get('link')
        result = np.eye(4)
        for matrix in reversed(chain):
            result = result @ matrix
        return result

    return (np.linalg.inv(base_from_link('camera_color_optical_frame'))
            @ base_from_link('mid360_link'))


def attribute_run(run_directory, max_samples=4):
    report = json.loads((run_directory / 'result.json').read_text())
    trace = run_directory / 'fusion_trace'
    events = [json.loads(line) for line in (trace / 'events.jsonl').read_text().splitlines()]
    samples = [event for event in events if event.get('source_posterior_file')]
    # The same image can appear with several point clouds; use one atomic pair.
    samples = list({tuple(event['source_id']): event for event in samples}.values())
    indexes = np.linspace(0, len(samples) - 1, min(max_samples, len(samples)), dtype=int)
    alignment = report['recording']['alignment']
    world_from_map = _pose_matrix(alignment['truth'], 'world') @ np.linalg.inv(
        _pose_matrix(alignment['estimate'], 'odom'))
    references = list(load_target_references(report['case']).values())
    bounds = [(value.triangles.min(axis=(0, 1)), value.triangles.max(axis=(0, 1)))
              for value in references]
    profile = open_profile(report['case']['ontology_profile'])
    result = {'scope': 'partial_chair_table_visual_geometry_proxy',
              'endpoint_surface_tolerance_m': .10, 'ray_depth_tolerance_m': .15,
              'not_strict_pointwise_or_image_ground_truth': True,
              'missing_scene_geometry_stays_unknown': True, 'frames': []}
    for index in indexes:
        event = samples[index]
        with np.load(trace / event['file']) as stored:
            arrays = {key: stored[key] for key in stored.files}
        world_points = arrays['points'] @ world_from_map[:3, :3].T + world_from_map[:3, 3]
        world_from_camera = (world_from_map @ arrays['source_to_map']
                             @ np.linalg.inv(arrays['lidar_to_camera']))
        origin = world_from_camera[:3, 3]
        labels = np.argmax(arrays['logits'], axis=1)
        counters, flags = Counter(), np.zeros(len(labels), dtype=int)
        for point_index, (point, predicted) in enumerate(zip(world_points, labels)):
            candidates = [(reference, closest_surface(point, reference.triangles)[1])
                          for reference, (minimum, maximum) in zip(references, bounds)
                          if np.all(point >= minimum - .10) and np.all(point <= maximum + .10)]
            reference, distance = min(candidates, key=lambda item: item[1], default=(None, 1.))
            if reference is None or distance > .10:
                counters['endpoint_unknown'] += 1
                continue
            endpoint_class = profile.classes.index(reference.class_name)
            vector = point - origin
            measured_range = np.linalg.norm(vector)
            direction = vector / measured_range
            intersections = [(first_ray_hit(origin, direction, value.triangles), value)
                             for value in references]
            hit, foreground = min(intersections, key=lambda value: value[0])
            if hit < measured_range - .15:
                counters['known_mesh_in_front_of_endpoint'] += 1
                if predicted != endpoint_class and predicted == profile.classes.index(
                        foreground.class_name):
                    counters['label_matches_known_foreground'] += 1
                    flags[point_index] = 3
            elif (abs(hit - measured_range) <= .15
                  and foreground.class_name == reference.class_name):
                if predicted == endpoint_class:
                    counters['known_surface_label_agrees'] += 1
                    flags[point_index] = 1
                else:
                    counters['known_surface_label_disagrees'] += 1
                    flags[point_index] = 2
            else:
                counters['camera_ray_unknown'] += 1
        used = arrays['lidar_to_camera']
        declaration = declared_projection(run_directory / 'robot_description.urdf')
        result['frames'].append({
            'file': event['file'], 'rgb': event['source_rgb_file'], 'counts': dict(counters),
            'point_image_skew_sec': (event['observation_ns'] - event['source_id'][1]) / 1e9,
            'extrinsic_declaration_max_abs_difference': float(np.max(np.abs(used - declaration))),
            'intrinsic_camera_info_max_abs_difference': float(np.max(np.abs(
                arrays['camera_matrix'] - np.array(report['recording']['camera_info']['k'])
                .reshape(3, 3)))),
        })
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        rgb = np.asarray(Image.open(trace / event['source_rgb_file']))
        with np.load(trace / event['source_posterior_file']) as stored:
            posterior = stored['posterior']
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(rgb)
        axes[0].set_title('Atomic source RGB')
        axes[1].imshow(rgb)
        axes[1].imshow(posterior[..., profile.classes.index('chair')], alpha=.55,
                       extent=(0, rgb.shape[1], rgb.shape[0], 0), vmin=0, vmax=1, cmap='magma')
        axes[1].set_title('Chair posterior (0 to 1)')
        axes[2].imshow(rgb)
        colors = np.array(['#9e9e9e', '#32cd32', '#ffa500', '#ff2020'])
        axes[2].scatter(arrays['pixel_u'], arrays['pixel_v'], s=4, c=colors[flags])
        axes[2].set_title('Green agree / orange disagree / red foreground / gray unknown')
        for axis in axes:
            axis.set_xlim(0, rgb.shape[1])
            axis.set_ylim(rgb.shape[0], 0)
            axis.axis('off')
        figure.tight_layout()
        figure.savefig(run_directory / f'attribution_{index:04d}.png', dpi=120)
        plt.close(figure)
    (run_directory / 'attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    result = attribute_run(args.run_directory)
    print(json.dumps({'scope': result['scope'], 'frames': len(result['frames'])}))


if __name__ == '__main__':
    main()
