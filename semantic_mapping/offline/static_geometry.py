"""Explicit visual-mesh references for the fixed Small House benchmark targets."""

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    values = np.asarray(pose, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def _node_matrix(node):
    matrix = np.eye(4)
    for child in node:
        kind = child.tag.rsplit('}', 1)[-1]
        if kind not in ('matrix', 'translate', 'rotate', 'scale'):
            continue
        values = np.fromstring(child.text, sep=' ')
        operation = np.eye(4)
        if kind == 'matrix':
            operation = values.reshape(4, 4)
        elif kind == 'translate':
            operation[:3, 3] = values
        elif kind == 'scale':
            operation[:3, :3] = np.diag(values)
        else:
            axis = values[:3] / np.linalg.norm(values[:3])
            operation[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(values[3])).as_matrix()
        matrix = matrix @ operation
    return matrix


def load_dae_triangles(path):
    """Apply mesh indices, scene-node transforms, up-axis and declared meter units."""
    root = ET.parse(path).getroot()
    namespace = {'d': root.tag.split('}')[0][1:]}
    unit = root.find('d:asset/d:unit', namespace)
    meter = 1.0 if unit is None else float(unit.get('meter', '1'))
    up = root.findtext('d:asset/d:up_axis', 'Y_UP', namespace)
    axis_rotation = {'Z_UP': np.eye(3),
                     'Y_UP': Rotation.from_euler('x', 90, degrees=True).as_matrix(),
                     'X_UP': Rotation.from_euler('y', -90, degrees=True).as_matrix()}[up]
    geometries = {}
    for geometry in root.findall('d:library_geometries/d:geometry', namespace):
        mesh = geometry.find('d:mesh', namespace)
        sources = {}
        for source in mesh.findall('d:source', namespace):
            array = source.find('d:float_array', namespace)
            accessor = source.find('d:technique_common/d:accessor', namespace)
            if array is not None:
                sources[source.get('id')] = np.fromstring(array.text, sep=' ').reshape(
                    -1, int(accessor.get('stride', '1')))
        vertices = {node.get('id'): node.find('d:input[@semantic="POSITION"]', namespace)
                    .get('source').lstrip('#') for node in mesh.findall('d:vertices', namespace)}
        triangles = []
        for primitive in mesh:
            kind = primitive.tag.rsplit('}', 1)[-1]
            if kind not in ('triangles', 'polylist'):
                continue
            inputs = primitive.findall('d:input', namespace)
            vertex = next(value for value in inputs if value.get('semantic') == 'VERTEX')
            stride = max(int(value.get('offset', '0')) for value in inputs) + 1
            indexes = np.fromstring(primitive.findtext('d:p', '', namespace), sep=' ', dtype=int)
            indexes = indexes.reshape(-1, stride)[:, int(vertex.get('offset', '0'))]
            positions = sources[vertices[vertex.get('source').lstrip('#')]][:, :3]
            if kind == 'triangles':
                triangles.extend(positions[indexes.reshape(-1, 3)])
            else:
                counts = np.fromstring(primitive.findtext('d:vcount', '', namespace),
                                       sep=' ', dtype=int)
                offset = 0
                for count in counts:
                    face = indexes[offset:offset + count]
                    for index in range(1, count - 1):
                        triangles.append(positions[face[[0, index, index + 1]]])
                    offset += count
        geometries[geometry.get('id')] = np.asarray(triangles)
    scene_id = root.find('d:scene/d:instance_visual_scene', namespace).get('url').lstrip('#')
    scene = root.find(f'd:library_visual_scenes/d:visual_scene[@id="{scene_id}"]', namespace)
    transformed = []

    def visit(node, parent):
        matrix = parent @ _node_matrix(node)
        for instance in node.findall('d:instance_geometry', namespace):
            triangles = geometries[instance.get('url').lstrip('#')]
            points = triangles.reshape(-1, 3)
            points = points @ matrix[:3, :3].T + matrix[:3, 3]
            transformed.append((points * meter @ axis_rotation.T).reshape(-1, 3, 3))
        for child in node.findall('d:node', namespace):
            visit(child, matrix)

    for node in scene.findall('d:node', namespace):
        visit(node, np.eye(4))
    return np.concatenate(transformed)


def closest_surface(point, triangles):
    """Exact closest point on triangles, including edges and degenerate faces."""
    point = np.asarray(point, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length = np.einsum('ij,ij->i', edge, edge)
        fraction = np.einsum('ij,ij->i', point - start, edge) / np.maximum(length, 1e-30)
        candidates.append(start + np.clip(fraction, 0, 1)[:, None] * edge)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm_sq = np.einsum('ij,ij->i', normal, normal)
    projected = point - normal * (
        np.einsum('ij,ij->i', point - a, normal) / np.maximum(norm_sq, 1e-30))[:, None]
    offset = projected - a
    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ij,ij->i', offset, ab)
    d21 = np.einsum('ij,ij->i', offset, ac)
    denominator = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / np.maximum(denominator, 1e-30)
    v = (d00 * d21 - d01 * d20) / np.maximum(denominator, 1e-30)
    inside = (norm_sq > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1)
    projected[~inside] = np.nan
    candidates.append(projected)
    points = np.stack(candidates)
    distances = np.linalg.norm(points - point, axis=-1)
    index = np.unravel_index(np.nanargmin(distances), distances.shape)
    return points[index], float(distances[index])


@dataclass(frozen=True)
class GeometryReference:
    instance_id: str
    class_name: str
    origin: np.ndarray
    triangles: np.ndarray

    def measure(self, predicted):
        vertices = self.triangles.reshape(-1, 3)
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        center = (minimum + maximum) / 2
        nearest, distance = closest_surface(predicted, self.triangles)
        return {'reference_instance_id': self.instance_id,
                'model_origin_xyz': self.origin.tolist(),
                'visual_aabb_min_xyz': minimum.tolist(),
                'visual_aabb_max_xyz': maximum.tolist(),
                'visual_aabb_center_xyz': center.tolist(),
                'origin_distance_xy_m': float(np.linalg.norm(predicted[:2] - self.origin[:2])),
                'visual_center_distance_xy_m': float(np.linalg.norm(predicted[:2] - center[:2])),
                'visual_surface_distance_3d_m': distance,
                'nearest_visual_surface_xyz': nearest.tolist()}


def load_target_references(case):
    models = Path(case['world_path']).parent.parent / 'models'
    references = {}
    for target in case['targets']:
        asset = models / target['asset'].split('://', 1)[1]
        sdf = ET.parse(asset / 'model.sdf').getroot().find('model')
        parts = []
        for link in sdf.findall('link'):
            link_pose = np.fromstring(link.findtext('pose', '0 0 0 0 0 0'), sep=' ')
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError(
                        f'Benchmark target requires an explicit visual mesh: {target["id"]}')
                uri = mesh.findtext('uri').split('://', 1)[1].split('/', 1)[1]
                triangles = load_dae_triangles(asset / uri)
                scale = np.fromstring(mesh.findtext('scale', '1 1 1'), sep=' ')
                visual_pose = np.fromstring(visual.findtext('pose', '0 0 0 0 0 0'), sep=' ')
                matrix = (pose_matrix(target['pose']) @ pose_matrix(link_pose)
                          @ pose_matrix(visual_pose))
                points = triangles.reshape(-1, 3) * scale
                parts.append((points @ matrix[:3, :3].T + matrix[:3, 3]).reshape(-1, 3, 3))
        references[target['id']] = GeometryReference(
            target['id'], target['class'], np.asarray(target['pose'][:3]), np.concatenate(parts))
    return references


def evaluate_geometry(run_directory):
    """Keep the historical model ID fixed; add geometry metrics without replacing origins."""
    report = json.loads((run_directory / 'result.json').read_text())
    references = load_target_references(report['case'])
    instance = report['metrics']['nearest_gt_id']
    target = np.asarray(report['metrics']['target_world_xyz'])
    result = references[instance].measure(target)
    result.update(reference_selection='historical_report_nearest_origin_model_id',
                  scope='reference_geometry_only_not_pointwise_semantic_gt')
    (run_directory / 'geometry_reference.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    print(json.dumps(evaluate_geometry(parser.parse_args().run_directory), indent=2))
