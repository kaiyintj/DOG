"""Record actual fusion inputs and pruning events for matched offline comparisons."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class FusionTraceWriter:
    """An explicitly enabled experiment trace; it does not alter map inputs."""

    def __init__(self, directory, parameters):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'parameters.json').write_text(
            json.dumps(parameters, indent=2, ensure_ascii=False) + '\n')
        self.index = 0
        self.sources = {}

    def _event(self, value):
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

    def observation(self, frame, points, observation_ns, counts, processed_sim_sec):
        arrays = {'points': points, 'reliability': frame['reliability'],
                  'logits': frame['logits']}
        for name in ('features', 'colors'):
            if frame[name] is not None:
                arrays[name] = frame[name]
        diagnostics = frame.get('diagnostics')
        image_file = posterior_file = None
        if diagnostics is not None:
            for name in ('pixel_u', 'pixel_v', 'camera_depth', 'camera_matrix',
                         'lidar_to_camera', 'source_to_map'):
                arrays[name] = diagnostics[name]
            arrays['sensor_points'] = frame['points']
            source = tuple(frame['source_id'])
            if source not in self.sources:
                source_index = len(self.sources)
                self.sources[source] = (None, None)
                # A few atomic RGB/posterior samples suffice for attribution;
                # all projected points and pixel associations are retained.
                if source_index % 5 == 0:
                    image_file = f'source_{source_index:06d}.png'
                    Image.fromarray(diagnostics['rgb']).save(
                        self.directory / image_file, compress_level=1)
                    if diagnostics['posterior'] is not None:
                        posterior_file = f'source_{source_index:06d}_posterior.npz'
                        np.savez_compressed(self.directory / posterior_file,
                                            posterior=diagnostics['posterior'].astype(np.float16))
                    self.sources[source] = (image_file, posterior_file)
            image_file, posterior_file = self.sources[source]
        name = f'frame_{self.index:06d}.npz'
        np.savez_compressed(self.directory / name, **arrays)
        self._event({'event': 'observation', 'file': name,
                     'observation_ns': observation_ns, 'source_id': frame['source_id'],
                     'processed_sim_sec': processed_sim_sec,
                     'source_rgb_file': image_file, 'source_posterior_file': posterior_file,
                     'online_new_updated': list(counts)})
        self.index += 1

    def prune(self, now_sec, center):
        self._event({'event': 'prune', 'now_sec': now_sec, 'processed_sim_sec': now_sec,
                     'center': None if center is None else np.asarray(center).tolist()})
