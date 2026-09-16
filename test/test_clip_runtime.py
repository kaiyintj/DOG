import threading

import numpy as np
import torch
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from semantic_mapping.runtime.clip_node import ClipNode
from semantic_mapping.runtime.semantic_posterior import float32_image_to_array


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeBridge:
    def compressed_imgmsg_to_cv2(self, message, encoding):
        return np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3)

    def cv2_to_imgmsg(self, image, encoding):
        message = Image()
        message.height = image.shape[0]
        message.width = image.shape[1]
        message.encoding = encoding
        return message


class FakeModel:
    def encode_image(self, batch):
        rows = torch.arange(
            1,
            batch.shape[0] * 4 + 1,
            dtype=torch.float32,
        )
        return rows.reshape(batch.shape[0], 4)


def test_clip_callback_publishes_one_header_bound_logits_feature_frame():
    node = object.__new__(ClipNode)
    node.interval = 0.0
    node.last_inference_ns = None
    node.get_clock = lambda: type(
        'Clock', (), {'now': lambda self: type(
            'Now', (), {'nanoseconds': 5_000_000_000})()},
    )()
    node.image_is_compressed = True
    node.image_encoding = 'rgb8'
    node.bridge = FakeBridge()
    node.grid_rows = 2
    node.grid_cols = 3
    node.K = 2
    node.device = 'cpu'
    node.preprocess = lambda image: torch.ones((3, 2, 2))
    node.model = FakeModel()
    node.model_lock = threading.Lock()
    node.text_feats = torch.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    node.logit_scale = 1.0
    node.pub_clip_frame = CapturingPublisher()
    node.pub_source_image = CapturingPublisher()
    warnings = []
    node.get_logger = lambda: type(
        'Logger', (), {'warn': lambda self, message: warnings.append(message)})()
    message = CompressedImage()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 34
    message.header.frame_id = 'camera'

    node.callback(message)

    assert warnings == []
    assert len(node.pub_clip_frame.messages) == 1
    assert len(node.pub_source_image.messages) == 1
    frame_message = node.pub_clip_frame.messages[0]
    frame = float32_image_to_array(frame_message, expected_channels=6)
    assert frame.shape == (2, 3, 6)
    assert frame_message.header == message.header
    assert node.pub_source_image.messages[0].header == message.header


def test_clip_query_feature_carries_its_source_text_identity():
    node = object.__new__(ClipNode)
    node.vocab = ['road', 'car']
    node.text_feats = torch.asarray([[1.0, 0.0], [0.0, 1.0]])
    node.pub_query_feat = CapturingPublisher()
    node.get_logger = lambda: type(
        'Logger', (), {'info': lambda self, message: None})()

    node.query_callback(String(data='car'))

    message = node.pub_query_feat.messages[0]
    assert message.layout.dim[0].label == 'query:car'
    assert list(message.data) == [0.0, 1.0]
