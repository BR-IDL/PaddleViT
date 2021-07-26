"""
DPT model implementation
"""

import paddle
import paddle.nn as nn
from .backbones.vit import VisualTransformer
from .decoders.dpt_head import DPTHead

class DPTSeg(nn.Layer):
    """DPT Segmentation model
    """
    def __init__(self, config):
        super(DPTSeg, self).__init__()
        self.backbone = VisualTransformer(config)
        self.head = DPTHead(config)

    def forward(self, inputs):
        features = self.backbone(inputs)
        out = self.head(features)
        return out