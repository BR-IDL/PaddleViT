"""                                                                                                                                                                                                                 
This module implements DPT
Vision Transformers for Dense Prediction
<https://arxiv.org/abs/2103.13413.pdf>
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

    def init__decoder_lr_coef(self, coef):
        for param in self.head.parameters():
            param.optimize_attr['learning_rate'] = coef