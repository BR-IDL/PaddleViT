#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
