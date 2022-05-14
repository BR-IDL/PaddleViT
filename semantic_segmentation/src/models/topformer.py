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
Implement TopFormer
"""

import paddle.nn as nn
from src.models.decoders import *
from src.models.backbones import *
import paddle
import logging
import warnings
warnings.filterwarnings("ignore") 


class TopFormer(nn.Layer):
    """TopFormer Segmentation model
    """
    def __init__(self, config):
        super(TopFormer, self).__init__()
        if config.MODEL.ENCODER.TYPE == "SwinTransformer":
            self.encoder = SwinTransformer(config)
        elif config.MODEL.ENCODER.TYPE == "CSwinTransformer":
            self.encoder = CSwinTransformer(config)
        elif config.MODEL.ENCODER.TYPE == "FocalTransformer":
            self.encoder = FocalTransformer(config)
        elif config.MODEL.ENCODER.TYPE == "TopTransformer":
            self.encoder = TopTransformer(config)
        
        if config.MODEL.PRETRAINED is not None:
            logging.info('Load pretrained backbone from local path!')
            self.encoder.set_state_dict(paddle.load(config.MODEL.PRETRAINED))

        if 'MaskTransformer' in config.MODEL.DECODER_TYPE:
            self.decoder = MaskTransformer(config)
        elif 'Linear' in config.MODEL.DECODER_TYPE:
            self.decoder = LinearDecoder(config)
        elif 'SimpleHead' in config.MODEL.DECODER_TYPE:
            self.decoder = SimpleHead(config)

    def forward(self, inputs):
        features = self.encoder(inputs)
        out = self.decoder(features, inputs.shape)
        return (out,)
