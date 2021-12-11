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

import paddle.nn as nn

from .backbones.mix_transformer import MixVisionTransformer
from .decoders.segformer_head import SegformerHead


class Segformer(nn.Layer):
    """Segformer model implementation
    
    """
    def __init__(self, config):
        super(Segformer, self).__init__()
        self.backbone = MixVisionTransformer(
            in_channels=config.MODEL.TRANS.IN_CHANNELS,
            embed_dims=config.MODEL.TRANS.EMBED_DIM,
            num_stages=config.MODEL.TRANS.NUM_STAGES,
            num_layers=config.MODEL.TRANS.NUM_LAYERS,
            num_heads=config.MODEL.TRANS.NUM_HEADS,
            patch_sizes=config.MODEL.TRANS.PATCH_SIZE,
            strides=config.MODEL.TRANS.STRIDES,
            sr_ratios=config.MODEL.TRANS.SR_RATIOS,
            out_indices=config.MODEL.ENCODER.OUT_INDICES,
            mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
            qkv_bias=config.MODEL.TRANS.QKV_BIAS,
            drop_rate=config.MODEL.DROPOUT,
            attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
            drop_path_rate=config.MODEL.DROP_PATH,
            pretrained=config.MODEL.PRETRAINED)
        self.decode_head = SegformerHead(
            in_channels=config.MODEL.SEGFORMER.IN_CHANNELS,
            channels=config.MODEL.SEGFORMER.CHANNELS,
            num_classes=config.DATA.NUM_CLASSES,
            align_corners=config.MODEL.SEGFORMER.ALIGN_CORNERS)

    def forward(self, inputs):
        features = self.backbone(inputs)
        out = self.decode_head(features)
        return out
