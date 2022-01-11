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

import copy
import paddle
import paddle.nn as nn
from src.models.backbones.vit import EncoderLayer

class MaskTransformer(nn.Layer):
    """
    Segmenter decoder use transformer as decoder for segmentation,
    performs better than the linear layer.
    the decoder has the same embedding dimensions as the encoder
    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        mask_tokens: several tokens added for segmentation, each for a certain class.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.MODEL.TRANS.HIDDEN_SIZE
        self.feature_size = (config.DATA.CROP_SIZE[0] // config.MODEL.TRANS.PATCH_SIZE,
                            config.DATA.CROP_SIZE[1] // config.MODEL.TRANS.PATCH_SIZE)
        self.cls_num = config.DATA.NUM_CLASSES

        self.layers = nn.LayerList([
            copy.deepcopy(EncoderLayer(config)) for _ in range(config.MODEL.SEGMENTER.NUM_LAYERS)])

        self.mask_tokens = self.create_parameter(shape=(1, self.cls_num, hidden_size))

        self.proj_decoder = nn.Linear(hidden_size, hidden_size)

        weight_attr_patch = paddle.ParamAttr(
            initializer=nn.initializer.Normal(std=hidden_size ** -0.5)
        )
        self.proj_patch = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=weight_attr_patch,
            bias_attr=False
        )
        weight_attr_class = paddle.ParamAttr(
            initializer=nn.initializer.Normal(std=hidden_size ** -0.5)
        )
        self.proj_class = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=weight_attr_class,
            bias_attr=False
        )

        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.mask_norm = nn.LayerNorm(self.cls_num)

    def forward(self, x):
        H, W = self.feature_size
        x = self.proj_decoder(x)
        mask_tokens = self.mask_tokens.expand((x.shape[0], -1, -1))
        x = paddle.concat([x, mask_tokens], axis=1)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.decoder_norm(x)
        patches, masks = x[:, :-self.cls_num], x[:, -self.cls_num:]
        patches = self.proj_patch(patches)
        masks = self.proj_class(masks)
        patches = patches / paddle.norm(patches, axis=-1, keepdim=True)
        masks = masks / paddle.norm(masks, axis=-1, keepdim=True)
        masks = patches @ masks.transpose((0, 2, 1))
        masks = self.mask_norm(masks)
        #[b, (h w), n] -> [b, n, h, w]
        masks = masks.reshape((masks.shape[0], H, W, masks.shape[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return masks

class LinearDecoder(nn.Layer):
    """
    simple linear decoder with only one linear layer and the step to
    resize the one-dimensional vectors to two-dimensional masks.
    """
    def __init__(self, config):
        super().__init__()
        self.feature_size = (config.DATA.CROP_SIZE[0] // config.MODEL.TRANS.PATCH_SIZE,
                             config.DATA.CROP_SIZE[1] // config.MODEL.TRANS.PATCH_SIZE)
        self.head = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE, config.DATA.NUM_CLASSES)

    def forward(self, x):
        H, W = self.feature_size

        masks = self.head(x)
        #[b, (h w), n] -> [b, n, h, w]
        masks = masks.reshape((masks.shape[0], H, W, masks.shape[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return masks
