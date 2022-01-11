#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
Implement MLP Class for ResMLP
"""

import math
import copy
import paddle
import paddle.nn as nn
from droppath import DropPath


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class PatchEmbedding(nn.Layer):
    """Patch Embeddings

    Apply patch embeddings on input images. Embeddings is implemented using a Conv2D op.

    Attributes:
        image_size: int, input image size, default: 224
        patch_size: int, size of patch, default: 4
        in_channels: int, input image channels, default: 3
        embed_dim: int, embedding dimension, default: 96
    """

    def __init__(self, image_size=224, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super(PatchEmbedding, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2D(in_channels=in_channels,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.norm = norm_layer if norm_layer is not None else Identity()

    def forward(self, x):
        x = self.patch_embed(x) # [batch, embed_dim, h, w] h,w = patch_resolution
        x = x.flatten(start_axis=2, stop_axis=-1) # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1]) # [batch, h*w, embed_dim]
        x = self.norm(x) # [batch, num_patches, embed_dim]
        return x


class Mlp(nn.Layer):
    """ MLP module
    
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """
    
    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
    
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Layer):
    def __init__(self, dim, seq_len, mlp_ratio=4, init_values=1e-5, dropout=0., droppath=0.):
        super(ResBlock, self).__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = Affine(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(droppath)
        self.norm2 = Affine(dim)
        self.mlp_channels = Mlp(dim, channels_dim, dropout=dropout)

        self.ls1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

        self.ls2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = x.transpose([0, 2, 1])
        x = self.linear_tokens(x)
        x = x.transpose([0, 2, 1])
        x = self.ls1 * x
        x = self.drop_path(x)
        x = x + h
        
        h = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = self.ls2 * x
        x = self.drop_path(x)
        x = x + h

        return x


class Affine(nn.Layer):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.alpha = paddle.create_parameter(
            shape=[1, 1, dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(1))

        self.beta = paddle.create_parameter(
            shape=[1, 1, dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x = paddle.multiply(self.alpha, x)
        x = self.beta + x
        return x


class ResMlp(nn.Layer):
    def __init__(self,
                 num_classes=1000,
                 image_size=224,
                 in_channels=3,
                 patch_size=16,
                 num_mixer_layers=24,
                 embed_dim=384,
                 mlp_ratio=4,
                 dropout=0.,
                 droppath=0.,
                 patch_embed_norm=False):
        super(ResMlp, self).__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        norm_layer=nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_embed_norm else None)

        self.mixer_layers = nn.Sequential(
            *[ResBlock(embed_dim,
                         self.patch_embed.num_patches,
                         mlp_ratio,
                         dropout,
                         droppath) for _ in range(num_mixer_layers)])

        self.norm = Affine(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(axis=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_res_mlp(config):
    model = ResMlp(num_classes=config.MODEL.NUM_CLASSES,
                   image_size=config.DATA.IMAGE_SIZE,
                   patch_size=config.MODEL.MIXER.PATCH_SIZE,
                   in_channels=3,
                   num_mixer_layers=config.MODEL.MIXER.NUM_LAYERS,
                   embed_dim=config.MODEL.MIXER.HIDDEN_SIZE,
                   mlp_ratio=4,
                   dropout=config.MODEL.DROPOUT,
                   droppath=config.MODEL.DROP_PATH)
    return model
