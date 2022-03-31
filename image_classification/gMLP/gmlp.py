# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
gMLP in Paddle
A Paddle implementation of gMLP as described in:
"Pay Attention to MLPs"
    - Paper Link: https://arxiv.org/abs/2105.08050
"""
import math
import copy
from functools import partial
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


class GMlp(nn.Layer):
    """ GatedMLP module
    
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> gate -> fc -> dropout
    
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        gate: gate layer
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """
    
    def __init__(self, in_features, hidden_features, gate_layer=None, dropout=0.):
        super(GMlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = Identity()
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
        x = self.gate(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SpatialGatingUnit(nn.Layer):
    def __init__(self, dim, seq_len):
        super(SpatialGatingUnit, self).__init__()
        gate_dim = dim // 2
        self.norm = nn.LayerNorm(gate_dim, epsilon=1e-6)
        w_attr, b_attr = self._init_weights()
        self.proj = nn.Linear(seq_len,
                              seq_len,
                              weight_attr=w_attr,
                              bias_attr=b_attr)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(std=1e-6))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1))
        return weight_attr, bias_attr

    def forward(self, x):
        u, v = x.chunk(2, axis=-1)
        v = self.norm(v)
        v = self.proj(v.transpose([0, 2, 1]))
        return u * v.transpose([0, 2, 1])


class SpatialGatingBlock(nn.Layer):
    def __init__(self, dim, seq_len, mlp_ratio=4, dropout=0., droppath=0.):
        super(SpatialGatingBlock, self).__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = GMlp(dim, channels_dim, gate_layer=sgu, dropout=dropout)
        self.drop_path = DropPath(droppath)

    def forward(self, x):
        h = x
        x = self.norm(x)
        x = self.mlp_channels(x)
        x = self.drop_path(x)
        x = x + h

        return x


class GatedMlp(nn.Layer):
    def __init__(self,
                 num_classes=1000,
                 image_size=224,
                 in_channels=3,
                 patch_size=16,
                 num_mixer_layers=30,
                 embed_dim=256,
                 mlp_ratio=6,
                 dropout=0.,
                 droppath=0.,
                 patch_embed_norm=False):
        super(GatedMlp, self).__init__()
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
            *[SpatialGatingBlock(
                embed_dim,
                self.patch_embed.num_patches,
                mlp_ratio,
                dropout,
                droppath) for _ in range(num_mixer_layers)])

        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)
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


def build_gmlp(config):
    model = GatedMlp(num_classes=config.MODEL.NUM_CLASSES,
                     image_size=config.DATA.IMAGE_SIZE,
                     in_channels=config.DATA.IMAGE_CHANNELS,
                     num_mixer_layers=config.MODEL.MIXER.DEPTH,
                     embed_dim=config.MODEL.MIXER.EMBED_DIM,
                     mlp_ratio=config.MODEL.MIXER.MLP_RATIO,
                     dropout=config.MODEL.DROPOUT,
                     droppath=config.MODEL.DROPPATH)
    return model
