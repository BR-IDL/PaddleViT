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
Implement DeiT
"""

import math
import copy
import numpy as np
import paddle
import paddle.nn as nn
from droppath import DropPath


class Identity(nn.Layer):
    """ Identity layer
    
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class PatchEmbedding(nn.Layer):
    """Patch Embeddings

    Then a proj (conv2d) layer is applied as the patch embedding.

    Args:
        image_size: int, input image size, default: 224
        patch_size: int, patch size for patch embedding (k and stride for proj conv), default: 8
        in_channels: int, input channels, default: 3
        embed_dim: int, output dimension of patch embedding, default: 384
    """
    def __init__(self,
                 image_size=224,
                 patch_size=8,
                 in_channels=3,
                 embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        
        # define patch embeddings
        self.proj = nn.Conv2D(in_channels,
                              embed_dim,
                              kernel_size = patch_size,
                              stride = patch_size)
        # num patches
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
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
    
    def __init__(self, in_features, hidden_features, dropout=0.):
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
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """ Attention

    Regular Attention module same as ViT

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim,
                             dim * 3,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim,
                              dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.proj_dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])

        new_shape = z.shape[:-2] + [self.embed_dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class EncoderLayer(nn.Layer):
    """Transformer Encoder Layer

    Transformer encoder module, same as ViT

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: float, ratio to multiply with dim for mlp hidden feature dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.norm1 = nn.LayerNorm(dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1,
                                  epsilon=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights()
        self.norm2 = nn.LayerNorm(dim,
                                  weight_attr=w_attr_2,
                                  bias_attr=b_attr_2,
                                  epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio))
    
    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x

        return x


class Deit(nn.Layer):
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 num_classes=1000,
                 patch_size=16,
                 embed_dim=192,
                 num_heads=3,
                 depth=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.num_classes = num_classes
        # patch embedding
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim)
        # class token
        self.class_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.))
        # distillation token
        self.distill_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        # positional embedding
        self.pos_embed = paddle.create_parameter(
            shape=[1, self.patch_embed.num_patches + 2, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.pos_dropout = nn.Dropout(dropout)

        self.layers = nn.LayerList([
            copy.deepcopy(EncoderLayer(dim=embed_dim,
                                       num_heads=num_heads,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias,
                                       attention_dropout=attention_dropout,
                                       droppath=droppath)) for _ in range(depth)])
        w_attr_1, b_attr_1 = self._init_weights_norm()
        self.norm = nn.LayerNorm(embed_dim,
                                 weight_attr=w_attr_1,
                                 bias_attr=b_attr_1,
                                 epsilon=1e-6)

        w_attr_2, b_attr_2 = self._init_weights_linear()
        self.head = nn.Linear(embed_dim,
                              num_classes,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        w_attr_3, b_attr_3 = self._init_weights_linear()
        self.head_distill = nn.Linear(embed_dim,
                                      num_classes, 
                                      weight_attr=w_attr_3,
                                      bias_attr=b_attr_3)

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward_features(self, x):
        x = self.patch_embed(x)
        class_tokens = self.class_token.expand([x.shape[0], -1, -1])
        distill_tokens = self.distill_token.expand([x.shape[0], -1, -1])
        x = paddle.concat((class_tokens, distill_tokens, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_distill = self.forward_features(x)
        x = self.head(x)
        x_distill = self.head_distill(x_distill)
        if self.training:
            return x, x_distill
        else:
            return (x + x_distill) / 2


def build_deit(config):
    """build deit model using config"""
    model = Deit(image_size=config.DATA.IMAGE_SIZE,
                 in_channels=config.MODEL.TRANS.IN_CHANNELS,
                 num_classes=config.MODEL.NUM_CLASSES,
                 patch_size=config.MODEL.TRANS.PATCH_SIZE,
                 embed_dim=config.MODEL.TRANS.EMBED_DIM,
                 num_heads=config.MODEL.TRANS.NUM_HEADS,
                 depth=config.MODEL.TRANS.DEPTH,
                 mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                 qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                 dropout=config.MODEL.DROPOUT,
                 attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                 droppath=config.MODEL.DROPPATH)
    return model
