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
Shuffle Transformer in Paddle

A Paddle Implementation of ShuffleTransformer as described in:

"Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer"
	- Paper Link: https://arxiv.org/abs/2106.03650
"""

import numpy as np
import paddle
import paddle.nn as nn
from droppath import DropPath


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """
    def forward(self, x):
        return x


class PatchEmbedding(nn.Layer):
    """Patch embedding layer

    Apply patch embeddings on input images. Embeddings in implemented using
    2 stacked Conv2D layers.

    Attriubutes:
        image_size: int, input image size, default: 224
        patch_size: int, size of an image patch, default: 4
        in_channels: int, input image channels, default: 3
        inter_dim: int, intermediate dim for conv layers, default: 32
        embed_dim: int, embedding dimension, default: 48
    """
    def __init__(self,
                 image_size=224,
                 inter_dim=32,
                 embed_dim=48,
                 in_channels=3):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_batchnorm()
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels, inter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(inter_dim, weight_attr=w_attr_1, bias_attr=b_attr_1),
            nn.ReLU6())

        w_attr_2, b_attr_2 = self._init_weights_batchnorm()
        self.conv2 = nn.Sequential(
            nn.Conv2D(inter_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2),
            nn.ReLU6())

        self.conv3 = nn.Conv2D(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

        # 4 = stride * stride
        self.num_patches = (image_size // 4) * (image_size // 4)

    def _init_weights_batchnorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class MLP(nn.Layer):
    """MLP module

    A MLP layer which uses 1x1 conv instead of linear layers.
    ReLU6 is used as activation function.

    Args:
        in_features: int, input feature dim.
        hidden_features: int, hidden feature dim.
        out_features: int, output feature dim.
        dropout: flaot, dropout rate, default: 0.0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1, 1, 0)
        self.act = nn.ReLU6()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        out = self.fc1(inputs) # [batch_size, hidden_dim, height, width]
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class WindowAttention(nn.Layer):
    """ Window Multihead Aelf-attention Module.
    This module use 1x1 Conv as the qkv proj and linear proj
    Args:
        dim: int, input dimension.
        num_heads: int, number of attention heads.
        windows_size: int, the window size of attention modules, default: 1
        shuffle: bool, if True, use output shuffle, default: False
        qk_scale: float, if set, override default qk scale, default: None
        qkv_bias: bool, if True, enable bias to qkv, default: False
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=1,
                 shuffle=False,
                 qk_scale=None,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.window_size = window_size
        self.shuffle = shuffle
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Conv2D(dim, dim * 3, kernel_size=1, bias_attr=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Conv2D(dim, dim, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size - 1) * (2 * window_size - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        # relative position index for each token inside window
        coords_h = paddle.arange(0, self.window_size)
        coords_w = paddle.arange(0, self.window_size)
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))
        coords_flatten = paddle.flatten(coords, 1) # [2, window_h * window_w]
        # 2, window_h * window_w, window_h * window_h
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        # winwod_h*window_w, window_h*window_w, 2
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # [window_size * window_size, window_size*window_size]
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        table = self.relative_position_bias_table # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1])
        # window_h*window_w * window_h*window_w
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def transpose_multihead(self, x):
        B, C, H, W = x.shape
        n_window = H // self.window_size
        if self.shuffle:
            x = x.reshape([B,
                           self.num_heads,
                           self.head_dim,
                           self.window_size, # window_size first
                           n_window,
                           self.window_size,
                           n_window])
            x = x.transpose([0, 4, 6, 1, 3, 5, 2]) # order matters
        else:
            x = x.reshape([B,
                           self.num_heads,
                           self.head_dim,
                           n_window, # n_window first
                           self.window_size,
                           n_window,
                           self.window_size])
            x = x.transpose([0, 3, 5, 1, 4, 6, 2]) # order metters

        x = x.reshape([B * n_window * n_window,
                       self.num_heads,
                       self.window_size * self.window_size,
                       self.head_dim])
        return x

    def transpose_multihead_reverse(self, x, B, H, W):
        assert H == W
        n_window = H // self.window_size
        x = x.reshape([B,
                       n_window,
                       n_window,
                       self.num_heads,
                       self.window_size,
                       self.window_size,
                       self.head_dim])
        if self.shuffle:
            x = x.transpose([0, 3, 6, 4, 1, 5, 2])
        else:
            x = x.transpose([0, 3, 6, 1, 4, 2, 5])
        x = x.reshape([B,
                       self.num_heads * self.head_dim,
                       self.window_size * n_window,
                       self.window_size * n_window])
        return x

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        qkv = self.qkv(inputs).chunk(3, axis=1) # qkv is a tuple: (q, k, v)

        # Now q, k, and v has the following shape:
        # Case1: [B, (num_heads * head_dim), (window_size * n_window), (window_size * n_window)]
        # Case2: [B, (num_heads * head_dim), (n_window * window_size), (n_window * window_size)]
        # where Case 1 is used when shuffle is True, Case 2 is used for no shuffle

        # with/without spatial shuffle
        # shape = [(B * n_window * n_window), num_heads, (window_size * window_size), head_dim]
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)

        relative_position_bias = self.get_relative_pos_bias_from_pos_index()

        relative_position_bias = relative_position_bias.reshape(
            [self.window_size * self.window_size,
             self.window_size * self.window_size,
             -1])
        # nH, window_h * window_w, window_h * window_h
        relative_position_bias = paddle.transpose(relative_position_bias, perm=[2, 0, 1])

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        z = paddle.matmul(attn, v)


        # shape = [(B * n_window * n_window), num_heads, (window_size * window_size), head_dim]
        # new shape=[B, (num_heads * head_dim), (n_window * window_size), (n_window * window_size)]
        z = self.transpose_multihead_reverse(z, B, H, W)

        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class ShuffleBlock(nn.Layer):
    """Shuffle block layers

    Shuffle block layers contains multi head attention, conv,
    droppath, mlp, batch_norm and residual.

    Attributes:
        dim: int, embedding dimension
        out_dim: int, stage output dim
        num_heads: int, num of attention heads
        window_size: int, window size, default: 1
        shuffle: bool, if True, apply channel shuffle, default: False
        mlp_ratio: float, ratio of mlp hidden dim and input dim, default: 4.
        qk_scale: float, if set, override default qk scale, default: None
        qkv_bias: bool, if True, enable bias to qkv, default: False
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """
    def __init__(self,
                 dim,
                 out_dim,
                 num_heads,
                 window_size=1,
                 shuffle=False,
                 mlp_ratio=4,
                 qk_scale=None,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_batchnorm()
        self.norm1 = nn.BatchNorm2D(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    shuffle=shuffle,
                                    qk_scale=qk_scale,
                                    qkv_bias=qkv_bias,
                                    dropout=dropout,
                                    attention_dropout=attention_dropout)
        # neighbor-window connection enhancement (NWC)
        self.local = nn.Conv2D(dim,
                               dim,
                               kernel_size=window_size,
                               stride=1,
                               padding=window_size // 2,
                               groups=dim)
        self.drop_path = DropPath(droppath)
        w_attr_2, b_attr_2 = self._init_weights_batchnorm()
        self.norm2 = nn.BatchNorm2D(dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, out_dim, dropout)
        w_attr_3, b_attr_3 = self._init_weights_batchnorm()
        self.norm3 = nn.BatchNorm2D(dim, weight_attr=w_attr_3, bias_attr=b_attr_3)

    def _init_weights_batchnorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        # attention
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x
        # neighbor-window connection enhancement (NWC)
        h = x
        x = self.norm2(x)
        x = self.local(x)
        x = h + x
        # mlp
        h = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class PatchMerging(nn.Layer):
    """Patch Merging
    Merge the patches by a BatchNorm and a Conv2D with kernel size 2x2
    and stride 2, to reduce the number of tokens
    """
    def __init__(self, in_dim=32, out_dim=64):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_batchnorm()
        self.norm = nn.BatchNorm2D(in_dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.reduction = nn.Conv2D(in_dim,
                                   out_dim,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0,
                                   bias_attr=False)

    def _init_weights_batchnorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, inputs):
        out = self.norm(inputs)
        out = self.reduction(out)
        return out


class StageModule(nn.Layer):
    """Stage layer for shuffle transformer

    Stage layers contains a number of Transformer blocks and an optional
    patch merging layer, patch merging is not applied after last stage

    Attributes:
        num_layers: int, num of blocks in stage
        dim: int, embedding dimension
        out_dim: int, stage output dim
        num_heads: int, num of attention heads
        window_size: int, window size, default: 1
        mlp_ratio: float, ratio of mlp hidden dim and input dim, default: 4.
        qk_scale: float, if set, override default qk scale, default: None
        qkv_bias: bool, if True, enable bias to qkv, default: False
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """
    def __init__(self,
                 num_layers,
                 dim,
                 out_dim,
                 num_heads,
                 window_size=1,
                 shuffle=True,
                 mlp_ratio=4.,
                 qk_scale=None,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        assert num_layers % 2 == 0, "Stage layers must be even for shifted block."
        if dim != out_dim:
            self.patch_partition = PatchMerging(in_dim=dim, out_dim=out_dim)
        else:
            self.patch_partition = Identity()

        self.layers = nn.LayerList()
        for idx in range(num_layers):
            shuffle = idx % 2 != 0
            self.layers.append(ShuffleBlock(dim=out_dim,
                                            out_dim=out_dim,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            shuffle=shuffle,
                                            mlp_ratio=mlp_ratio,
                                            qk_scale=qk_scale,
                                            qkv_bias=qkv_bias,
                                            dropout=dropout,
                                            attention_dropout=attention_dropout,
                                            droppath=droppath))

    def forward(self, inputs):
        out = self.patch_partition(inputs)
        for layer in self.layers:
            out = layer(out)

        return out


class ShuffleTransformer(nn.Layer):
    """Shuffle Transformer
    Args:
        image_size: int, input image size, default: 224
        num_classes: int, num of classes, default: 1000
        token_dim: int, intermediate feature dim in PatchEmbedding, default: 32
        embed_dim: int, embedding dim (out dim for PatchEmbedding), default: 96
        mlp_ratio: float, ratio for mlp dim, mlp hidden_dim = mlp in_dim * mlp_ratio, default: 4.
        layers: list of int, num of layers in each stage, default: [2, 2, 6, 2]
        num_heads: list of int, num of heads in each stage, default: [3, 6, 12, 24]
        window_size: int, attention window size, default: 7
        qk_scale: float, if set, override default qk scale (head_dim**-0.5), default: None
        qkv_bias: bool, if True, qkv layers is set with bias, default: False
        attention_dropout: float, dropout rate of attention, default: 0.0
        dropout: float, dropout rate for output, default: 0.0
        droppath: float, droppath rate, default: 0.0
    """

    def __init__(self,
                 image_size=224,
                 num_classes=1000,
                 token_dim=32,
                 embed_dim=96,
                 mlp_ratio=4.,
                 layers=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 qk_scale=None,
                 qkv_bias=False,
                 attention_dropout=0.,
                 dropout=0.,
                 droppath=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        dims = [embed_dim]
        dims.extend([i * 32 for i in num_heads]) # dims for each stage

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              inter_dim=token_dim,
                                              embed_dim=embed_dim)
        #num_patches = self.patch_embedding.num_patches
        self.num_stages = len(layers)
        dprs = [x.item() for x in np.linspace(0, droppath, self.num_stages)]

        self.stages = nn.LayerList()
        for i in range(self.num_stages):
            self.stages.append(StageModule(layers[i],
                                           dims[i],
                                           dims[i+1],
                                           num_heads[i],
                                           window_size=window_size,
                                           mlp_ratio=mlp_ratio,
                                           qk_scale=qk_scale,
                                           qkv_bias=qkv_bias,
                                           attention_dropout=attention_dropout,
                                           dropout=dropout,
                                           droppath=dprs[i]))
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        w_attr_1, b_attr_1 = self._init_weights()
        self.head = nn.Linear(dims[-1], num_classes, weight_attr=w_attr_1, bias_attr=b_attr_1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward_features(self, inputs):
        out = self.patch_embedding(inputs)
        B, C, H, W = out.shape

        for idx, stage in enumerate(self.stages):
            out = stage(out)

        out = self.avgpool(out)
        out = paddle.flatten(out, 1)
        return out

    def forward(self, inputs):
        out = self.forward_features(inputs)
        out = self.head(out)
        return out


def build_shuffle_transformer(config):
    """ build shuffle transformer using config"""
    model = ShuffleTransformer(image_size=config.DATA.IMAGE_SIZE,
                               num_classes=config.MODEL.NUM_CLASSES,
                               token_dim=config.MODEL.TOKEN_DIM,
                               embed_dim=config.MODEL.EMBED_DIM,
                               mlp_ratio=config.MODEL.MLP_RATIO,
                               layers=config.MODEL.STAGE_DEPTHS,
                               num_heads=config.MODEL.NUM_HEADS,
                               window_size=config.MODEL.WINDOW_SIZE,
                               qk_scale=config.MODEL.QK_SCALE,
                               qkv_bias=config.MODEL.QKV_BIAS,
                               attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                               dropout=config.MODEL.DROPOUT,
                               droppath=config.MODEL.DROPPATH)
    return model
