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
PiT in Paddle
A Paddle Implementation of PiT as described in:
"Rethinking Spatial Dimensions of Vision Transformers"
    - Paper Link: https://arxiv.org/abs/2103.16302
"""

import math
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from droppath import DropPath


trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def forward(self, x):
        return x


class Mlp(nn.Layer):
    """ MLP module

    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout

    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc1 and fc2
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape([B, N, 3, self.num_heads, C // self.num_heads])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Layer):
    def __init__(self,
                 base_dim,
                 depth,
                 heads,
                 mlp_ratio,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_prob=None):
        super().__init__()
        self.layers = nn.LayerList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.LayerList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, cls_tokens):
        b, c, h, w = x.shape
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = x.transpose([0, 2, 3, 1]).reshape([b, h * w, c])

        token_length = cls_tokens.shape[1]
        x = paddle.concat([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose([0, 2, 1]).reshape([b, c, h, w])

        return x, cls_tokens


class ConvHeadPooling(nn.Layer):
    def __init__(self, in_feature, out_feature, stride, padding_mode="zeros"):
        super().__init__()

        self.conv = nn.Conv2D(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            padding_mode=padding_mode,
            groups=in_feature,
        )
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Layer):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias_attr=True,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Layer):
    def __init__(self,
                 image_size,
                 patch_size,
                 stride,
                 base_dims,
                 depth,
                 heads,
                 mlp_ratio=4,
                 num_classes=1000,
                 in_chans=3,
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 drop_path_rate=0.0):
        super().__init__()
        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor((image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.depth = depth

        self.patch_size = patch_size

        self.pos_embed = paddle.create_parameter(
            shape=[1, base_dims[0] * heads[0], width, width],
            dtype="float32",
            default_initializer=trunc_normal_,
        )

        self.patch_embed = ConvEmbedding(
            in_chans, base_dims[0] * heads[0], patch_size, stride, padding
        )

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, base_dims[0] * heads[0]],
            dtype="float32",
            default_initializer=trunc_normal_,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.LayerList([])
        self.pools = nn.LayerList([])

        for stage, stage_depth in enumerate(self.depth):
            drop_path_prob = [
                drop_path_rate * i / total_block
                for i in range(block_idx, block_idx + stage_depth)
            ]
            block_idx += stage_depth

            self.transformers.append(
                Transformer(
                    base_dims[stage],
                    stage_depth,
                    heads[stage],
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_prob,
                )
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    ConvHeadPooling(
                        base_dims[stage] * heads[stage],
                        base_dims[stage + 1] * heads[stage + 1],
                        stride=2,
                    )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], epsilon=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1])

        for stage, pool_layer in enumerate(self.pools):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = pool_layer(x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cls_token = paddle.create_parameter(
            shape=[1, 2, self.base_dims[0] * self.heads[0]],
            dtype="float32",
            default_initializer=trunc_normal_,
        )

        if self.num_classes > 0:
            self.head_dist = nn.Linear(
                self.base_dims[-1] * self.heads[-1], self.num_classes
            )
        else:
            self.head_dist = Identity()

        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        return (x_cls + x_dist) / 2


def build_pit(config):
    if config.TRAIN.DISTILLATION_TYPE != 'none':
        model = DistilledPoolingTransformer(
            image_size=config.DATA.IMAGE_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=config.MODEL.PATCH_SIZE,
            stride=config.MODEL.STRIDE,
            base_dims=config.MODEL.BASE_DIMS,
            depth=config.MODEL.DEPTH,
            heads=config.MODEL.NUM_HEADS,
            mlp_ratio=config.MODEL.MLP_RATIO,
            in_chans=config.DATA.IMAGE_CHANNELS,
            attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
            drop_rate=config.MODEL.DROPOUT,
            drop_path_rate=config.MODEL.DROPPATH,
        )
    else:
        model = PoolingTransformer(
            image_size=config.DATA.IMAGE_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=config.MODEL.PATCH_SIZE,
            stride=config.MODEL.STRIDE,
            base_dims=config.MODEL.BASE_DIMS,
            depth=config.MODEL.DEPTH,
            heads=config.MODEL.NUM_HEADS,
            mlp_ratio=config.MODEL.MLP_RATIO,
            in_chans=config.DATA.IMAGE_CHANNELS,
            attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
            drop_rate=config.MODEL.DROPOUT,
            drop_path_rate=config.MODEL.DROPPATH,
        )

    return model
