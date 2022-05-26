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
ResT/ResTV2 in Paddle
A Paddle Implementation of ResT/ResTV2 as described in:
"ResT: An Efficient Transformer for Visual Recognition" 
    - Paper Link: https://arxiv.org/abs/2105.13677
"ResT V2: Simpler, Faster and Stronger"
    - Paper Link: https://arxiv.org/abs/2104.06399
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from droppath import DropPath


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

        w_attr, b_attr = self._init_weights_layer()
        self.fc1 = nn.Linear(in_features, hidden_features, weight_attr=w_attr, bias_attr=b_attr)
        self.act = act_layer()
        w_attr, b_attr = self._init_weights_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, weight_attr=w_attr, bias_attr=b_attr)
        self.drop = nn.Dropout(drop)

    def _init_weights_layer(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 sr_ratio=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads
        self.scale = qk_scale or self.attn_head_size ** -0.5

        w_attr_1, b_attr_1 = self._init_weights_layer()
        self.q = nn.Linear(embed_dim,
                           self.all_head_size,  # weights for q
                           weight_attr=w_attr_1,
                           bias_attr=b_attr_1 if qkv_bias else False)

        w_attr_2, b_attr_2 = self._init_weights_layer()
        self.kv = nn.Linear(embed_dim,
                            self.all_head_size * 2,  # weights for k,v
                            weight_attr=w_attr_2,
                            bias_attr=b_attr_2 if qkv_bias else False)

        w_attr_3, b_attr_3 = self._init_weights_layer()
        self.proj = nn.Linear(self.all_head_size,
                              embed_dim,
                              weight_attr=w_attr_3,
                              bias_attr=b_attr_3)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)
    
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(embed_dim,
                                embed_dim,
                                kernel_size=sr_ratio + 1,
                                stride=sr_ratio,
                                padding=sr_ratio // 2,
                                groups=embed_dim)
            w_attr, b_attr = self._init_weights_norm()
            self.sr_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

        self.up = nn.Sequential(
            nn.Conv2D(embed_dim,
                      sr_ratio * sr_ratio * embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=embed_dim),
            nn.PixelShuffle(upscale_factor=sr_ratio),
        )
        w_attr, b_attr = self._init_weights_norm()
        self.up_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

    def _init_weights_layer(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape([B, N, self.num_heads, self.attn_head_size])
        q = q.transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = x.transpose([0, 2, 1])  # [B, N, C] -> [B, C, N]
            x = x.reshape([B, C, H, W])  # [B, C, N] -> [B, C, H, W]
            x = self.sr(x)
            x = x.reshape([B, C, -1])  # [B, C, H, W] -> [B, C, N]
            x = x.transpose([0, 2, 1])  # [B, C, N] -> [B, N, C]
            x = self.sr_norm(x)

        kv = self.kv(x)
        kv = kv.reshape([B, -1, 2, self.num_heads, self.attn_head_size])
        kv = kv.transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        z = z.reshape([B, N, C])

        identity = v.transpose([0, 1, 3, 2])
        identity = identity.reshape([B, C, H // self.sr_ratio, W // self.sr_ratio])
        identity = self.up(identity)
        identity = identity.flatten(2).transpose([0, 2, 1])
        identity = self.up_norm(identity)

        z = z + identity
        z = self.proj(z)
        z = self.proj_dropout(z)
        return z
        

class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 sr_ratio=1):
        super().__init__()
        w_attr, b_attr = self._init_weights_norm()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        self.attn = Attention(embed_dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attention_dropout=attention_dropout,
                              dropout=dropout,
                              sr_ratio=sr_ratio)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr, b_attr = self._init_weights_norm()
        self.norm2 = nn.LayerNorm(dim, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=dropout)

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, H, W):
        h = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class PA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2D(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Layer):
    def __init__(self,
                 patch_size=2,
                 in_channels=3,
                 out_channels=96,
                 with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=patch_size + 1,
                              stride=patch_size,
                              padding=patch_size // 2)

        w_attr, b_attr = self._init_weights_norm()
        self.norm = nn.LayerNorm(out_channels, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

        self.pos = PA(out_channels) if with_pos else Identity()

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = self.pos(x)
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm(x)
        H, W = H // self.patch_size, W // self.patch_size
        return x, (H, W)


class ConvStem(nn.Layer):
    def __init__(self, patch_size=2, in_channels=3, out_channels=96, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        stem = []
        in_dim, out_dim = in_channels, out_channels // 2
        for i in range(2):
            stem.append(nn.Conv2D(in_dim,
                                  out_dim,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  bias_attr=False))
            stem.append(nn.BatchNorm2D(out_dim))
            stem.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim * 2
        stem.append(nn.Conv2D(in_dim,
                              out_channels,
                              kernel_size=1,
                              stride=1))
        self.proj = nn.Sequential(*stem)

        self.pos = PA(out_channels) if with_pos is True else Identity()

        w_attr, b_attr = self._init_weights_norm()
        self.norm = nn.LayerNorm(out_channels, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = self.pos(x)
        x = x.flatten(2).transpose([0, 2, 1])  # [B, C, H, W] -> [B, N, C]
        x = self.norm(x)
        H = H // self.patch_size
        W = W // self.patch_size
        return x, (H, W)


class Stem(nn.Layer):
    def __init__(self, in_channels=3, out_channels=96, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=patch_size,
                              stride=patch_size)
        w_attr, b_attr = self._init_weights_norm()
        self.norm = nn.LayerNorm(out_channels, weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose([0, 2, 1])  # [B, C, H, W] -> [B, N, C]
        x = self.norm(x)
        H = H // self.patch_size
        W = W // self.patch_size
        return x, (H, W)


class ResTV2(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Stem Layers
        self.stem = ConvStem(patch_size=4,
                             in_channels=in_channels,
                             out_channels=embed_dims[0])
        # Patch Embeddings
        self.patch_embeds = nn.LayerList([
            PatchEmbed(patch_size=2,
                       in_channels=embed_dims[i],
                       out_channels=embed_dims[i+1],
                       with_pos=True) for i in range(3)
        ])
        # Encoder
        dpr = [x.item() for x in paddle.linspace(0, dropout, sum(depths))]
        self.stages = nn.LayerList()
        cur = 0 
        for idx, (depth, embed_dim, num_head, mlp_ratio, sr_ratio) in enumerate(
            zip(self.depths, embed_dims, num_heads, mlp_ratios, sr_ratios)):
            self.stages.append(
                nn.LayerList([
                    Block(embed_dim,
                          num_head,
                          mlp_ratio,
                          qkv_bias,
                          qk_scale,
                          dropout,
                          attention_dropout,
                          droppath=dpr[cur + i],
                          sr_ratio=sr_ratio)
                    for i in range(depth)])
            )
            cur += depth

        w_attr, b_attr = self._init_weights_norm()
        self.norm = nn.LayerNorm(embed_dims[3], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        # Head
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        w_attr, b_attr = self._init_weights_layer()
        self.head = nn.Linear(embed_dims[3],
                              num_classes,
                              weight_attr=w_attr,
                              bias_attr=b_attr) if num_classes > 0 else Identity()

    def _init_weights_layer(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, _, H, W = x.shape
        x, (H, W) = self.stem(x)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                x, (H, W) = self.patch_embeds[stage_idx - 1](x)
            
            for block_idx, block in enumerate(stage):
                x = block(x, H, W)

            if stage_idx == 3:
                x = self.norm(x)

            x = x.transpose([0, 2, 1]).reshape([B, -1, H, W])

        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x


def build_restv2(config):
    """build rest model from config"""
    model = ResTV2(in_channels=config.DATA.IMAGE_CHANNELS,
                 num_classes=config.MODEL.NUM_CLASSES,
                 embed_dims=config.MODEL.EMBED_DIMS,
                 num_heads=config.MODEL.NUM_HEADS,
                 mlp_ratios=config.MODEL.MLP_RATIOS,
                 depths=config.MODEL.DEPTHS,
                 sr_ratios=config.MODEL.SR_RATIOS,
                 qkv_bias=config.MODEL.QKV_BIAS,
                 qk_scale=config.MODEL.QK_SCALE,
                 dropout=config.MODEL.DROPOUT,
                 attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                 droppath=config.MODEL.DROPPATH)
    return model
