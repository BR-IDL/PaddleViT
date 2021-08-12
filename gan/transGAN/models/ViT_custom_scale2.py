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
Implement ViT_custom_scale2
"""

import numpy as np
import paddle
import paddle.nn as nn
from utils import trunc_normal_
from utils import gelu
from utils import pixel_upsample
from utils import drop_path
from utils import DiffAugment
from utils import leakyrelu
from utils import normal_
from utils import uniform_
from utils import constant_
from models.ViT_custom import Identity
from models.ViT_custom import matmul
from models.ViT_custom import PixelNorm
from models.ViT_custom import CustomNorm
from models.ViT_custom import Mlp
from models.ViT_custom import CustomAct
from models.ViT_custom import DropPath

class Attention(nn.Layer):
    """ attention layer

    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        dim: defalut embedding dim, set in config
        num_heads: number of heads
        qkv_bias: a nn.Linear for q, k, v mapping
        qk_scale: 1 / sqrt(single_head_feature_dim)
        attn_drop: dropout for attention
        proj_drop: final dropout before output
        softmax: softmax op for attention
        window_size: attention size

    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size
        zeros_ = nn.initializer.Constant(value=0.)
        self.noise_strength_1 = self.create_parameter(shape=[1], default_initializer=zeros_)

        if self.window_size != 0:
            zeros_ = nn.initializer.Constant(value=0.)
            self.relative_position_bias_table = self.create_parameter(
                shape=((2 * window_size - 1) * (2 * window_size - 1),
                       num_heads), default_initializer=zeros_)
            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size)
            coords_w = paddle.arange(window_size)
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            # 2, Wh*Ww, Wh*Ww
            relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
            relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        x = x + paddle.randn([x.shape[0], x.shape[1], 1]) * self.noise_strength_1
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose([0, 1, 3, 2]))) * self.scale
        if self.window_size != 0:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.flatten().clone()]
            relative_position_bias = relative_position_bias.reshape((
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1))
            # nH, Wh*Ww, Wh*Ww
            relative_position_bias = relative_position_bias.transpose((2, 0, 1))
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DisBlock(nn.Layer):
    """ block layer
    Make up the basic unit of the network

    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=leakyrelu,
                 norm_layer=nn.LayerNorm,
                 window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x*self.gain + self.drop_path(self.attn(self.norm1(x)))*self.gain
        x = x*self.gain + self.drop_path(self.mlp(self.norm2(x)))*self.gain
        return x


class Discriminator(nn.Layer):
    """ Discriminator layer

    Discriminator module for transGAN
    Attributes:
        args: the input args
        img_size: the size of img
        patch_size: the patch size of the attention
        num_classes: the num of class, There are actually only two
        embed_dim: the dim of embedding dim
        depth: the block depth
        num_heads: number of heads
        mlp_ratio: decide the mlp_hidden_dim, defalut 4
        qkv_bias: a nn.Linear for q, k, v mapping
        qk_scale: 1 / sqrt(single_head_feature_dim)
        drop_rate: the dropout before output
        attn_drop_rate:  dropout for attention
        drop_path_rate: the dropout before output
        norm_layer: which norm method

    """
    def __init__(self,
                 args,
                 patch_size=None,
                 num_classes=1,
                 embed_dim=None,
                 depth=7,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.MODEL.DF_DIM
        depth = args.MODEL.D_DEPTH
        self.args = args
        self.patch_size = patch_size = args.MODEL.PATCH_SIZE
        norm_layer = args.MODEL.D_NORM
        self.window_size = args.MODEL.D_WINDOW_SIZE
        act_layer = args.MODEL.D_ACT
        self.fRGB_1 = nn.Conv2D(3,
                                embed_dim//4*3,
                                kernel_size=patch_size,
                                stride=patch_size, padding=0)
        self.fRGB_2 = nn.Conv2D(3,
                                embed_dim//4,
                                kernel_size=patch_size*2,
                                stride=patch_size*2,
                                padding=0)

        num_patches_1 = (args.DATA.IMG_SIZE // patch_size)**2
        num_patches_2 = ((args.DATA.IMG_SIZE//2) // patch_size)**2

        zeros_ = nn.initializer.Constant(value=0.)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.pos_embed_1 = self.create_parameter(
            shape=(1, int(num_patches_1), embed_dim//4*3), default_initializer=zeros_)
        self.pos_embed_2 = self.create_parameter(
            shape=(1, int(num_patches_2), embed_dim), default_initializer=zeros_)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth decay rule
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks_1 = nn.LayerList([
            DisBlock(dim=embed_dim//4*3,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=0,
                     act_layer=act_layer,
                     norm_layer=norm_layer,
                     window_size=args.MODEL.BOTTOM_WIDTH*4//2) for i in range(depth)])
        self.blocks_2 = nn.LayerList([
            DisBlock(dim=embed_dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=0,
                     act_layer=act_layer,
                     norm_layer=norm_layer,
                     window_size=args.MODEL.BOTTOM_WIDTH*4//4) for i in range(depth)])

        self.last_block = nn.Sequential(
            DisBlock(dim=embed_dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=dpr[0],
                     act_layer=act_layer,
                     norm_layer=norm_layer,
                     window_size=0)
            )

        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()

        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.Hz_fbank = None
        self.Hz_geom = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)

    def forward_features(self, x, aug=True, epoch=0):
        if "None" not in self.args.DATA.DIFF_AUG and aug:
            x = DiffAugment(x, self.args.DATA.DIFF_AUG, True, [self.Hz_geom, self.Hz_fbank])
        B, _, H, W = x.shape
        H = W = H//self.patch_size
        x_1 = self.fRGB_1(x).flatten(2).transpose([0, 2, 1])
        x_2 = self.fRGB_2(x).flatten(2).transpose([0, 2, 1])
        B = x.shape[0]
        x = x_1 + self.pos_embed_1
        B, _, C = x.shape
        for blk in self.blocks_1:
            x = blk(x)
        _, _, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = nn.AvgPool2D(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose([0, 2, 1])
        x = paddle.concat([x, x_2], axis=-1)
        x = x + self.pos_embed_2
        for blk in self.blocks_2:
            x = blk(x)
        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat((cls_tokens, x), axis=1)
        x = self.last_block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, aug=True, epoch=0):
        x = self.forward_features(x, aug=aug, epoch=epoch)
        x = self.head(x)
        return x
