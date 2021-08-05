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
Implement transGAN_custom
"""

import paddle
import paddle.nn as nn
from utils_paddle import trunc_normal_, gelu, pixel_upsample, drop_path

class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class matmul(nn.Layer):
    """ matmul layer

    Matrix-vector multiplication, like np.dot(x1, x2)

    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1@x2
        return x

class PixelNorm(nn.Layer):
    """ PixelNorm layer

    Pixel level norm

    """
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * paddle.rsqrt(paddle.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

class CustomNorm(nn.Layer):
    """ CustomNorm layer

    Custom norm method set, defalut "ln"

    """
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1D(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1D(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2, 1))
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)

class CustomAct(nn.Layer):
    """ CustomAct layer

    Custom act method set, defalut "gelu"

    """
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        else:
            self.act_layer = gelu

    def forward(self, x):
        return self.act_layer(x)

class Mlp(nn.Layer):
    """ mlp layer

    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2

    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=gelu,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Layer):
    """ attention layer

    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        num_heads: number of heads
        qkv_bias: a nn.Linear for q, k, v mapping
        qk_scale: 1 / sqrt(single_head_feature_dim)
        attn_drop: dropout for attention
        proj_drop: final dropout before output
        softmax: softmax op for attention
        window_size: window_size

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
        if self.window_size != 0:
            zeros_ = nn.initializer.Constant(value=0.)
             # 2*Wh-1 * 2*Ww-1, nH
            self.relative_position_bias_table = self.create_parameter(
                shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
                default_initializer=zeros_
            )
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
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose([0, 1, 3, 2]))) * self.scale
        if self.window_size != 0:
            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table,
                self.relative_position_index.flatten().clone())
            relative_position_bias = relative_position_bias.reshape((
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1))  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose((2, 0, 1)) # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Layer):
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
                 act_layer=gelu,
                 norm_layer=nn.LayerNorm,
                 window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StageBlock(nn.Layer):
    """ stageblock layer
    Organize Block

    """
    def __init__(self,
                 depth,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=gelu,
                 norm_layer=nn.LayerNorm,
                 window_size=16):
        super().__init__()
        self.depth = depth
        self.block = nn.LayerList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                window_size=window_size
            ) for i in range(depth)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x



class Generator(nn.Layer):
    """ generator layer

    Generator module for transGAN
    Attributes:
        args: args
        img_size: the resize size of img
        patch_size: the patch size of the attention
        in_chans: img's channel
        embed_dim: the dim of embedding dim
        depth: the block's depth
        num_heads: number of MLP heads
        mlp_ratio: decide the mlp_hidden_dim, defalut 4
        qkv_bias: a nn.Linear for q, k, v mapping
        qk_scale: 1 / sqrt(single_head_feature_dim)
        drop_rate: the dropout before output
        attn_drop_rate:  dropout for attention
        drop_path_rate: the dropout before output
        hybrid_backbone: if there some hybrid_backbone
        norm_layer: which norm method

    """
    def __init__(self,
                 args,
                 embed_dim=384,
                 depth=5,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer="ln"):
        super().__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.MODEL.BOTTOM_WIDTH
        self.embed_dim = embed_dim = args.MODEL.GF_DIM
        norm_layer = args.MODEL.G_NORM
        mlp_ratio = args.MODEL.G_MLP
        depth = [int(i) for i in args.MODEL.G_DEPTH.split(",")]
        act_layer = args.MODEL.G_ACT

        zeros_ = nn.initializer.Constant(value=0.)
        self.l1 = nn.Linear(args.MODEL.LATENT_DIM, (self.bottom_width ** 2) * self.embed_dim)
        self.pos_embed_1 = self.create_parameter(
            shape=(1, self.bottom_width**2, embed_dim), default_initializer=zeros_)
        self.pos_embed_2 = self.create_parameter(
            shape=(1, (self.bottom_width*2)**2, embed_dim//4), default_initializer=zeros_)
        self.pos_embed_3 = self.create_parameter(
            shape=(1, (self.bottom_width*4)**2, embed_dim//16), default_initializer=zeros_)
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]
        self.blocks = StageBlock(
            depth=depth[0],
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=8)
        self.upsample_blocks = nn.LayerList([
            StageBlock(
                depth=depth[1],
                dim=embed_dim//4,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0,
                act_layer=act_layer,
                norm_layer=norm_layer
                ),
            StageBlock(
                depth=depth[2],
                dim=embed_dim//16,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0,
                act_layer=act_layer,
                norm_layer=norm_layer,
                window_size=32
                )
        ])
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)
        self.deconv = nn.Sequential(
            nn.Conv2D(self.embed_dim//16, 3, 1, 1, 0)
        )

    def set_arch(self, x, cur_stage):
        pass

    def forward(self, z, epoch):
        if self.args.LATENT_NORM:
            latent_size = z.shape[-1]
            z = (z/z.norm(axis=-1, keepdim=True) * (latent_size ** 0.5))

        x = self.l1(z).reshape((-1, self.bottom_width ** 2, self.embed_dim))
        x = x + self.pos_embed[0]
        H, W = self.bottom_width, self.bottom_width
        x = self.blocks(x)
        for index, blk in enumerate(self.upsample_blocks):
            x, H, W = pixel_upsample(x, H, W)
            x = x + self.pos_embed[index+1]
            x = blk(x)
        output = self.deconv(x.transpose((0, 2, 1)).reshape((-1, self.embed_dim//16, H, W)))
        return output
