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
LeViT in Paddle

A Paddle Implementation of LeViT as described in:

"LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference"
    - Paper Link: https://arxiv.org/pdf/2104.01136v2.pdf
"""

from functools import partial 
import paddle
import paddle.nn as nn
from droppath import DropPath


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """
    def forward(self, inputs):
        return inputs


class PatchEmbed(nn.Sequential):
    def __init__(self, in_channels, out_channels, act=nn.Hardswish, resolution=224):
        super().__init__(
            ConvNorm(in_channels, out_channels // 8, kernel_size=3, stride=2, padding=1),
            act(),
            ConvNorm(out_channels // 8, out_channels // 4, kernel_size=3, stride=2, padding=1),
            act(),
            ConvNorm(out_channels // 4, out_channels // 2, kernel_size=3, stride=2, padding=1),
            act(),
            ConvNorm(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
        )

        
class ConvNorm(nn.Layer):
    """Layer ops: Conv2D -> BatchNorm2D"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 norm=nn.BatchNorm2D):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
                              bias_attr=False if bias_attr is False else paddle.ParamAttr(initializer=nn.initializer.Constant(0.)))
        self.norm = Identity() if norm is None else norm(out_channels)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class LinearNorm(nn.Layer):
    """Layer ops: Linear -> BatchNorm1D"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias_attr=False,
                 norm=nn.BatchNorm1D):
        super().__init__()
        self.linear = nn.Linear(in_channels,
                                out_channels,
                                weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
                                bias_attr=bias_attr)
        self.norm = Identity() if norm is None else norm(out_channels)

    def forward(self, inputs):
        out = self.linear(inputs)
        s = out.shape
        out = out.flatten(0, 1)
        out = self.norm(out)
        out = out.reshape(s)
        return out


class NormLinear(nn.Layer):
    """Layer ops: BatchNorm1D -> Linear"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias_attr=False,
                 norm=nn.BatchNorm1D):
        super().__init__()
        self.norm = Identity() if norm is None else norm(in_channels)
        self.linear = nn.Linear(in_channels,
                                out_channels,
                                weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
                                bias_attr= paddle.ParamAttr(initializer=nn.initializer.Constant(0.)) if bias_attr else False)

    def forward(self, inputs):
        out = self.norm(inputs)
        out = self.linear(out)
        return out


class Subsample(nn.Layer):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        N, L, C = x.shape
        x = x.reshape([N, self.resolution, self.resolution, C])
        x = x[:, ::self.stride, ::self.stride]
        return x.reshape([N, -1, C])


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0., use_conv=False):
        super().__init__()
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.ln1 = ln_layer(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.Hardswish()
        self.ln2 = ln_layer(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=14, 
                 resolution_out=14,
                 use_conv=False):
        super().__init__()
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.resolution = resolution
        self.resolution_out = resolution_out
        self.resolution_out_area = resolution_out ** 2
        self.use_conv = use_conv
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        
        self.qkv = ln_layer(embed_dim, self.val_attn_dim + self.key_attn_dim * 2)
        self.proj = nn.Sequential(
            nn.Hardswish(),
            ln_layer(self.val_attn_dim, embed_dim))

        self.attn_biases = paddle.create_parameter(
            shape=[num_heads, resolution ** 2], dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.))

        pos = paddle.stack(paddle.meshgrid(paddle.arange(resolution), paddle.arange(resolution))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        self.register_buffer('attn_bias_idxs', rel_pos)

        self.softmax = nn.Softmax(-1)

    def get_attn_biases(self):
        res = paddle.zeros([self.num_heads] + self.attn_bias_idxs.shape)
        for idx in range(self.resolution_out_area):
            res[:, idx, :] = paddle.gather(self.attn_biases, self.attn_bias_idxs[idx, :], axis=1)
        return res

    def forward(self, x):
        if self.use_conv:
            N, C, H, W = x.shape
            qkv = self.qkv(x)
            qkv = qkv.reshape([N, self.num_heads, -1, H * W])
            q, k, v  = qkv.split([self.key_dim, self.key_dim, self.val_dim], axis=2)

            q = q * self.scale
            attn = paddle.matmul(q, k, transpose_y=True)
            attn = attn + self.get_attn_biases()
            attn  = self.softmax(attn)

            z = paddle.matmul(attn, v)
            z = z.transpose([0, 1, 3, 2])
            z = z.reshape([B, -1, self.resolution, self.resolution])
            z = self.proj(z)
        else:
            N, L, C = x.shape
            qkv = self.qkv(x)
            qkv = qkv.reshape([N, L, self.num_heads, -1])
            qkv = qkv.transpose([0, 2, 1, 3])
            q, k, v  = qkv.split([self.key_dim, self.key_dim, self.val_dim], axis=3)

            q = q * self.scale
            attn = paddle.matmul(q, k, transpose_y=True)

            attn = attn + self.get_attn_biases()
            attn  = self.softmax(attn)

            z = paddle.matmul(attn, v)

            z = z.transpose([0, 2, 1, 3])
            z = z.reshape([N, -1, self.val_attn_dim])
            z = self.proj(z)
        return z


class AttentionSubsample(nn.Layer):
    def __init__(self,
                 embed_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=14, 
                 resolution_out=7,
                 use_conv=False):
        super().__init__()
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.resolution = resolution
        self.resolution_out = resolution_out
        self.resolution_out_area = resolution_out ** 2
        stride = 2  # only for subsample
        if use_conv:
            sub_layer = partial(nn.AvgPool2D, kernel_size=1, padding=0)
        else:
            sub_layer = partial(Subsample, resolution=resolution)

        self.use_conv = use_conv
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        
        self.q = nn.Sequential(sub_layer(stride=stride), ln_layer(embed_dim, self.key_attn_dim))
        self.kv = ln_layer(embed_dim, self.val_attn_dim + self.key_attn_dim)
        self.proj = nn.Sequential(
            nn.Hardswish(),
            ln_layer(self.val_attn_dim, out_dim))

        self.attn_biases = paddle.create_parameter(
            shape=[num_heads, resolution ** 2], dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.))

        k_pos = paddle.stack(paddle.meshgrid(paddle.arange(resolution), paddle.arange(resolution))).flatten(1)
        q_pos = paddle.stack(paddle.meshgrid(
            paddle.arange(0, resolution, step=stride),
            paddle.arange(0, resolution, step=stride))).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        self.register_buffer('attn_bias_idxs', rel_pos)

        self.softmax = nn.Softmax(-1)

    def get_attn_biases(self):
        res = paddle.zeros([self.num_heads] + self.attn_bias_idxs.shape)
        for idx in range(self.resolution_out_area):
            res[:, idx, :] = paddle.gather(self.attn_biases, self.attn_bias_idxs[idx, :], axis=1)
        return res

    def forward(self, x):
        if self.use_conv:
            N, C, H, W = x.shape
            q = self.q(x)
            q = q.reshape([N, self.num_heads, self.key_dim, self.resolution_out_area])

            kv = self.kv(x)
            kv = kv.reshape([N, self.num_heads, -1, H * W])
            k, v  = kv.split([self.key_dim, self.val_dim], axis=2)

            q = q * self.scale
            attn = paddle.matmul(q, k, transpose_y=True)
            attn = attn + self.get_attn_biases()
            attn  = self.softmax(attn)

            z = paddle.matmul(attn, v)
            z = z.transpose([0, 1, 3, 2])
            z = z.reshape([B, -1, self.resolution, self.resolution])
            z = self.proj(z)
        else:
            N, L, C = x.shape
            q = self.q(x)
            q = q.reshape([N, self.resolution_out_area, self.num_heads, self.key_dim])
            q = q.transpose([0, 2, 1, 3])

            kv = self.kv(x)
            kv = kv.reshape([N, L, self.num_heads, -1])
            kv = kv.transpose([0, 2, 1, 3])
            k, v  = kv.split([self.key_dim, self.val_dim], axis=3)

            q = q * self.scale
            attn = paddle.matmul(q, k, transpose_y=True)
            attn = attn + self.get_attn_biases()
            attn  = self.softmax(attn)

            z = paddle.matmul(attn, v)
            z = z.transpose([0, 2, 1, 3])
            z = z.reshape([N, -1, self.val_attn_dim])
            z = self.proj(z)
        return z


class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 mlp_ratio=2,
                 attn_ratio=2,
                 resolution=14,
                 dropout=0.,
                 droppath=0.,
                 use_conv=False,
                 subsample=False):
        super().__init__()
        self.subsample = subsample
        self.ln_layer = ConvNorm if use_conv else LinearNorm
        resolution_out = (resolution - 1) // 2 + 1 if subsample else resolution  # stride = 2
        attn = AttentionSubsample if subsample else Attention
        self.attn = attn(embed_dim=embed_dim,
                         out_dim=out_dim,
                         key_dim=key_dim,
                         num_heads=num_heads,
                         attn_ratio=attn_ratio,
                         resolution=resolution,
                         resolution_out=resolution_out,
                         use_conv=use_conv)
        self.mlp = Mlp(out_dim, mlp_ratio, dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()

    def forward(self, x):
        h = x
        x = self.attn(x)
        x = self.drop_path(x)
        if not self.subsample:   # subsample does not have residual pass
            x = h + x

        h = x
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h

        return x


class LeViT(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=(192,),
                 key_dim=64,
                 depth=(12, ),
                 num_heads=(3,),
                 attn_ratio=2,
                 mlp_ratio=2,
                 use_conv=False,
                 dropout=0.,
                 droppath=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.use_conv = use_conv
        resolution = image_size // patch_size
        self.patch_embed = PatchEmbed(in_channels, embed_dim[0], resolution=resolution)
        layer_list = []
        for stage_idx, stage_depth in enumerate(depth):
            block_embed_dim = embed_dim[stage_idx]
            block_num_heads = num_heads[stage_idx]
            for block_idx in range(stage_depth):
                layer_list.append(
                    EncoderLayer(embed_dim=block_embed_dim,
                                 out_dim=block_embed_dim,
                                 key_dim=key_dim,
                                 num_heads=block_num_heads,
                                 mlp_ratio=mlp_ratio,
                                 attn_ratio=attn_ratio,
                                 resolution=resolution,
                                 dropout=dropout,
                                 droppath=droppath,
                                 use_conv=use_conv,
                                 subsample=False))
            if stage_idx != len(depth) - 1: # last stage does not have subsample attn layers
                layer_list.append(
                    EncoderLayer(embed_dim=block_embed_dim,
                                 out_dim=embed_dim[stage_idx + 1],  # set for subsample
                                 key_dim=key_dim,
                                 num_heads=block_embed_dim // key_dim,  # set for subsample
                                 mlp_ratio=mlp_ratio,
                                 attn_ratio=4,  # set for subsample
                                 resolution=resolution,
                                 dropout=dropout,
                                 droppath=droppath,
                                 use_conv=use_conv,
                                 subsample=True))
                resolution = (resolution - 1) // 2 + 1 

        self.blocks = nn.LayerList(layer_list)
        self.head = NormLinear(embed_dim[-1], num_classes, bias_attr=True) if num_classes > 0 else Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if not self.use_conv:
            x = x.flatten(2).transpose([0, 2, 1])
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_conv:
            x = x.mean((-2, -1))
        else:
            x = x.mean(1)
        x = self.head(x)
        return x

    
class LeViTDistilled(LeViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dist = NormLinear(self.num_features, self.num_classes, bias_attr=True) if self.num_classes > 0 else Identity()
        self.distilled_training = False

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_conv:
            x = x.mean((-2, -1))
        else:
            x = x.mean(1)
        x, x_dist = self.head(x), self.head_dist(x)

        if not self.training and not self.distilled_training:
            return (x + x_dist) / 2
        else:
            return x, x_dist


def build_levit(config):
    """Build LeViT by reading options in config object
    Args:
        config: config instance contains setting options
    Returns:
        model: nn.Layer, TopFormer model
    """
    if config.TRAIN.DISTILLATION_TYPE == 'none':
        levit = LeViT
    else:
        levit = LeViTDistilled

    model = levit(image_size=config.DATA.IMAGE_SIZE,
                  patch_size=config.MODEL.PATCH_SIZE,
                  in_channels=config.DATA.IMAGE_CHANNELS,
                  num_classes=config.MODEL.NUM_CLASSES,
                  embed_dim=config.MODEL.EMBED_DIM,
                  key_dim=config.MODEL.KEY_DIM,
                  depth=config.MODEL.DEPTH,
                  num_heads=config.MODEL.NUM_HEADS,
                  attn_ratio=config.MODEL.ATTN_RATIO,
                  mlp_ratio=config.MODEL.MLP_RATIO,
                  droppath=config.MODEL.DROPPATH,
                  dropout=config.MODEL.DROPOUT,
                  use_conv=config.MODEL.USE_CONV)
    return model

        


