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
XCiT in Paddle
A Paddle Impelementation of XCiT as described in:
"Cross-Covariance Image Transformer"
    - Paper Link: https://arxiv.org/pdf/2106.09681.pdf
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


class Mlp(nn.Layer):
    """MLP module
    MLP using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc1 -> act -> dropout -> fc2 -> dropout
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


class Identity(nn.Layer):
    """Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class PositionalEncodingFourier(nn.Layer):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper.
    """
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2D(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B, H, W):
        mask = paddle.zeros([B, H, W]).astype("bool")
        not_mask = paddle.logical_not(mask)
        y_embed = not_mask.cumsum(1, dtype="float32")
        x_embed = not_mask.cumsum(2, dtype="float32")
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = paddle.arange(self.hidden_dim, dtype="int64")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed.unsqueeze(3) / dim_t
        pos_y = y_embed.unsqueeze(3) / dim_t
        pos_x = paddle.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        pos = self.token_projection(pos)
        return pos


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return paddle.nn.Sequential(
        nn.Conv2D(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias_attr=False),
        nn.BatchNorm2D(out_planes))


class ConvPatchEmbed(nn.Layer):
    """ Image to Patch Embedding using multiple convolutional layers
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = paddle.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = paddle.nn.Sequential(
                conv3x3(3, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise ValueError("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose([0, 2, 1])

        return x, (Hp, Wp)


class LPI(nn.Layer):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = paddle.nn.Conv2D(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )
        self.act = act_layer()
        self.bn = nn.BatchNorm2D(in_features)
        self.conv2 = paddle.nn.Conv2D(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape([B, C, N]).transpose([0, 2, 1])

        return x

#class ClassAttention(nn.Layer):
#    """ Class Attention (timm version)
#
#    Class Attention module
#
#    Args:
#        dim: int, all heads dimension
#        dim_head: int, single heads dimension, default: None
#        num_heads: int, num of heads
#        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
#        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
#        attention_dropout: float, dropout rate for attention dropout, default: 0.
#        dropout: float, dropout rate for projection dropout, default: 0.
#    """
#
#    def __init__(self,
#                 dim,
#                 num_heads=8,
#                 qkv_bias=False,
#                 qk_scale=None,
#                 attention_dropout=0.,
#                 dropout=0.):
#        super().__init__()
#        self.num_heads = num_heads
#        self.dim_head = dim // num_heads
#        self.scale = qk_scale or self.dim_head ** -0.5
#
#        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
#        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
#        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
#
#        self.attn_dropout = nn.Dropout(attention_dropout)
#        self.proj = nn.Linear(dim, dim)
#        self.proj_dropout = nn.Dropout(dropout)
#        self.softmax = nn.Softmax(axis=-1)
#
#    def forward(self, x):
#        B, N, C = x.shape
#
#        q = self.q(x[:, :1, :]) # same as x[:, 0], but more intuitive
#        q = q.reshape([B, self.num_heads, 1, self.dim_head])
#
#        k = self.k(x)
#        k = k.reshape([B, N, self.num_heads, self.dim_head])
#        k = k.transpose([0, 2, 1, 3])
#
#        v = self.v(x)
#        v = v.reshape([B, N, self.num_heads, self.dim_head])
#        v = v.transpose([0, 2, 1, 3])
#
#        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
#        attn = self.softmax(attn)
#        attn = self.attn_dropout(attn)
#
#        cls_embed = paddle.matmul(attn, v)
#        cls_embed = cls_embed.transpose([0, 2, 1, 3])
#        cls_embed = cls_embed.reshape([B, 1, C])
#        cls_embed = self.proj(cls_embed)
#        cls_embed = self.proj_dropout(cls_embed)
#        return cls_embed

class ClassAttention(nn.Layer):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qc = q[:, :, 0:1]  # CLS token
        attn_cls = (qc * k).sum(axis=-1) * self.scale
        attn_cls = F.softmax(attn_cls, axis=-1)
        attn_cls = self.attn_drop(attn_cls)

        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose([0, 1, 3, 2]).reshape([B, 1, C])
        cls_tkn = self.proj(cls_tkn)
        x = paddle.concat([self.proj_drop(cls_tkn), x[:, 1:]], axis=1)
        return x


class ClassAttentionBlock(nn.Layer):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eta=None,
                 tokens_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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

        # LayerScale Initialization (no layerscale when None)
        if eta is not None:
            self.gamma1 = paddle.create_parameter(
                shape=[dim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=eta),
            )
            self.gamma2 = paddle.create_parameter(
                shape=[dim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=eta),
            )
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        ##timm version
        #x_norm1 = self.norm1(x)
        #x_attn = paddle.concat([self.attn(x_norm1), x_norm1[:, 1:]], axis=1)
        #x = x + self.drop_path(self.gamma1 * x_attn)

        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1] = self.norm2(x[:, 0:1])

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = paddle.concat([cls_token, x[:, 1:]], axis=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Layer):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = paddle.create_parameter(
            shape=[num_heads, 1, 1], dtype="float32", default_initializer=ones_
        )

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose([0, 1, 3, 2])
        k = k.transpose([0, 1, 3, 2])
        v = v.transpose([0, 1, 3, 2])

        q = paddle.nn.functional.normalize(q, axis=-1)
        k = paddle.nn.functional.normalize(k, axis=-1)

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.temperature
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 3, 1, 2]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCABlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_tokens=196,
                 eta=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.gamma1 = paddle.create_parameter(
            shape=[dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=eta),
        )
        self.gamma2 = paddle.create_parameter(
            shape=[dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=eta),
        )
        self.gamma3 = paddle.create_parameter(
            shape=[dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=eta),
        )

        # self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        # self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        # self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Layer):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 cls_attn_layers=2,
                 use_pos=True,
                 patch_proj="linear",
                 eta=None,
                 tokens_norm=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilson=1e-6)

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        )

        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim], dtype="float32", default_initializer=trunc_normal_
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.LayerList(
            [
                XCABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_tokens=num_patches,
                    eta=eta,
                )
                for i in range(depth)
            ]
        )

        self.cls_attn_blocks = nn.LayerList(
            [
                ClassAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    eta=eta,
                    tokens_norm=tokens_norm,
                )
                for i in range(cls_attn_layers)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
        self.use_pos = use_pos
        self.head = (
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else Identity()
        )

        # Classifier head
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]

        x, (Hp, Wp) = self.patch_embed(x)


        if self.use_pos:
            pos_encoding = (
                self.pos_embeder(B, Hp, Wp)
                .reshape([B, -1, x.shape[1]])
                .transpose([0, 2, 1])
            )
            x = x + pos_encoding

        x = self.pos_drop(x)


        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat((cls_tokens, x), axis=1)

        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        if self.train:
            return x, x

        return x


def build_xcit(config):
    model = XCiT(
        img_size=config.DATA.IMAGE_SIZE,
        patch_size=config.MODEL.PATCH_SIZE,
        embed_dim=config.MODEL.EMBED_DIM,
        num_classes=config.MODEL.NUM_CLASSES,
        depth=config.MODEL.DEPTH,
        num_heads=config.MODEL.NUM_HEADS,
        eta=config.MODEL.ETA,
        tokens_norm=config.MODEL.TOKENS_NORM,
        in_chans=config.DATA.IMAGE_CHANNELS,
        mlp_ratio=config.MODEL.MLP_RATIO,
        qkv_bias=config.MODEL.QKV_BIAS,
        qk_scale=config.MODEL.QK_SCALE,
        drop_rate=config.MODEL.DROPOUT,
        attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
        drop_path_rate=config.MODEL.DROPPATH,
        cls_attn_layers=config.MODEL.CLS_ATTN_LAYERS,
        use_pos=config.MODEL.USE_POS,
    )
    return model
