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
Implement T2T-ViT Transformer
"""

import copy
import math
import numpy as np
import paddle
import paddle.nn as nn
from droppath import DropPath
from utils import orthogonal


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

    Apply patch embeddings (tokens-to-token) on input images. Embeddings is
    implemented using one of the following ops: Performer, Transformer.

    Attributes:
        image_size: int, input image size, default: 224
        token_type: string, type of token embedding, in ['performer', 'transformer', 'convolution'], default: 'performer'
        patch_size: int, size of patch, default: 4
        in_channels: int, input image channels, default: 3
        embed_dim: int, embedding dimension, default: 96
        token_dim: int, intermediate dim for patch_embedding module, default: 64
    """
    def __init__(self,
                 image_size=224,
                 token_type='performer',
                 in_channels=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()
        if token_type == 'transformer':
            # paddle v 2.1 has bugs on nn.Unfold,
            # use paddle.nn.functional.unfold method instead
            # replacements see forward method.
            #self.soft_split0 = nn.Unfold(kernel_size=7, strides=4, paddings=2)
            #self.soft_split1 = nn.Unfold(kernel_size=3, strides=2, paddings=1)
            #self.soft_split2 = nn.Unfold(kernel_size=3, strides=2, paddings=1)

            self.attn1 = TokenTransformer(dim=in_channels * 7 * 7,
                                          in_dim=token_dim,
                                          num_heads=1,
                                          mlp_ratio=1.0)
            self.attn2 = TokenTransformer(dim=token_dim * 3 * 3,
                                          in_dim=token_dim,
                                          num_heads=1,
                                          mlp_ratio=1.0)

            w_attr_1, b_attr_1 = self._init_weights() # init for linear
            self.proj = nn.Linear(token_dim * 3 * 3,
                                  embed_dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1)

        elif token_type == 'performer':
            # paddle v 2.1 has bugs on nn.Unfold,
            # use paddle.nn.functional.unfold method instead
            # replacements see forward method.
            #self.soft_split0 = nn.Unfold(kernel_sizes=7, strides=4, paddings=2)
            #self.soft_split1 = nn.Unfold(kernel_sizes=3, strides=2, paddings=1)
            #self.soft_split2 = nn.Unfold(kernel_sizes=3, strides=2, paddings=1)

            self.attn1 = TokenPerformer(dim=in_channels * 7 * 7,
                                        in_dim=token_dim,
                                        kernel_ratio=0.5)
            self.attn2 = TokenPerformer(dim=token_dim * 3 * 3,
                                        in_dim=token_dim,
                                        kernel_ratio=0.5)

            w_attr_1, b_attr_1 = self._init_weights() # init for linear
            self.proj = nn.Linear(token_dim * 3 * 3,
                                  embed_dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1)

        elif token_type == 'convolution': # NOTE: currently not supported!!!
            # 1st conv
            self.soft_split0 = nn.Conv2D(in_channels=in_channels,
                                         out_channels=token_dim,
                                         kernel_size=7,
                                         stride=4,
                                         padding=2)
            # 2nd conv
            self.soft_split1 = nn.Conv2D(in_channels=token_dim,
                                         out_channels=token_dim,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
            # 3rd conv
            self.proj = nn.Conv2D(in_channels=token_dim,
                                  out_channels=embed_dim,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        else:
            raise ValueError(f'token_type: {token_type} is not supported!')

        # 3 soft splits, each has stride 4, 2, 2, respectively.
        self.num_patches = (image_size // (4 * 2 * 2)) * (image_size // (4 * 2 * 2))

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        # x = self.soft_split0(x)
        # input x: [B, C, IMAGE_H, IMAGE_W]
        x = paddle.nn.functional.unfold(x, kernel_sizes=7, strides=4, paddings=2)
        # unfolded x: [B, C * k * k, k * k * num_patches]
        x = x.transpose([0, 2, 1])
        # transposed x: [B, k * k * num_patches, C * k * k]

        x = self.attn1(x)
        B, HW, C = x.shape
        x = x.transpose([0, 2, 1])
        x = x.reshape([B, C, int(np.sqrt(HW)), int(np.sqrt(HW))])
        #x = self.soft_split1(x)
        x = paddle.nn.functional.unfold(x, kernel_sizes=3, strides=2, paddings=1)
        x = x.transpose([0, 2, 1])

        x = self.attn2(x)
        B, HW, C = x.shape
        x = x.transpose([0, 2, 1])
        x = x.reshape([B, C, int(np.sqrt(HW)), int(np.sqrt(HW))])
        #x = self.soft_split2(x)
        x = paddle.nn.functional.unfold(x, kernel_sizes=3, strides=2, paddings=1)
        x = x.transpose([0, 2, 1])

        x = self.proj(x)
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

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             out_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """ Self-Attention

    Args:
        dim: int, all heads dimension
        dim_head: int, single heads dimension, default: None
        num_heads: int, num of heads
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
        skip_connection: bool, if Ture, use v to do skip connection, used in TokenTransformer
    """
    def __init__(self,
                 dim,
                 in_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.,
                 skip_connection=False):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim or dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5
        # same as original repo
        w_attr_1, b_attr_1 = self._init_weights() # init for linear
        self.qkv = nn.Linear(dim,
                             self.in_dim * 3,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.attn_dropout = nn.Dropout(attention_dropout)
        w_attr_2, b_attr_2 = self._init_weights() # init for linear
        self.proj = nn.Linear(self.in_dim,
                              self.in_dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

        # use V to do skip connection, used in TokenTransformer
        self.skip = skip_connection

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        if self.skip: # token transformer
            new_shape = x.shape[:-1] + [self.num_heads, self.in_dim]
        else: # regular attention
            new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        B, H, C = x.shape
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        if self.skip: # token transformer
            z = z.reshape([B, -1, self.in_dim])
        else: # regular attention
            z = z.reshape([B, -1, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        # skip connection
        if self.skip:
            z = z + v.squeeze(1)

        return z


class Block(nn.Layer):
    """ Transformer block layers

    Transformer block layers contains regular self-attention layers,
    mlp layers, norms layers and residual blocks.

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, scale factor to replace dim_head ** -0.5, default: None
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        droppath: float, drop path rate, default: 0.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_layernorm() # init for layernorm
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              dropout=dropout,
                              attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights_layernorm() # init for layernorm
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
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


class TokenPerformer(nn.Layer):
    """ Token Performer layers

    Performer layers contains single-attention layers,
    mlp layers, norms layers and residual blocks. This module
    is used in 'tokens-to-token', which converts image into tokens
    and gradually tokenized the tokens.

    Args:
        dim: int, all heads dimension
        in_dim: int, qkv and out dimension in attention
        num_heads: int, num of heads
        kernel_ratio: ratio to multiply on prm input dim, default: 0.5.
        dropout: float, dropout rate for projection dropout, default: 0.
    """
    def __init__(self, dim, in_dim, num_heads=1, kernel_ratio=0.5, dropout=0.1):
        super().__init__()
        self.embed_dim = in_dim * num_heads
        w_attr_1, b_attr_1 = self._init_weights() # init for linear
        self.kqv = nn.Linear(dim, 3 * self.embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.dropout = nn.Dropout(dropout)
        w_attr_2, b_attr_2 = self._init_weights() # init for linear
        self.proj = nn.Linear(self.embed_dim,
                              self.embed_dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.num_heads = num_heads
        w_attr_3, b_attr_3 = self._init_weights_layernorm() # init for layernorm
        w_attr_4, b_attr_4 = self._init_weights_layernorm() # init for layernorm
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=w_attr_3, bias_attr=b_attr_3)
        self.norm2 = nn.LayerNorm(self.embed_dim, epsilon=1e-6, weight_attr=w_attr_4, bias_attr=b_attr_4)

        w_attr_5, b_attr_5 = self._init_weights() # init for linear
        w_attr_6, b_attr_6 = self._init_weights() # init for linear
        self.mlp = nn.Sequential(nn.Linear(self.embed_dim,
                                           self.embed_dim,
                                           weight_attr=w_attr_5,
                                           bias_attr=b_attr_5),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim,
                                           self.embed_dim,
                                           weight_attr=w_attr_6,
                                           bias_attr=b_attr_6),
                                 nn.Dropout(dropout))

        self.m = int(self.embed_dim  * kernel_ratio)

        self.w = np.random.random(size=(int(self.embed_dim * kernel_ratio), self.embed_dim))
        # init with orthognal matrix
        self.w = orthogonal(self.w)

        self.w = paddle.create_parameter(
            shape=[int(self.embed_dim * kernel_ratio), self.embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Assign(self.w / math.sqrt(self.m)))

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    # paddle version 2.1 does not support einsum
    def prm_exp(self, x):
        # x: [B, T, hs]
        # w: [m, hs]
        # return x: B, T, m
        xd = (x * x).sum(axis=-1, keepdim=True)
        xd = xd.expand([xd.shape[0], xd.shape[1], self.m]) / 2
        # same as einsum('bti,mi->btm', x, self.w)
        wtx = paddle.matmul(x, self.w, transpose_y=True)
        out = paddle.exp(wtx - xd) / math.sqrt(self.m)
        return out

    def single_attention(self, x):
        kqv = self.kqv(x).chunk(3, axis=-1)
        k, q, v = kqv[0], kqv[1], kqv[2]

        qp = self.prm_exp(q)
        kp = self.prm_exp(k)

        # same as einsum('bti,bi->bt, qp, kp.sum(axi=1).unsqueeze(2)')
        D = paddle.matmul(qp, kp.sum(axis=1).unsqueeze(2))
        # same as einsum('bti,bim->bnm')
        kptv = paddle.matmul(v, kp, transpose_x=True)
        # same as einsum('bti,bni->btn')
        y = paddle.matmul(qp, kptv, transpose_y=True)
        y = y / (D.expand([D.shape[0], D.shape[1], self.embed_dim]) + 1e-8)

        # skip connection
        y = self.proj(y)
        y = self.dropout(y)
        y = v + y
        return y

    def forward(self, x):
        x = self.norm1(x)
        x = self.single_attention(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + x
        return x


class TokenTransformer(nn.Layer):
    """ Token Transformer layers

    Transformer layers contains regular self-attention layers,
    mlp layers, norms layers and residual blocks. This module
    is used in 'tokens-to-token', which converts image into tokens
    and gradually tokenized the tokens.

    Args:
        dim: int, all heads dimension
        in_dim: int, qkv and out dimension in attention
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 1.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, scale factor to replace dim_head ** -0.5, default: None
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        droppath: float, drop path rate, default: 0.
    """
    def __init__(self,
                 dim,
                 in_dim,
                 num_heads,
                 mlp_ratio=1.0,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(dim,
                              in_dim=in_dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              dropout=dropout,
                              attention_dropout=attention_dropout,
                              skip_connection=True)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm2 = nn.LayerNorm(in_dim, epsilon=1e-6, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(in_features=in_dim,
                       hidden_features=int(in_dim * mlp_ratio),
                       out_features=in_dim,
                       dropout=dropout)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class T2TViT(nn.Layer):
    """ T2T-ViT model
    Args:
        image_size: int, input image size, default: 224
        in_channels: int, input image channels, default: 3
        num_classes: int, num of classes, default: 1000
        token_type: string, type of token embedding ['performer', 'transformer'], default: 'performer'
        embed_dim: int, dim of each patch after patch embedding, default: 768
        depth: int, num of self-attention blocks, default: 12
        num_heads: int, num of attention heads, default: 12
        mlp_ratio: float, mlp hidden dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if True, qkv projection is set with bias, default: True
        qk_scale: float, scale factor to replace dim_head ** -0.5, default: None
        dropout: float, dropout rate for linear projections, default: 0.
        attention_dropout: float, dropout rate for attention, default: 0.
        droppath: float, drop path rate, default: 0.
        token_dim: int, intermediate dim for patch_embedding module, default: 64
    """
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 num_classes=1000,
                 token_type='performer',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        # convert image to paches: T2T-Module
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          token_type=token_type,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim,
                                          token_dim=token_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        # dropout for positional embeddings
        self.pos_dropout = nn.Dropout(dropout)
        # droppath deacay rate
        depth_decay = paddle.linspace(0, droppath, depth)

        # craete self-attention layers
        layer_list = []
        for i in range(depth):
            block_layers = Block(dim=embed_dim,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 dropout=dropout,
                                 attention_dropout=attention_dropout,
                                 droppath=depth_decay[i])
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks = nn.LayerList(layer_list)
        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6, weight_attr=w_attr_1, bias_attr=b_attr_1)
        # classifier head
        w_attr_2, b_attr_2 = self._init_weights()
        self.head = nn.Linear(embed_dim,
                              num_classes,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2) if num_classes > 0 else Identity()

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Self-Attention blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0] # returns only cls_tokens

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_t2t_vit(config):
    """build t2t-vit model using config"""
    model = T2TViT(image_size=config.DATA.IMAGE_SIZE,
                   token_type=config.MODEL.TRANS.TOKEN_TYPE,
                   embed_dim=config.MODEL.TRANS.EMBED_DIM,
                   depth=config.MODEL.TRANS.DEPTH,
                   num_heads=config.MODEL.TRANS.NUM_HEADS,
                   mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                   qk_scale=config.MODEL.TRANS.QK_SCALE,
                   qkv_bias=config.MODEL.TRANS.QKV_BIAS)
    return model
