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
Implement CaiT Transformer
"""

import copy
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

    Apply patch embeddings on input images. Embeddings is implemented using a Conv2D op.

    Attributes:
        image_size: int, input image size, default: 224
        patch_size: int, size of patch, default: 4
        in_channels: int, input image channels, default: 3
        embed_dim: int, embedding dimension, default: 96
    """

    def __init__(self, image_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
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
        # CaiT norm is not included
        #self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x) # [batch, embed_dim, h, w] h,w = patch_resolution
        x = x.flatten(start_axis=2, stop_axis=-1) # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1]) # [batch, h*w, embed_dim]
        #x = self.norm(x) # [batch, num_patches, embed_dim] # CaiT norm is not needed
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


class ClassAttention(nn.Layer):
    """ Class Attention

    Class Attention module

    Args:
        dim: int, all heads dimension
        dim_head: int, single heads dimension, default: None
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
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.q = nn.Linear(dim, dim, weight_attr=w_attr_1, bias_attr=b_attr_1, if qkv_bias else False)
        w_attr_2, b_attr_2 = self._init_weights()
        self.k = nn.Linear(dim, dim, weight_attr=w_attr_2, bias_attr=b_attr_2, if qkv_bias else False)
        w_attr_3, b_attr_3 = self._init_weights()
        self.v = nn.Linear(dim, dim, weight_attr=w_attr_3, bias_attr=b_attr_3, if qkv_bias else False)

        self.attn_dropout = nn.Dropout(attention_dropout)
        w_attr_4, b_attr_4 = self._init_weights()
        self.proj = nn.Linear(dim, dim, weight_attr=w_attr_4, bias_attr=b_attr_4)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x[:, :1, :]) # same as x[:, 0], but more intuitive
        q = q.reshape([B, self.num_heads, 1, self.dim_head])

        k = self.k(x)
        k = k.reshape([B, N, self.num_heads, self.dim_head])
        k = k.transpose([0, 2, 1, 3])

        v = self.v(x)
        v = v.reshape([B, N, self.num_heads, self.dim_head])
        v = v.transpose([0, 2, 1, 3])

        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        cls_embed = paddle.matmul(attn, v)
        cls_embed = cls_embed.transpose([0, 2, 1, 3])
        cls_embed = cls_embed.reshape([B, 1, C])
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_dropout(cls_embed)
        return cls_embed


class TalkingHeadAttention(nn.Layer):
    """ Talking head attention

    Talking head attention (https://arxiv.org/abs/2003.02436),
    applies linear projections across the attention-heads dimension,
    before and after the softmax operation.

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim, dim * 3, weight_attr=w_attr_1, bias_attr=b_attr_1 if qkv_bias else False)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim, dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.proj_dropout = nn.Dropout(dropout)

        # talking head
        w_attr_3, b_attr_3 = self._init_weights()
        self.proj_l = nn.Linear(num_heads, num_heads, weight_attr=w_attr_3, bias_attr=b_attr_3)
        w_attr_4, b_attr_4 = self._init_weights()
        self.proj_w = nn.Linear(num_heads, num_heads, weight_attr=w_attr_4, bias_attr=b_attr_4)

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
        B, H, C = x.shape # H: num_patches
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv) #[B, num_heads, num_patches, single_head_dim]

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True) #[B, num_heads, num_patches, num_patches]

        # projection across heads (before softmax)
        attn = attn.transpose([0, 2, 3, 1]) #[B, num_patches, num_patches, num_heads]
        attn = self.proj_l(attn)
        attn = attn.transpose([0, 3, 1, 2]) #[B, num_heads, num_patches, num_patches]

        attn = self.softmax(attn)

        # projection across heads (after softmax)
        attn = attn.transpose([0, 2, 3, 1]) #[B, num_patches, num_patches, num_heads]
        attn = self.proj_w(attn)
        attn = attn.transpose([0, 3, 1, 2]) #[B, num_heads, num_patches, num_patches]

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) #[B, num_heads, num_patches, single_head_dim]
        z = z.transpose([0, 2, 1, 3]) #[B, num_patches, num_heads, single_head_dim]

        z = z.reshape([B, H, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class LayerScaleBlockClassAttention(nn.Layer):
    """ LayerScale layers for class attention

    LayerScale layers for class attention contains regular class-attention layers,
    in addition with gamma_1 and gamma_2, which apply per-channel multiplication
    after each residual block (attention and mlp layers).

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        init_values: initial values for learnable weights gamma_1 and gamma_2, default: 1e-4
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)
        self.attn = ClassAttention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   dropout=dropout,
                                   attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights()
        self.norm2 = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, x_cls):
        u = paddle.concat([x_cls, x], axis=1)

        u = self.norm1(u)
        u = self.attn(u)
        u = self.gamma_1 * u
        u = self.drop_path(u)
        x_cls = u + x_cls

        h = x_cls
        x_cls = self.norm2(x_cls)
        x_cls = self.mlp(x_cls)
        x_cls = self.gamma_2 * x_cls
        x_cls = self.drop_path(x_cls)
        x_cls = h + x_cls

        return x_cls


class LayerScaleBlock(nn.Layer):
    """ LayerScale layers

    LayerScale layers contains regular self-attention layers,
    in addition with gamma_1 and gamma_2, which apply per-channel multiplication
    after each residual block (attention and mlp layers).

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        init_values: initial values for learnable weights gamma_1 and gamma_2, default: 1e-4
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)
        self.attn = TalkingHeadAttention(dim,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights()
        self.norm2 = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.gamma_1 * x #[B, num_patches, embed_dim]
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2 * x #[B, num_patches, embed_dim]
        x = self.drop_path(x)
        x = h + x
        return x


class Cait(nn.Layer):
    """ CaiT model
    Args:
        image_size: int, input image size, default: 224
        in_channels: int, input image channels, default: 3
        num_classes: int, num of classes, default: 1000
        patch_size: int, patch size for patch embedding, default: 16
        embed_dim: int, dim of each patch after patch embedding, default: 768
        depth: int, num of self-attention blocks, default: 12
        num_heads: int, num of attention heads, default: 12
        mlp_ratio: float, mlp hidden dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if True, qkv projection is set with bias, default: True
        dropout: float, dropout rate for linear projections, default: 0.
        attention_dropout: float, dropout rate for attention, default: 0.
        droppath: float, drop path rate, default: 0.
        init_values: initial value for layer scales, default: 1e-4
        mlp_ratio_class_token: float, mlp_ratio for mlp used in class attention blocks, default: 4.0
        depth_token_only, int, num of class attention blocks, default: 2
    """
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 num_classes=1000,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 init_values=1e-4,
                 mlp_ratio_class_token=4.0,
                 depth_token_only=2):
        super().__init__()
        self.num_classes = num_classes
        # convert image to paches
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))

        self.pos_dropout = nn.Dropout(dropout)

        # create self-attention(layer-scale) layers
        layer_list = []
        for i in range(depth):
            block_layers = LayerScaleBlock(dim=embed_dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           dropout=dropout,
                                           attention_dropout=attention_dropout,
                                           droppath=droppath,
                                           init_values=init_values)
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks = nn.LayerList(layer_list)

        # craete class-attention layers
        layer_list = []
        for i in range(depth_token_only):
            block_layers = LayerScaleBlockClassAttention(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_class_token,
                qkv_bias=qkv_bias,
                dropout=0.,
                attention_dropout=0.,
                droppath=0.,
                init_values=init_values)
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks_token_only = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights_norm()
        self.norm = nn.LayerNorm(embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)
        w_attr_2, b_attr_2 = self._init_weights_linear()
        self.head = nn.Linear(embed_dim,
                              num_classes,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2) if num_classes > 0 else Identity()

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1]) # [B, 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        # Self-Attention blocks
        for idx, block in enumerate(self.blocks):
            x = block(x) # [B, num_patches, embed_dim]
        # Class-Attention blocks
        for idx, block in enumerate(self.blocks_token_only):
            cls_tokens = block(x, cls_tokens) # [B, 1, embed_dim]
        # Concat outputs
        x = paddle.concat([cls_tokens, x], axis=1)
        x = self.norm(x) # [B, num_patches + 1, embed_dim]
        return x[:, 0] # returns only cls_tokens

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_cait(config):
    """build cait model using config"""
    model = Cait(image_size=config.DATA.IMAGE_SIZE,
                 num_classes=config.MODEL.NUM_CLASSES,
                 in_channels=config.MODEL.TRANS.IN_CHANNELS,
                 patch_size=config.MODEL.TRANS.PATCH_SIZE,
                 embed_dim=config.MODEL.TRANS.EMBED_DIM,
                 depth=config.MODEL.TRANS.DEPTH,
                 num_heads=config.MODEL.TRANS.NUM_HEADS,
                 mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                 qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                 dropout=config.MODEL.DROPOUT,
                 attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                 droppath=config.MODEL.DROPPATH,
                 init_values=config.MODEL.TRANS.INIT_VALUES,
                 mlp_ratio_class_token=config.MODEL.TRANS.MLP_RATIO,
                 depth_token_only=config.MODEL.TRANS.DEPTH_TOKEN_ONLY):
    return model
