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
Implement VOLO Class
"""

import math
import copy
import numpy as np
import paddle
import paddle.nn as nn
from droppath import DropPath
from fold import fold
from utils import MyPrint

myprint = MyPrint()

class Identity(nn.Layer):
    """ Identity layer
    
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class Downsample(nn.Layer):
    """Apply a Conv2D with kernel size = patch_size and stride = patch_size
    The shape of input tensor is [N, H, W, C], which will be transposed to
    [N, C, H, W] and feed into Conv, finally the output is transposed back
    to [N, H, W, C].

    Args:
        in_embed_dim: int, input feature dimension
        out_embed_dim: int, output feature dimension
        patch_size: kernel_size and stride
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2D(in_embed_dim,
                              out_embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        x = self.proj(x)
        x = x.transpose([0, 2, 3, 1])
        return x


class PatchEmbedding(nn.Layer):
    """Patch Embeddings with stem conv layers

    If stem conv layers are set, the image is firstly feed into stem layers,
    stem layers contains 3 conv-bn-relu blocks.
    Then a proj (conv2d) layer is applied as the patch embedding.

    Args:
        image_size: int, input image size, default: 224
        stem_conv: bool, if apply stem conv layers, default: False
        stem_stride: int, conv stride in stem layers, default: 1
        patch_size: int, patch size for patch embedding (k and stride for proj conv), default: 8
        in_channels: int, input channels, default: 3
        hidden_dim: int, input dimension of patch embedding (out dim for stem), default: 64
        embed_dim: int, output dimension of patch embedding, default: 384

    """
    def __init__(self,
                 image_size=224,
                 stem_conv=False,
                 stem_stride=1,
                 patch_size=8,
                 in_channels=3,
                 hidden_dim=64,
                 embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        
        # define stem conv layers
        if stem_conv:
            self.stem = nn.Sequential(
                nn.Conv2D(in_channels,
                          hidden_dim,
                          kernel_size=7,
                          stride=stem_stride,
                          padding=3,
                          bias_attr=False),
                nn.BatchNorm2D(hidden_dim, momentum=0.9),
                nn.ReLU(),
                nn.Conv2D(hidden_dim,
                          hidden_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias_attr=False),
                nn.BatchNorm2D(hidden_dim, momentum=0.9),
                nn.ReLU(),
                nn.Conv2D(hidden_dim,
                          hidden_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias_attr=False),
                nn.BatchNorm2D(hidden_dim, momentum=0.9),
                nn.ReLU(),
            )
        else:
            self.stem = Identity()

        # define patch embeddings
        self.proj = nn.Conv2D(hidden_dim,
                              embed_dim,
                              kernel_size = patch_size // stem_stride,
                              stride = patch_size // stem_stride)
        # num patches
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

    def forward(self, x):
        x = self.stem(x) # Identity layer if stem is not set
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
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class OutlookerAttention(nn.Layer):
    """ Outlooker Attention

    Outlooker attention firstly applies a nn.Linear op, and unfold (im2col) the output
    tensor, then use tensor reshape to get the 'V'. 'Attn' is obtained by pool, linear and reshape
    ops applied on input tensor. Then a matmul is applied for 'V' and 'Attn'. Finally, a 
    fold op is applied with a linear projection to get the output.

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        kernel_size: int, size used in fold/unfold, and pool, default: 3
        padding: int, pad used in fold/unfold, default: 1
        stride: int, stride used in fold/unfold, and pool, default: 1
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn = nn.Linear(dim, (kernel_size ** 4) * num_heads)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

        self.pool = nn.AvgPool2D(kernel_size=stride, stride=stride, ceil_mode=True)

        self.unfold = paddle.nn.Unfold(kernel_sizes=kernel_size, strides=self.stride, paddings=self.padding)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x) # B, H, W, C
        v = v.transpose([0, 3, 1, 2]) # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # current paddle version has bugs using nn.Unfold 
        v = paddle.nn.functional.unfold(v,
                                        kernel_sizes=self.kernel_size,
                                        paddings=self.padding,
                                        strides=self.stride) # B, C*kernel_size*kernel_size, L(num of patches)

        v = v.reshape([B, 
                       self.num_heads,
                       C // self.num_heads,
                       self.kernel_size * self.kernel_size,
                       h * w])
        v = v.transpose([0, 1, 4, 3, 2])

        x = x.transpose([0, 3, 1, 2])
        attn = self.pool(x)
        attn = attn.transpose([0, 2, 3, 1]) # B, H', W', C
        attn = self.attn(attn)
        attn = attn.reshape([B,
                             h*w,
                             self.num_heads,
                             self.kernel_size * self.kernel_size,
                             self.kernel_size * self.kernel_size])
        attn = attn.transpose([0, 2, 1, 3, 4])
 
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 1, 4, 3, 2])
        new_shape = [B, C * self.kernel_size * self.kernel_size, h * w]
        z = z.reshape(new_shape)

        # Current Paddle dose not have Fold op, we hacked our fold op, see ./fold.py for details
        z = fold(z, output_size=(H, W), kernel_size=self.kernel_size,
                padding=self.padding, stride=self.stride)

        z = z.transpose([0, 2, 3, 1])
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class Outlooker(nn.Layer):
    """ Outlooker
    
    Outlooker contains norm layers, outlooker attention, mlp and droppath layers,
    and residual is applied during forward.

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        kernel_size: int, size used in fold/unfold, and pool, default: 3
        padding: int, pad used in fold/unfold, default: 1
        mlp_ratio: float, ratio to multiply with dim for mlp hidden feature dim, default: 3.
        stride: int, stride used in fold/unfold, and pool, default: 1
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """
    def __init__(self,
                 dim,
                 kernel_size,
                 padding,
                 stride=1,
                 num_heads=1,
                 mlp_ratio=3.,
                 attention_dropout=0.,
                 droppath=0.,
                 qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OutlookerAttention(dim,
                                       num_heads,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       attention_dropout=attention_dropout)
        self.drop_path = Droppath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio))

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
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape([B, H * W, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])

        z = z.reshape([B, H, W, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class Transformer(nn.Layer):
    """Transformer

    Transformer module, same as ViT

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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio))
    
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


class ClassAttention(nn.Layer):
    """ Class Attention

    Class Attention modlee same as CaiT

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
                 dim_head=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is not None:
            self.dim_head = dim_head
        else:
            self.dim_head = dim // num_heads

        self.scale = qk_scale or self.dim_head ** -0.5

        self.kv = nn.Linear(dim,
                            self.dim_head * self.num_heads * 2,
                            bias_attr=qkv_bias)
        self.q = nn.Linear(dim,
                           self.dim_head * self.num_heads,
                           bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(self.dim_head * self.num_heads, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x)
        kv = kv.reshape([B, N, 2, self.num_heads, self.dim_head])
        kv = kv.transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        q = self.q(x[:, :1, :])
        q = q.reshape([B, self.num_heads, 1, self.dim_head])
        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        cls_embed = paddle.matmul(attn, v)
        cls_embed = cls_embed.transpose([0, 2, 1, 3])
        cls_embed = cls_embed.reshape([B, 1, self.dim_head * self.num_heads])
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_dropout(cls_embed)
        return cls_embed
    

class ClassBlock(nn.Layer):
    """Class Attention Block (CaiT)

    CaiT module

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
                 dim_head=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim,
                                   num_heads=num_heads,
                                   dim_head=dim_head,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attention_dropout=attention_dropout,
                                   dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

    def forward(self, x):
        cls_embed = x[:, :1]

        h = self.norm1(x)
        h = self.attn(h)
        h = self.drop_path(h)
        cls_embed = cls_embed + h
        h = cls_embed
        cls_embed = self.norm2(cls_embed)
        cls_embed = self.mlp(cls_embed)
        cls_embed = self.drop_path(cls_embed)
        cls_embed = h + cls_embed
        out = paddle.concat([cls_embed, x[:, 1:]], axis=1)

        return out


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    # item() get the python native dtype
    return bbx1.item(), bby1.item(), bbx2.item(), bby2.item()


class VOLO(nn.Layer):
    def __init__(self,
               layers,
               image_size=224,
               in_channels=3,
               num_classes=1000,
               patch_size=8,
               stem_hidden_dim=64,
               embed_dims=None,
               num_heads=None,
               downsamples=None,
               outlook_attention=None,
               mlp_ratios=None,
               qkv_bias=False,
               qk_scale=None,
               dropout=0.,
               attention_dropout=0.,
               droppath=0.,
               num_post_layers=2,
               return_mean=False,
               return_dense=True,
               mix_token=True,
               pooling_scale=2,
               out_kernel=3,
               out_stride=2,
               out_padding=1):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                         stem_conv=True,
                                         stem_stride=2,
                                         patch_size=patch_size,
                                         in_channels=in_channels,
                                         hidden_dim=stem_hidden_dim,
                                         embed_dim=embed_dims[0])
        self.pos_embed = paddle.create_parameter(
            shape=[1,
                   image_size // patch_size // pooling_scale,
                   image_size // patch_size // pooling_scale,
                   embed_dims[-1]],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))

        self.pos_dropout = nn.Dropout(dropout)

        layer_list = []
        for i in range(len(layers)):
            blocks = []
            for block_idx in range(layers[i]):
                block_droppath = droppath * (
                    block_idx + sum(layers[:i])) / (sum(layers) - 1)
                if outlook_attention[i]:
                    blocks.append(
                        copy.deepcopy(
                        Outlooker(dim=embed_dims[i],
                                  kernel_size=out_kernel,
                                  padding=out_padding,
                                  stride=out_stride,
                                  num_heads=num_heads[i],
                                  mlp_ratio=mlp_ratios[i],
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  attention_dropout=attention_dropout,
                                  droppath=block_droppath)))
                else:
                    blocks.append(
                        copy.deepcopy(
                        Transformer(dim=embed_dims[i],
                                    num_heads=num_heads[i],
                                    mlp_ratio=mlp_ratios[i],
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    droppath=block_droppath))
                        )
            stage = nn.Sequential(*blocks)
            layer_list.append(stage)

            if downsamples[i]:
                layer_list.append(copy.deepcopy(Downsample(embed_dims[i], embed_dims[i + 1], 2)))

        self.model = nn.LayerList(layer_list)


        # POST Layers (from CaiT)
        self.post_model = None
        if num_post_layers is not None:
            self.post_model = nn.LayerList([
                copy.deepcopy(
                    ClassBlock(dim=embed_dims[-1],
                               num_heads=num_heads[-1],
                               mlp_ratio=mlp_ratios[-1],
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               attention_dropout=attention_dropout,
                               droppath=0.)
                ) for i in range(num_post_layers)
            ])
            self.cls_token = paddle.create_parameter(
                shape=[1, 1, embed_dims[-1]],
                dtype='float32',
                default_initializer=nn.initializer.TruncatedNormal(std=.02))

        # Output
        self.return_mean = return_mean # if True, return mean, not use class token
        self.return_dense = return_dense # if True, return class token and all feature tokens
        if return_dense:
            assert not return_mean, "Cannot return both mean and dense"
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:
            self.beta = 1.0
            assert return_dense, 'return all tokens if mix_token is enabled'
        if return_dense:
            self.aux_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else Identity()
        self.norm = nn.LayerNorm(embed_dims[-1])

        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else Identity()

        # For training:
        # TODO: set pos_embed, trunc_normal
        # TODO: set init weights for linear layers and layernorm layers
        # TODO: set no weight decay for pos_embed and cls_token


    def forward(self, x):
        # Step1: patch embedding
        x = self.patch_embed(x)
        x = x.transpose([0, 2, 3, 1])
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h = x.shape[1] // self.pooling_scale
            patch_w = x.shape[2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1 = self.pooling_scale * bbx1
            sbby1 = self.pooling_scale * bby1
            sbbx2 = self.pooling_scale * bbx2
            sbby2 = self.pooling_scale * bby2
            temp_x[:, sbbx1: sbbx2, sbby1: sbby2, :] = x.flip(axis=[0])[:, sbbx1: sbbx2, sbby1: sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
        # Step2: 2-stages tokens learning
        for idx, block in enumerate(self.model):
            if idx == 2: # add pos_embed after outlooker blocks (and a downsample layer)
                x = x + self.pos_embed
                x = self.pos_dropout(x)
            x = block(x)

        x = x.reshape([x.shape[0], -1, x.shape[-1]]) # B, H*W, C
        # Step3: post layers (from CaiT)
        if self.post_model is not None:
            cls_token = self.cls_token.expand([x.shape[0], -1, -1])
            x = paddle.concat([cls_token, x], axis=1)
            for block in self.post_model:
                x = block(x)
        x = self.norm(x)

        if self.return_mean:
            return self.head(x.mean(1))

        x_cls = self.head(x[:, 0])
        if not self.return_dense:
            return x_cls

        x_aux = self.aux_head(x[:, 1:])

        if not self.training:
            #NOTE: pytorch Tensor.max() returns a tuple of Tensor: (values, indices), while
            #      paddle Tensor.max() returns a single Tensor: values
            return x_cls + 0.5 * x_aux.max(1)

        if self.mix_token and self.training:
            x_aux = x_aux.reshape([x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1]])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(axis=[0])[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape([x_aux.shape[0], patch_h*patch_w, x_aux.shape[-1]])

        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

        

def build_volo(config):
    """build volo model using config"""
    model = VOLO(image_size=config.DATA.IMAGE_SIZE,
                 layers=config.MODEL.TRANS.LAYERS,
                 embed_dims=config.MODEL.TRANS.EMBED_DIMS,
                 mlp_ratios=config.MODEL.TRANS.MLP_RATIOS,
                 downsamples=config.MODEL.TRANS.DOWNSAMPLES,
                 outlook_attention=config.MODEL.TRANS.OUTLOOK_ATTENTION,
                 stem_hidden_dim=config.MODEL.STEM_HIDDEN_DIM,
                 num_heads=config.MODEL.TRANS.NUM_HEADS,
                 qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                 qk_scale=config.MODEL.TRANS.QK_SCALE)
    return model
