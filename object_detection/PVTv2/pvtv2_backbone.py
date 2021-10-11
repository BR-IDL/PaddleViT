#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
Implement Transformer Class for PVTv2
"""

import copy
import paddle
import paddle.nn as nn
from model_utils import DropPath


class Identity(nn.Layer):                      
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self):
        super(Identity, self).__init__()
 
    def forward(self, input):
        return input


class DWConv(nn.Layer):
    """Depth-Wise convolution 3x3

    Improve the local continuity of features.

    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose([0,2,1]).reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose([0,2,1])

        return x


class OverlapPatchEmbedding(nn.Layer):
    """Overlapping Patch Embedding

    Apply Overlapping Patch Embedding on input images. Embeddings is implemented using a Conv2D op.
    Making adjacent windows overlap by half of the area, and pad the feature map with zeros to keep 
    the resolution.

    Attributes:
        image_size: int, input image size, default: 224
        patch_size: int, size of patch, default: 7
        in_channels: int, input image channels, default: 3
        embed_dim: int, embedding dimension, default: 768
    """

    def __init__(self, image_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        image_size = (image_size, image_size) # TODO: add to_2tuple
        patch_size = (patch_size, patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.H, self.W = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.patch_embed = nn.Conv2D(in_channels=in_channels, 
                                     out_channels=embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=stride,
                                     padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.patch_embed(x) # [batch, embed_dim, h, w] h,w = patch_resolution
        _, _, H, W = x.shape
        x = x.flatten(start_axis=2, stop_axis=-1) # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1]) # [batch, h*w, embed_dim]
        x = self.norm(x) # [batch, num_patches, embed_dim]

        return x, H, W


class Mlp(nn.Layer):
    """ MLP module

    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> dwconv -> act -> dropout -> fc -> dropout

    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        dwconv: Depth-Wise Convolution
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """

    def __init__(self, in_features, hidden_features, dropout=0.0, linear=False):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
        
        self.dwconv = DWConv(hidden_features)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """ Attention module

    Attention module for PvT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.

    Attributes:
        dim: int, input dimension (channels)
        num_heads: number of heads
        q: a nn.Linear for q mapping
        kv: a nn.Linear for kv mapping
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
        linear: bool, if True, use linear spatial reduction attention instead of spatial reduction attention
        sr_ratio: the spatial reduction ratio of SRA (linear spatial reduction attention)
    """

    def __init__(self, 
                 dim, 
                 num_heads, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attention_dropout=0., 
                 dropout=0., 
                 sr_ratio=1, 
                 linear=False):
        """init Attention"""
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim, epsilon=1e-5)
        else:
            self.pool = nn.AdaptiveAvgPool2D(7)
            self.sr = nn.Conv2D(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim, epsilon=1e-5)
            self.act = nn.GELU()

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
                x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
            else:
                kv = self.kv(x).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(self.pool(x_)).reshape([B, C, -1]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class PvTv2Block(nn.Layer):
    """Pyramid VisionTransformerV2 block

    Contains multi head efficient self attention, droppath, mlp, norm.

    Attributes:
        dim: int, input dimension (channels)
        num_heads: int, number of attention heads
        mlp_ratio: float, ratio of mlp hidden dim and input embedding dim, default: 4.
        sr_ratio: the spatial reduction ratio of SRA (linear spatial reduction attention)
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        drop_path: float, drop path rate, default: 0.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., 
                 attention_dropout=0., drop_path=0., sr_ratio=1, linear=False):
        super(PvTv2Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale,
                              attention_dropout=attention_dropout, 
                              dropout=dropout, 
                              sr_ratio=sr_ratio, 
                              linear=linear)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim, 
                       hidden_features=int(dim*mlp_ratio), 
                       dropout=dropout, 
                       linear=linear)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class PyramidVisionTransformerV2(nn.Layer):
    """PyramidVisionTransformerV2 class

    Attributes:
        patch_size: int, size of patch
        image_size: int, size of image
        num_classes: int, num of image classes
        in_channels: int, channel of input image
        num_heads: int, num of heads in attention module 
        num_stages: int, num of stages contains OverlapPatch embedding and PvTv2 blocks      
        depths: list of int, num of PvTv2 blocks in each stage
        mlp_ratio: float, hidden dimension of mlp layer is mlp_ratio * mlp input dim
        sr_ratio: the spatial reduction ratio of SRA (linear spatial reduction attention)      
        qkv_bias: bool, if True, set qkv layers have bias enabled
        qk_scale: float, scale factor for qk.
        embed_dims: list of int, output dimension of patch embedding
        dropout: float, dropout rate for linear layer
        attention_dropout: float, dropout rate for attention
        drop_path: float, drop path rate, default: 0.
        linear: bool, if True, use linear spatial reduction attention instead of spatial reduction attention
        patch_embedding: PatchEmbedding, patch embedding instance
        norm: nn.LayerNorm, norm layer applied after transformer
        fc: nn.Linear, classifier op.
    """

    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 embed_dims=[32, 64, 160, 256],
                 num_classes=1000,
                 in_channels=3,
                 num_heads=[1, 2, 5, 8],
                 depths=[2, 2, 2, 2],
                 mlp_ratio=[8, 8, 4, 4],
                 sr_ratio=[8, 4, 2, 1],
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 drop_path=0.,
                 linear=False,
                 pretrained=None):
        super(PyramidVisionTransformerV2, self).__init__()

        self.patch_size = patch_size 
        self.image_size = image_size
        #self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.depths = depths
        self.num_stages = len(self.depths)
        self.mlp_ratio = mlp_ratio 
        self.sr_ratio = sr_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.attention_dropout = attention_dropout 
        self.drop_path = drop_path
        self.linear = linear

        depth_decay = [x.item() for x in paddle.linspace(0, self.drop_path, sum(self.depths))]
        cur = 0

        for i in range(self.num_stages):
            patch_embedding = OverlapPatchEmbedding(image_size=self.image_size if i == 0 else self.image_size // (2 ** (i + 1)),
                                                patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_channels=self.in_channels if i == 0 else self.embed_dims[i - 1],
                                                embed_dim=self.embed_dims[i])

            block = nn.LayerList([copy.deepcopy(PvTv2Block(
                dim=self.embed_dims[i], num_heads=self.num_heads[i], mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias, 
                qk_scale=self.qk_scale, dropout=self.dropout, attention_dropout=self.attention_dropout, 
                drop_path=depth_decay[cur + j], sr_ratio=self.sr_ratio[i], linear=self.linear))
                for j in range(self.depths[i])])
            norm = nn.LayerNorm(self.embed_dims[i], epsilon=1e-6)
            cur += self.depths[i]

            setattr(self, f"patch_embedding{i + 1}", patch_embedding)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        #self.head = nn.Linear(self.embed_dims[3], self.num_classes) if self.num_classes > 0 else Identity()

        self.init_weights(pretrained)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            model_state_dict = paddle.load(pretrained)
            self.set_state_dict(model_state_dict)
        
    def freeze_patch_embedding(self):
        self.patch_embedding1.requires_grad = False

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embedding = getattr(self, f"patch_embedding{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embedding(x)

            for idx, blk in enumerate(block):
                x = blk(x, H, W)
            x = norm(x)
            #if i != self.num_stages - 1:
            #    x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])

            x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)

        return x


def build_pvtv2(config):
    model = PyramidVisionTransformerV2(
        image_size=config.DATA.IMAGE_SIZE,
        patch_size=config.MODEL.TRANS.PATCH_SIZE,
        embed_dims=config.MODEL.TRANS.EMBED_DIMS,
        num_classes=config.MODEL.NUM_CLASSES,
        in_channels=config.MODEL.TRANS.IN_CHANNELS,
        num_heads=config.MODEL.TRANS.NUM_HEADS,
        depths=config.MODEL.TRANS.STAGE_DEPTHS,
        mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
        sr_ratio=config.MODEL.TRANS.SR_RATIO,
        qkv_bias=config.MODEL.TRANS.QKV_BIAS,
        qk_scale=config.MODEL.TRANS.QK_SCALE,
        dropout=config.MODEL.DROPOUT,
        attention_dropout=config.MODEL.ATTENTION_DROPOUT,
        drop_path=config.MODEL.DROP_PATH,
        linear=config.MODEL.TRANS.LINEAR,
        pretrained=None)
        #pretrained='/workspace/ppvit_github/weights/pvtv2/pvtv2_b0.pdparams')
    return model
