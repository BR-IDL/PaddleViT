#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
Implement Mix Transformer of Segformer
Segformer: https://arxiv.org/abs/2105.15203 

Adapted from two repositories below:
    https://github.com/NVlabs/SegFormer
    https://github.com/open-mmlab/mmsegmentation
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.utils import load_pretrained_model


def to_2tuple(ele):
    return (ele, ele)


def nlc_to_nchw(x, H, W):
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W
    return x.transpose([0, 2, 1]).reshape([B, C, H, W])


def nchw_to_nlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).transpose([0, 2, 1])


class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape                                                                                                                
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1
                                               )  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()  # mask
        output = inputs.divide(
            keep_prob
        ) * random_tensor  # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class PatchEmbed(nn.Layer):
    """
    use a conv layer to implement PatchEmbed.
    odd kernel size perform overlap patch embedding
    even kernel size perform non-overlap patch embedding
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape
            to multiple patch size. Default: True.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 pad_to_patch_size=True):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'
        self.patch_size = patch_size

        # Use conv layer to embed
        self.projection = nn.Conv2D(in_channels=in_channels,
                                    out_channels=embed_dims,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = nchw_to_nlc(x)
        x = self.norm(x)

        return x


class MixFFN(nn.Layer):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """
    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.):
        super(MixFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        in_channels = embed_dims

        self.act = nn.GELU()
        self.fc1 = nn.Conv2D(in_channels=in_channels,
                             out_channels=feedforward_channels,
                             kernel_size=1,
                             stride=1,
                             bias_attr=None)
        # 3x3 depth wise conv to provide positional encode information
        self.pe_conv = nn.Conv2D(in_channels=feedforward_channels,
                                 out_channels=feedforward_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=(3 - 1) // 2,
                                 bias_attr=None,
                                 groups=feedforward_channels)
        self.fc2 = nn.Conv2D(in_channels=feedforward_channels,
                             out_channels=in_channels,
                             kernel_size=1,
                             stride=1,
                             bias_attr=None)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x, H, W):
        x = nlc_to_nchw(x, H, W)
        x = self.fc1(x)
        x = self.pe_conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = nchw_to_nlc(x)

        return x


class EfficientAttention(nn.Layer):
    """ An implementation of Efficient Multi-head Attention of Segformer.
    
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super(EfficientAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads

        w_attr_0, b_attr_0 = self._init_weights()
        w_attr_1, b_attr_1 = self._init_weights()
        w_attr_2, b_attr_2 = self._init_weights()
        self.q = nn.Linear(dim,
                           dim,
                           weight_attr=w_attr_0,
                           bias_attr=b_attr_0 if qkv_bias else False)
        self.kv = nn.Linear(dim,
                            dim * 2,
                            weight_attr=w_attr_1,
                            bias_attr=b_attr_1 if qkv_bias else False)
        self.proj = nn.Linear(dim,
                              dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)

        self.scales = (dim // num_heads)**-0.5  # 0.125 for Large
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim,
                                dim,
                                kernel_size=sr_ratio,
                                stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = nlc_to_nchw(x, H, W)
            x_ = self.sr(x_)
            x_ = nchw_to_nlc(x_)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape([B,-1,2,C]).transpose([2, 0, 1,3])
        else:
            kv =  self.kv(x).reshape([B,-1,2,C]).transpose([2, 0, 1,3])
        k, v = kv[0], kv[1]

        q, k, v = [x.transpose([1,0,2]) for x in (q, k, v)]
        q, k, v = [x.reshape([-1,B*self.num_heads,C//self.num_heads]) for x in (q, k, v)]
        q, k, v = [x.transpose([1,0,2]) for x in (q, k, v)]
        attn = paddle.matmul(q, k, transpose_y=True)* self.scales
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose([1,0,2]).reshape([N, B, C])
        x = self.proj(x).transpose([1,0,2])
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(nn.Layer):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else Identity()

        self.attn = EfficientAttention(dim=embed_dims,
                                       num_heads=num_heads,
                                       attn_drop=attn_drop_rate,
                                       proj_drop=drop_rate,
                                       qkv_bias=qkv_bias,
                                       sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = MixFFN(embed_dims=embed_dims,
                          feedforward_channels=feedforward_channels,
                          ffn_drop=drop_rate)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Layer):
    """The backbone of Segformer.

    A Paddle implement of : `SegFormer: Simple and Efficient Design for
    Semantic Segmentation with Transformers` -
        https://arxiv.org/pdf/2105.15203.pdf

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        pretrained (str, optional): model pretrained path. Default: None.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pretrained=None):
        super(MixVisionTransformer, self).__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) \
            == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained

        dpr = [x for x in paddle.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.layers = nn.LayerList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(in_channels=in_channels,
                                     embed_dims=embed_dims_i,
                                     kernel_size=patch_sizes[i],
                                     stride=strides[i],
                                     padding=patch_sizes[i] // 2,
                                     pad_to_patch_size=False)
            layer = nn.LayerList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(nn.LayerList([patch_embed, layer, norm]))
            cur += num_layer

        if isinstance(self.pretrained, str):
            load_pretrained_model(self, self.pretrained)

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, H, W = layer[0](x), layer[0].DH, layer[0].DW
            for block in layer[1]:
                x = block(x, H, W)
            x = layer[2](x)
            x = nlc_to_nchw(x, H, W)
            if i in self.out_indices:
                outs.append(x)

        return outs
