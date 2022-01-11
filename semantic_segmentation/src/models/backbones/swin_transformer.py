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
Implement Transformer Class for Swin Transformer
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


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
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)

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
        #image_size = (image_size, image_size) # TODO: add to_2tuple
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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        # TODO (wutianyiRosun@gmail.com): padding for processing different resolutions of input images
        #if W % self.patch_size[1] !=0:
        #   x = F.pad(x (0, self.patch_size[1] - W % self.patch_size[1]))
        #if H % self.patch_size[1] !=0:
        #   x = F.pad(x (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
          
        x = self.patch_embed(x) # [batch, embed_dim, h, w] h,w = patch_resolution
        x = x.flatten(start_axis=2, stop_axis=-1) # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1]) # [batch, h*w, embed_dim]
        x = self.norm(x) # [batch, num_patches, embed_dim]
        return x


class PatchMerging(nn.Layer):
    """ Patch Merging class

    Merge multiple patch into one path and keep the out dim.
    Spefically, merge adjacent 2x2 patches(dim=C) into 1 patch.
    The concat dim 4*C is rescaled to 2*C

    Attributes:
        input_resolution: tuple of ints, the size of input
        dim: dimension of single patch
        reduction: nn.Linear which maps 4C to 2C dim
        norm: nn.LayerNorm, applied after linear layer.
    """

    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias_attr=False)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :] # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :] # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :] # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :] # [B, H/2, W/2, C]
        x = paddle.concat([x0, x1, x2, x3], -1) #[B, H/2, W/2, 4*C]
        x = x.reshape([b, -1, 4*c]) # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)

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

    def __init__(self, in_features, hidden_features, dropout):
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


class WindowAttention(nn.Layer):
    """Window based multihead attention, with relative position bias.

    Both shifted window and non-shifted window are supported.

    Attributes:
        dim: int, input dimension (channels)
        window_size: int, height and width of the window
        num_heads: int, number of attention heads
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        attention_dropout: float, dropout of attention
        dropout: float, dropout for output
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] -1) * (2 * window_size[1] - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        # relative position index for each token inside window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w])) # [2, window_h, window_w]
        coords_flatten = paddle.flatten(coords, 1) # [2, window_h * window_w]
        # 2, window_h * window_w, window_h * window_h
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        # winwod_h*window_w, window_h*window_w, 2
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2* self.window_size[1] - 1
        # [window_size * window_size, window_size*window_size]
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1]) # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        relative_position_bias = self.get_relative_pos_bias_from_pos_index()
        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1],
             -1])
        # nH, window_h*window_w, window_h*window_w
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(
                [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


def windows_partition(x, window_size):
    """ partite windows into window_size x window_size
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """

    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([-1, window_size, window_size, C]) #(num_windows*B, window_size, window_size, C)
    return x


def windows_reverse(windows, window_size, H, W):
    """ Window reverse
    Args:
        windows: (n_windows * B, window_size, window_size, C)
        window_size: (int) window size
        H: (int) height of image
        W: (int) width of image

    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x


class SwinTransformerBlock(nn.Layer):
    """Swin transformer block

    Contains window multi head self attention, droppath, mlp, norm and residual.

    Attributes:
        dim: int, input dimension (channels)
        input_resolution: int, input resoultion
        num_heads: int, number of attention heads
        windos_size: int, window size, default: 7
        shift_size: int, shift size for SW-MSA, default: 0
        mlp_ratio: float, ratio of mlp hidden dim and input embedding dim, default: 4.
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim,
                                    window_size=(self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim*mlp_ratio),
                       dropout=dropout)

        if self.shift_size > 0:
            H, W = self.input_resolution
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = paddle.zeros((1, Hp, Wp, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = paddle.where(attn_mask != 0,
                                     paddle.ones_like(attn_mask) * float(-100.0),
                                     attn_mask)
            attn_mask = paddle.where(attn_mask == 0,
                                     paddle.zeros_like(attn_mask),
                                     attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        h = x
        x = self.norm1(x)

        new_shape = [B, H, W, C]
        x = x.reshape(new_shape)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = x.transpose([0, 3, 1, 2]) # (B,C,H,W)
        x = F.pad(x, [pad_l, pad_r, pad_t, pad_b]) 
        x = x.transpose([0, 2, 3, 1]) # (B,H,W,C)
        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size, -self.shift_size),
                                    axis=(1, 2))
        else:
            shifted_x = x
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = windows_reverse(attn_windows, self.window_size, Hp, Wp)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        x = x.reshape([B, H*W, C])
        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        return x


class SwinTransformerStage(nn.Layer):
    """Stage layers for swin transformer

    Stage layers contains a number of Transformer blocks and an optional
    patch merging layer, patch merging is not applied after last stage

    Attributes:
        dim: int, embedding dimension
        input_resolution: tuple, input resoliution
        depth: list, num of blocks in each stage
        blocks: nn.LayerList, contains SwinTransformerBlocks for one stage
        downsample: PatchMerging, patch merging layer, none if last stage
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0., downsample=None):
        super(SwinTransformerStage, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.LayerList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    dropout=dropout, attention_dropout=attention_dropout,
                    droppath=droppath[i] if isinstance(droppath, list) else droppath))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return [x, x_down] 
        return [x, x]


class SwinTransformer(nn.Layer):
    """SwinTransformer class

    Attributes:
        num_classes: int, num of image classes
        num_stages: int, num of stages contains patch merging and Swin blocks
        depths: list of int, num of Swin blocks in each stage
        num_heads: int, num of heads in attention module
        embed_dim: int, output dimension of patch embedding
        num_features: int, output dimension of whole network before classifier
        mlp_ratio: float, hidden dimension of mlp layer is mlp_ratio * mlp input dim
        qkv_bias: bool, if True, set qkv layers have bias enabled
        qk_scale: float, scale factor for qk.
        ape: bool, if True, set to use absolute positional embeddings
        window_size: int, size of patch window for inputs
        dropout: float, dropout rate for linear layer
        dropout_attn: float, dropout rate for attention
        patch_embedding: PatchEmbedding, patch embedding instance
        patch_resolution: tuple, number of patches in row and column
        position_dropout: nn.Dropout, dropout op for position embedding
        stages: SwinTransformerStage, stage instances.
        norm: nn.LayerNorm, norm layer applied after transformer
        avgpool: nn.AveragePool2D, pooling layer before classifer
        fc: nn.Linear, classifier op.
    """
    def __init__(self, config):
        super(SwinTransformer, self).__init__()

        self.num_classes = config.DATA.NUM_CLASSES
        self.num_stages = len(config.MODEL.TRANS.STAGE_DEPTHS)
        self.depths = config.MODEL.TRANS.STAGE_DEPTHS
        self.num_heads = config.MODEL.TRANS.NUM_HEADS
        self.embed_dim = config.MODEL.TRANS.EMBED_DIM
        self.num_features = int(self.embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = config.MODEL.TRANS.MLP_RATIO
        self.qkv_bias = config.MODEL.TRANS.QKV_BIAS
        self.qk_scale = config.MODEL.TRANS.QK_SCALE
        self.ape = config.MODEL.TRANS.APE
        self.window_size = config.MODEL.TRANS.WINDOW_SIZE
        self.dropout = config.MODEL.DROPOUT
        self.attention_dropout = config.MODEL.ATTENTION_DROPOUT
        self.out_indices = config.MODEL.ENCODER.OUT_INDICES

        self.patch_embedding = PatchEmbedding(image_size=config.DATA.CROP_SIZE,
                                              patch_size=config.MODEL.TRANS.PATCH_SIZE,
                                              in_channels=config.MODEL.TRANS.IN_CHANNELS,
                                              embed_dim=config.MODEL.TRANS.EMBED_DIM)
        num_patches = self.patch_embedding.num_patches
        self.patches_resolution = self.patch_embedding.patches_resolution


        if self.ape:
            self.absolute_positional_embedding = paddle.nn.ParameterList([
                paddle.create_parameter(
                    shape=[1, num_patches, self.embed_dim], dtype='float32',
                    default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))])

        self.position_dropout = nn.Dropout(config.MODEL.DROPOUT)

        depth_decay = [x for x in paddle.linspace(0, config.MODEL.DROP_PATH, sum(self.depths))]

        self.stages = nn.LayerList()
        for stage_idx in range(self.num_stages):
            stage = SwinTransformerStage(
                dim=int(self.embed_dim * 2 ** stage_idx),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** stage_idx),
                    self.patches_resolution[1] // (2 ** stage_idx)),
                depth=self.depths[stage_idx],
                num_heads=self.num_heads[stage_idx],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                droppath=depth_decay[
                    sum(self.depths[:stage_idx]):sum(self.depths[:stage_idx+1])],
                downsample=PatchMerging if (
                    stage_idx < self.num_stages-1) else None,
                )
            self.stages.append(stage)

    def forward(self, x):
        x = self.patch_embedding(x)  # (B, HW/16, dim)
        if self.ape:
            x = x + self.absolute_positional_embedding
        x = self.position_dropout(x)
        outs = []
        for idx in range(len(self.stages)):
            x_out, x = self.stages[idx](x)
            if idx in self.out_indices:
                outs.append(x_out)
        return outs

