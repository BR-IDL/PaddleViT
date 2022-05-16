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
CoaT in Paddle
A Paddle Implementation of Co-Sacle Conv-Attentional Transformer (CoaT) as described in:
"Co-Scale Conv-Attentional Image Transformers"
    - Paper Link: https://arxiv.org/pdf/2104.06399.pdf
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from droppath import DropPath


class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def forward(self, x):
        return x


class Mlp(nn.Layer):
    """ MLP module

    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout

    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc1 and fc2
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

        w_attr, b_attr = self._init_weights_layer()
        self.fc1 = nn.Linear(in_features, hidden_features, weight_attr=w_attr, bias_attr=b_attr)
        self.act = act_layer()
        w_attr, b_attr = self._init_weights_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, weight_attr=w_attr, bias_attr=b_attr)
        self.drop = nn.Dropout(drop)

    def _init_weights_layer(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Layer):
    """Patch Embeddings
    Apply patch embedding (which is implemented using Conv2D) on input data.

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

        w_attr, b_attr = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(embed_dim,
                                 weight_attr=w_attr,
                                 bias_attr=b_attr)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.patch_embed(x)  # [batch, embed_dim, h, w]; h,w = patch_resolution
        x = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w]; h*w = num_patches
        x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
        x = self.norm(x)  # [batch, num_patches, embed_dim]
        return x


class ConvRelPosEnc(nn.Layer):
    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.LayerList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2D(cur_head_split*Ch,
                                 cur_head_split*Ch,
                                 kernel_size=(cur_window, cur_window), 
                                 padding=(padding_size, padding_size),
                                 dilation=(dilation, dilation),                          
                                 groups=cur_head_split*Ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose([0, 1, 3, 2]).reshape([B, h * Ch, H, W])
        v_img_list = paddle.split(v_img, self.channel_splits, axis=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = paddle.concat(conv_v_img_list, axis=1)
        conv_v_img = conv_v_img.reshape([B, h, Ch, H * W]).transpose([0, 1, 3, 2])

        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
        return EV_hat


class FactorAttnConvRelPosEnc(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim,
                             dim * 3,
                             weight_attr=w_attr_1,
                             bias_attr=False if qkv_bias is False else b_attr_1)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim,
                              dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.crpe = shared_crpe

        self.softmax = nn.Softmax(axis=2)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x)
        qkv = qkv.reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, Ch]

        # Factorized attention.

        k = self.softmax(k)
        factor_att = paddle.matmul(q, k, transpose_y=True)
        factor_att = paddle.matmul(factor_att, v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # [B, h, N, Ch]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose([0, 2, 1, 3]).reshape([B, N, C])  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Layer):
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2D(dim, dim, k, 1, k//2, groups=dim) 
    
    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        
        feat = img_tokens.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose([0, 2, 1])

        x = paddle.concat((cls_token, x), axis=1)

        return x


class SerialBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 shared_cpe=None,
                 shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(dim,
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop_rate=attn_drop_rate,
                                                      proj_drop_rate=drop_rate,
                                                      shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else Identity()

        # MLP.
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop_rate)

    def forward(self, x, size):
        # Conv-Attention
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur) 

        # MLP
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(nn.Layer):
    def __init__(self,
                 dims,
                 num_heads,
                 mlp_ratios,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 shared_crpes=None):
        super().__init__()
        # conv-attention
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm12 = nn.LayerNorm(dims[1], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm13 = nn.LayerNorm(dims[2], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm14 = nn.LayerNorm(dims[3], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        self.factoratt_crpe2 = FactorAttnConvRelPosEnc(dims[1],
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop_rate=attn_drop_rate,
                                                      proj_drop_rate=drop_rate,
                                                      shared_crpe=shared_crpes[1])
        self.factoratt_crpe3 = FactorAttnConvRelPosEnc(dims[2],
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop_rate=attn_drop_rate,
                                                      proj_drop_rate=drop_rate,
                                                      shared_crpe=shared_crpes[2])
        self.factoratt_crpe4 = FactorAttnConvRelPosEnc(dims[3],
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop_rate=attn_drop_rate,
                                                      proj_drop_rate=drop_rate,
                                                      shared_crpe=shared_crpes[3])
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else Identity()

        # MLP
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm22 = nn.LayerNorm(dims[1], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm23 = nn.LayerNorm(dims[2], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        w_attr, b_attr = self._init_weights_layernorm()
        self.norm24 = nn.LayerNorm(dims[3], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = Mlp(dims[1], mlp_hidden_dim, drop_rate)
        self.mlp3 = Mlp(dims[1], mlp_hidden_dim, drop_rate)
        self.mlp4 = Mlp(dims[1], mlp_hidden_dim, drop_rate)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def upsample(self, x, factor, size):
        return self.interpolate(x, scale_factor=factor, size=size)

    def downsample(self, x, factor, size):
        return self.interpolate(x, scale_factor=1.0/factor, size=size)

    def interpolate(seld, x, scale_factor, size):
        B, N, C = x.shape
        H, W = size

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose([0, 2, 1]).reshape([B, C, H, W])
        img_tokens = F.interpolate(img_tokens,
                                   scale_factor=scale_factor,
                                   mode='bilinear',
                                   align_corners=False)
        img_tokens = img_tokens.reshape([B, C, -1]).transpose([0, 2, 1])
        out = paddle.concat((cls_token, img_tokens), axis=1)
        return out

    def forward(self, x1, x2, x3, x4, sizes):
        _, S2, S3, S4 = sizes
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=S2)
        cur3 = self.factoratt_crpe3(cur3, size=S3)
        cur4 = self.factoratt_crpe4(cur4, size=S4)
        upsample3_2 = self.upsample(cur3, factor=2., size=S3)
        upsample4_3 = self.upsample(cur4, factor=2., size=S4)
        upsample4_2 = self.upsample(cur4, factor=4., size=S4)
        downsample2_3 = self.downsample(cur2, factor=2., size=S2)
        downsample3_4 = self.downsample(cur3, factor=2., size=S3)
        downsample2_4 = self.downsample(cur2, factor=4., size=S2)
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2) 
        x3 = x3 + self.drop_path(cur3) 
        x4 = x4 + self.drop_path(cur4) 

        # MLP. 
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4) 

        return x1, x2, x3, x4


class CoaT(nn.Layer):
    def __init__(self,
                 image_size,
                 patch_size,
                 in_channels=3,
                 num_classes=1000,
                 embed_dims=(0, 0, 0, 0),
                 serial_depths=(0, 0, 0, 0),
                 parallel_depth=0,
                 num_heads=0,
                 mlp_ratios=(0, 0, 0, 0),
                 qkv_bias=True,
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 return_interm_layers=False,
                 out_features=None,
                 crpe_window={3:2, 5:3, 7:3},
                 global_pool='token'):
        super().__init__()
        assert global_pool in ['token', 'avg']
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool

        self.patch_embeds = nn.LayerList()
        self.cls_tokens = nn.ParameterList()
        self.cpes = nn.LayerList()
        self.crpes = nn.LayerList()
        self.serial_blocks = nn.LayerList()

        for idx in range(4):
            # patch embedding
            self.patch_embeds.append(
                PatchEmbed(image_size=image_size if idx == 0 else image_size // (2**(idx+1)),
                           patch_size=patch_size if idx == 0 else 2,
                           in_channels=in_channels if idx == 0 else embed_dims[idx - 1],
                           embed_dim=embed_dims[idx]))
            # class token
            self.cls_tokens.append(
                paddle.create_parameter(
                    shape=[1, 1, embed_dims[idx]],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02),
                ))

            # convolutional position encoding
            self.cpes.append(ConvPosEnc(dim=embed_dims[idx], k=3))

            # convolution relative position encoding
            self.crpes.append(ConvRelPosEnc(Ch=embed_dims[idx] // num_heads, h=num_heads, window=crpe_window))

            serial_block = []
            for _ in range(serial_depths[idx]):
                serial_block.append(
                    SerialBlock(dim=embed_dims[idx],
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratios[idx],
                                qkv_bias=qkv_bias,
                                drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate,
                                drop_path_rate=drop_path_rate,
                                shared_cpe=self.cpes[idx],
                                shared_crpe=self.crpes[idx],
                                ))
            self.serial_blocks.append(nn.LayerList(serial_block))

        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            parallel_blocks = []
            for _ in range(parallel_depth):
                parallel_blocks.append(
                    ParallelBlock(dims=embed_dims,
                                  num_heads=num_heads,
                                  mlp_ratios=mlp_ratios,
                                  qkv_bias=qkv_bias,
                                  drop_rate=drop_rate,
                                  attn_drop_rate=attn_drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  shared_crpes=self.crpes))
            self.parallel_blocks = nn.LayerList(parallel_blocks)
        else:
            self.parallel_blocks = None

        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                w_attr, b_attr = self._init_weights_layernorm()
                self.norm2 = nn.LayerNorm(embed_dims[1], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
                w_attr, b_attr = self._init_weights_layernorm()
                self.norm3 = nn.LayerNorm(embed_dims[2], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)
            else:
                self.norm2 = None
                self.norm3 = None

            w_attr, b_attr = self._init_weights_layernorm()
            self.norm4 = nn.LayerNorm(embed_dims[3], weight_attr=w_attr, bias_attr=b_attr, epsilon=1e-6)

            if self.parallel_depth > 0:
                # coat
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = nn.Conv1D(3, 1, 1)
                w_attr, b_attr = self._init_weights()
                self.head = nn.Linear(self.num_features,
                                      num_classes,
                                      weight_attr=w_attr,
                                      bias_attr=b_attr) if num_classes > 0 else Identity()
            else:
                # coat lite
                self.aggregate = None
                w_attr, b_attr = self._init_weights()
                self.head = nn.Linear(self.num_features,
                                      num_classes,
                                      weight_attr=w_attr,
                                      bias_attr=b_attr) if num_classes > 0 else Identity()

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward_features(self, x0):
        B = x0.shape[0]
        # serial block 1
        x1 = self.patch_embeds[0](x0)
        H1, W1 = self.patch_embeds[0].patches_resolution
        cls_tokens1 = self.cls_tokens[0].expand((x1.shape[0], -1, -1))
        x1 = paddle.concat((cls_tokens1, x1), axis=1)
        for idx, block in enumerate(self.serial_blocks[0]):
            x1 = block(x1, size=(H1, W1))
        x1_nocls = x1[:, 1:, :].reshape([B, H1, W1, -1]).transpose([0, 3, 1, 2])

        # serial block 2
        x2 = self.patch_embeds[1](x1_nocls)
        H2, W2 = self.patch_embeds[1].patches_resolution
        cls_tokens2 = self.cls_tokens[1].expand((x2.shape[0], -1, -1))
        x2 = paddle.concat((cls_tokens2, x2), axis=1)
        for block in self.serial_blocks[1]:
            x2 = block(x2, size=(H2, W2))
        x2_nocls = x2[:, 1:, :].reshape([B, H2, W2, -1]).transpose([0, 3, 1, 2])

        # serial block 3
        x3 = self.patch_embeds[2](x2_nocls)
        H3, W3 = self.patch_embeds[2].patches_resolution
        cls_tokens3 = self.cls_tokens[2].expand((x3.shape[0], -1, -1))
        x3 = paddle.concat((cls_tokens3, x3), axis=1)
        for block in self.serial_blocks[2]:
            x3 = block(x3, size=(H3, W3))
        x3_nocls = x3[:, 1:, :].reshape([B, H3, W3, -1]).transpose([0, 3, 1, 2])

        # serial block 4
        x4 = self.patch_embeds[3](x3_nocls)
        H4, W4 = self.patch_embeds[3].patches_resolution
        cls_tokens4 = self.cls_tokens[3].expand((x4.shape[0], -1, -1))
        x4 = paddle.concat((cls_tokens4, x4), axis=1)
        for block in self.serial_blocks[3]:
            x4 = block(x4, size=(H4, W4))
        x4_nocls = x4[:, 1:, :].reshape([B, H4, W4, -1]).transpose([0, 3, 1, 2])

        if self.parallel_blocks is None:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                return x4
        
        # Parallel blocks
        for block in self.parallel_blocks:
            x2 = self.cpes[1](x2, (H2, W2))
            x3 = self.cpes[2](x3, (H3, W3))
            x4 = self.cpes[3](x4, (H4, W4))
            x1, x2, x3, x4 = block(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = x1[:, 1:, :].reshape([B, H1, W1, -1]).transpose([0, 3, 1, 2])
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = x2[:, 1:, :].reshape([B, H2, W2, -1]).transpose([0, 3, 1, 2])
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = x3[:, 1:, :].reshape([B, H3, W3, -1]).transpose([0, 3, 1, 2])
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = x4[:, 1:, :].reshape([B, H4, W4, -1]).transpose([0, 3, 1, 2])
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            return [x2, x3, x4]

    def forward_head(self, x_feat, pre_logits=False):
        if isinstance(x_feat, list):
            assert self.aggregate is not None
            if self.global_pool == 'avg':
                x = paddle.concat([xl[:, 1:].mean(1, keepdim=True) for xl in x_feat], axis=1) # [B, 3, C]
            else:
                x = paddle.stack([xl[:, 0] for xl in x_feat], axis=1)  # [B, 3, C]
            x = self.aggregate(x).squeeze(1)  # [B, C]
        else:
            x = x_feat[:, 1:].mean(1) if self.global_pool == 'avg' else x_feat[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        if self.return_interm_layers:
            return self.forward_features(x)
        else:
            x_feat = self.forward_features(x)
            x = self.forward_head(x_feat)
            return x


def build_coat(config):
    model = CoaT(
        image_size=config.DATA.IMAGE_SIZE,
        patch_size=config.MODEL.PATCH_SIZE,
        in_channels=config.DATA.IMAGE_CHANNELS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dims=config.MODEL.EMBED_DIMS,
        serial_depths=config.MODEL.SERIAL_DEPTHS,
        parallel_depth=config.MODEL.PARALLEL_DEPTH,
        num_heads=config.MODEL.NUM_HEADS,
        mlp_ratios=config.MODEL.MLP_RATIOS,
        qkv_bias=config.MODEL.QKV_BIAS,
        attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
        drop_rate=config.MODEL.DROPOUT,
        drop_path_rate=config.MODEL.DROPPATH,
        return_interm_layers=False,
        out_features=None,
        crpe_window={3:2, 5:3, 7:3},
        global_pool='token',
    )
    return model
