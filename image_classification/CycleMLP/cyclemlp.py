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
CycleMLP in Paddle
A Paddle Implementation of CycleMLP as described in:
"CycleMLP: A MLP-like Architecture for Dense Prediction"
    - Paper Link: https://arxiv.org/abs/2107.10224
"""

import os
import math
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.vision.ops import deform_conv2d
import paddle.nn.functional as F
from droppath import DropPath


zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)
trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
kaiming_uniform_ = nn.initializer.KaimingUniform()


class Identity(nn.Layer):
    """Identity layer
    This is does nothing but passing the input as output
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc
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


class CycleFC(nn.Layer):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if stride != 1:
            raise ValueError("stride must be 1")
        if padding != 0:
            raise ValueError("padding must be 0")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups

        self.weight = self.create_parameter(
            shape=[out_channels, in_channels // groups, 1, 1],
            default_initializer=kaiming_uniform_,
        )  # kernel size == 1

        if bias:
            bound = 1 / math.sqrt(self.weight.shape[1])
            self.bias = self.create_parameter(
                shape=[out_channels],
                default_initializer=nn.initializer.Uniform(-bound, bound),
            )
        else:
            self.bias = None
        self.register_buffer("offset", self.gen_offset())

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = paddle.empty([1, self.in_channels * 2, 1, 1])
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (
                    self.kernel_size[1] // 2
                )
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (
                    self.kernel_size[0] // 2
                )
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = inputs.shape
        deformable_groups = self.offset.shape[1] // (
            2 * self.weight.shape[2] * self.weight.shape[3])

        return deform_conv2d(inputs,
                             self.offset.expand([B, -1, H, W]),
                             self.weight,
                             self.bias,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             deformable_groups=deformable_groups)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + "("
        s += "{in_channels}"
        s += ", {out_channels}"
        s += ", kernel_size={kernel_size}"
        s += ", stride={stride}"
        s += ", padding={padding}" if self.padding != (0, 0) else ""
        s += ", dilation={dilation}" if self.dilation != (1, 1) else ""
        s += ", groups={groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"
        return s.format(**self.__dict__)


class CycleMLP(nn.Layer):
    def __init__(self,
                 dim,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.transpose([0, 3, 1, 2])).transpose([0, 2, 3, 1])
        w = self.sfc_w(x.transpose([0, 3, 1, 2])).transpose([0, 2, 3, 1])
        c = self.mlp_c(x)

        a = (h + w + c).transpose([0, 3, 1, 2]).flatten(2).mean(2)
        a = F.softmax(self.reweight(a).reshape((B, C, 3)).transpose([2, 0, 1]), axis=0)
        a = a.unsqueeze(2)
        a = a.unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CycleBlock(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 skip_lam=1.0,
                 mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Layer):
    """2D Image to Patch Embedding with overlapping"""

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 groups=1):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=padding,
                              groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Layer):
    """Downsample transition stage"""
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2D(in_embed_dim,
                              out_embed_dim,
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=1)

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        x = self.proj(x)  # B, C, H, W
        x = x.transpose([0, 2, 3, 1])
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 mlp_ratio=3.0,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 drop_path_rate=0.0,
                 skip_lam=1.0,
                 mlp_fn=CycleMLP,
                 **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            CycleBlock(
                dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop_path=block_dpr,
                skip_lam=skip_lam,
                mlp_fn=mlp_fn,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class CycleNet(nn.Layer):
    """CycleMLP Network"""
    def __init__(self,
                 layers,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=None,
                 transitions=None,
                 segment_dim=None,
                 mlp_ratios=None,
                 skip_lam=1.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 mlp_fn=CycleMLP,
                 fork_feat=False):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = PatchEmbedOverlapping(patch_size=7,
                                                 stride=4,
                                                 padding=2,
                                                 in_chans=3,
                                                 embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(dim=embed_dims[i],
                                 index=i,
                                 layers=layers,
                                 mlp_ratio=mlp_ratios[i],
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer,
                                 skip_lam=skip_lam,
                                 mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))

        self.network = nn.LayerList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    layer = Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_layer(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.transpose([0, 2, 3, 1])
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out.transpose([0, 3, 1, 2]))
        if self.fork_feat:
            return outs

        B, H, W, C = x.shape
        x = x.reshape([B, -1, C])
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


def build_cyclemlp(config):
    '''build cyclemlp model'''
    model = CycleNet(num_classes=config.MODEL.NUM_CLASSES,
                     layers=config.MODEL.LAYERS,
                     embed_dims=config.MODEL.EMBED_DIMS,
                     patch_size=config.MODEL.PATCH_SIZE,
                     transitions=config.MODEL.TRANSITIONS,
                     mlp_ratios=config.MODEL.MLP_RATIOS,
                     mlp_fn=CycleMLP)
    return model
