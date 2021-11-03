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
Implement MLP Class for ViP
"""

import paddle.nn as nn
import paddle.nn.functional as F
from droppath import DropPath


trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class Mlp(nn.Layer):
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


class WeightedPermuteMLP(nn.Layer):
    def __init__(self,
                 dim,
                 segment_dim=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = (
            x.reshape([B, H, W, self.segment_dim, S])
            .transpose([0, 3, 2, 1, 4])
            .reshape([B, self.segment_dim, W, H * S])
        )
        h = (
            self.mlp_h(h)
            .reshape([B, self.segment_dim, W, H, S])
            .transpose([0, 3, 2, 1, 4])
            .reshape([B, H, W, C])
        )

        w = (
            x.reshape([B, H, W, self.segment_dim, S])
            .transpose([0, 1, 3, 2, 4])
            .reshape([B, H, self.segment_dim, W * S])
        )
        w = (
            self.mlp_w(w)
            .reshape([B, H, self.segment_dim, W, S])
            .transpose([0, 1, 3, 2, 4])
            .reshape([B, H, W, C])
        )

        c = self.mlp_c(x)

        a = (h + w + c).transpose([0, 3, 1, 2]).flatten(2).mean(2)
        a = self.reweight(a).reshape([B, C, 3]).transpose([2, 0, 1])
        a = F.softmax(a, axis=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PermutatorBlock(nn.Layer):
    def __init__(self,
                 dim,
                 segment_dim,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 skip_lam=1.0,
                 mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(
            dim,
            segment_dim=segment_dim,
            qkv_bias=qkv_bias,
            qk_scale=None,
            attn_drop=attn_drop,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Layer):
    """Image to Patch Embedding"""
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2D(in_embed_dim,
                              out_embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        x = self.proj(x)  # B, C, H, W
        x = x.transpose([0, 2, 3, 1])
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 segment_dim,
                 mlp_ratio=3.0,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0,
                 drop_path_rate=0.0,
                 skip_lam=1.0,
                 mlp_fn=WeightedPermuteMLP,
                 **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            PermutatorBlock(
                dim,
                segment_dim,
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


class VisionPermutator(nn.Layer):
    """Vision Permutator"""
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
                 mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i],
                                 i,
                                 layers,
                                 segment_dim[i],
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
        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = (
            nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.transpose([0, 2, 3, 1])
        return x

    def forward_tokens(self, x):
        for _, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape([B, -1, C])
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))


def build_vip(config):
    """build vip model using config"""
    model = VisionPermutator(num_classes=config.MODEL.NUM_CLASSES,
                             layers=config.MODEL.MIXER.LAYER,
                             embed_dims=config.MODEL.MIXER.EMBED_DIMS,
                             patch_size=7,
                             transitions=config.MODEL.MIXER.TRANSITIONS,
                             segment_dim=config.MODEL.MIXER.SEGMENT_DIM,
                             mlp_ratios=[3, 3, 3, 3],
                             mlp_fn=WeightedPermuteMLP)
    return model
