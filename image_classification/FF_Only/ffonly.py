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
Implement MLP Class for FF_Only
"""

from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import nn

from droppath import DropPath

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)
kaiming_normal_ = nn.initializer.KaimingNormal()


class Identity(nn.Layer):
    """Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    """MLP module

    MLP using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> dwconv -> act -> dropout -> fc -> dropout

    Args:
        in_features (int): input features.
        hidden_features (int): hidden features.
        out_features (int): output features.
        act_layer (nn.Layer): activation.
        drop (float): dropout.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):

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


class LinearBlock(nn.Layer):
    """Basic model components"""

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_tokens=197,
    ):
        super().__init__()

        # First stage
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm1 = norm_layer(dim)

        # Second stage
        self.mlp2 = Mlp(
            in_features=num_tokens,
            hidden_features=int(num_tokens * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(num_tokens)

        # Dropout (or a variant)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        x = x.transpose([0, 2, 1])
        x = x + self.drop_path(self.mlp2(self.norm2(x)))
        x = x.transpose([0, 2, 1])
        return x


class PatchEmbed(nn.Layer):
    """Wraps a convolution"""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class LearnedPositionalEncoding(nn.Layer):
    """Learned positional encoding with dynamic interpolation at runtime"""

    def __init__(self, height, width, embed_dim):
        super().__init__()
        self.height = height
        self.width = width

        self.pos_embed = self.create_parameter(
            shape=[1, embed_dim, height, width], default_initializer=trunc_normal_
        )
        self.add_parameter("pos_embed", self.pos_embed)

        self.cls_pos_embed = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=trunc_normal_
        )
        self.add_parameter("cls_pos_embed", self.cls_pos_embed)

    def forward(self, x):
        _, _, H, W = x.shape
        if H == self.height and W == self.width:
            pos_embed = self.pos_embed
        else:
            pos_embed = F.interpolate(
                self.pos_embed, size=[H, W], mode="bilinear", align_corners=False
            )
        return self.cls_pos_embed, pos_embed


class LinearVisionTransformer(nn.Layer):
    """
    Basically the same as the standard Vision Transformer, but with support for resizable
    or sinusoidal positional embeddings.
    """

    def __init__(
        self,
        *,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        positional_encoding="learned",
        learned_positional_encoding_size=(14, 14),
        block_cls=LinearBlock
    ):
        super().__init__()

        # Config
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        # Class token
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=trunc_normal_
        )
        self.add_parameter("cls_token", self.cls_token)

        # Positional encoding
        if positional_encoding == "learned":
            (
                height,
                width,
            ) = self.learned_positional_encoding_size = learned_positional_encoding_size
            self.pos_encoding = LearnedPositionalEncoding(height, width, embed_dim)
        else:
            raise NotImplementedError("Unsupposed positional encoding")
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList(
            [
                block_cls(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_tokens=1 + (224 // patch_size) ** 2,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()
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

        # Patch embedding
        B, _, _, _ = x.shape  # B x C x H x W
        x = self.patch_embed(x)  # B x E x H//p x W//p

        # Positional encoding
        # NOTE: cls_pos_embed for compatibility with pretrained models
        cls_pos_embed, pos_embed = self.pos_encoding(x)

        # Flatten image, append class token, add positional encoding
        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = x.flatten(2).transpose([0, 2, 1])  # flatten
        x = paddle.concat((cls_tokens, x), axis=1)  # class token
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])  # flatten
        pos_embed = paddle.concat([cls_pos_embed, pos_embed], axis=1)  # class pos emb
        x = x + pos_embed
        x = self.pos_drop(x)

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        # Final layernorm
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_ffonly(config):
    model = LinearVisionTransformer(
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.MIXER.EMBED_DIM,
    )
    return model
