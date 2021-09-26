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
Implement MLP Class for ConvMLP
"""

import paddle
import paddle.nn as nn

from droppath import DropPath

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)
kaiming_normal_ = nn.initializer.KaimingNormal()


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class ConvTokenizer(nn.Layer):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2D(
                3,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias_attr=False,
            ),
            nn.BatchNorm2D(embedding_dim // 2),
            nn.ReLU(),
            nn.Conv2D(
                embedding_dim // 2,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias_attr=False,
            ),
            nn.BatchNorm2D(embedding_dim // 2),
            nn.ReLU(),
            nn.Conv2D(
                embedding_dim // 2,
                embedding_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias_attr=False,
            ),
            nn.BatchNorm2D(embedding_dim),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, x):
        return self.block(x)


class ConvStage(nn.Layer):
    def __init__(
        self, num_blocks=2, embedding_dim_in=64, hidden_dim=128, embedding_dim_out=128
    ):
        super().__init__()
        self.conv_blocks = nn.LayerList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2D(
                    embedding_dim_in,
                    hidden_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias_attr=False,
                ),
                nn.BatchNorm2D(hidden_dim),
                nn.ReLU(),
                nn.Conv2D(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias_attr=False,
                ),
                nn.BatchNorm2D(hidden_dim),
                nn.ReLU(),
                nn.Conv2D(
                    hidden_dim,
                    embedding_dim_in,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias_attr=False,
                ),
                nn.BatchNorm2D(embedding_dim_in),
                nn.ReLU(),
            )
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2D(
            embedding_dim_in,
            embedding_dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class Mlp(nn.Layer):
    def __init__(
        self,
        embedding_dim_in,
        hidden_dim=None,
        embedding_dim_out=None,
        activation=nn.GELU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvMLPStage(nn.Layer):
    def __init__(self, embedding_dim, dim_feedforward=2048, stochastic_depth_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(
            embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.connect = nn.Conv2D(
            embedding_dim,
            embedding_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=embedding_dim,
            bias_attr=False,
        )
        self.connect_norm = nn.LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(
            embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward
        )
        self.drop_path = (
            DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0 else Identity()
        )

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect(self.connect_norm(src).transpose([0, 3, 1, 2])).transpose(
            [0, 2, 3, 1]
        )
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class ConvDownsample(nn.Layer):
    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = nn.Conv2D(
            embedding_dim_in,
            embedding_dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        x = self.downsample(x)
        return x.transpose([0, 2, 3, 1])


class BasicStage(nn.Layer):
    def __init__(
        self,
        num_blocks,
        embedding_dims,
        mlp_ratio=1,
        stochastic_depth_rate=0.1,
        downsample=True,
    ):
        super().__init__()
        self.blocks = nn.LayerList()
        dpr = [x.item() for x in paddle.linspace(0, stochastic_depth_rate, num_blocks)]
        for i in range(num_blocks):
            block = ConvMLPStage(
                embedding_dim=embedding_dims[0],
                dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                stochastic_depth_rate=dpr[i],
            )
            self.blocks.append(block)

        self.downsample_mlp = (
            ConvDownsample(embedding_dims[0], embedding_dims[1])
            if downsample
            else Identity()
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x


class ConvMLP(nn.Layer):
    def __init__(
        self,
        blocks,
        dims,
        mlp_ratios,
        channels=64,
        n_conv_blocks=3,
        classifier_head=True,
        num_classes=1000,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert (
            len(blocks) == len(mlp_ratios) == len(mlp_ratios)
        ), f"blocks, dims and mlp_ratios must agree in size, {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(
            n_conv_blocks,
            embedding_dim_in=channels,
            hidden_dim=dims[0],
            embedding_dim_out=dims[0],
        )

        self.stages = nn.LayerList()
        for i in range(0, len(blocks)):
            stage = BasicStage(
                num_blocks=blocks[i],
                embedding_dims=dims[i : i + 2],
                mlp_ratio=mlp_ratios[i],
                stochastic_depth_rate=0.1,
                downsample=(i + 1 < len(blocks)),
            )
            self.stages.append(stage)
        if classifier_head:
            self.norm = nn.LayerNorm(dims[-1])
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.head = None
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        x = x.transpose([0, 2, 3, 1])
        for stage in self.stages:
            x = stage(x)
        if self.head is None:
            return x
        B, _, _, C = x.shape
        x = x.reshape([B, -1, C])
        x = self.norm(x)
        x = x.mean(axis=1)
        x = self.head(x)
        return x

    def init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1D)):
            trunc_normal_(m.weight)
            if isinstance(m, (nn.Linear, nn.Conv1D)) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2D):
            ones_(m.weight)
            zeros_(m.bias)


def build_convmlp(config):
    model = ConvMLP(
        blocks=config.MODEL.MIXER.BLOCKS,
        dims=config.MODEL.MIXER.DIMS,
        mlp_ratios=config.MODEL.MIXER.MLP_RATIOS,
        channels=config.MODEL.MIXER.CHANNELS,
        n_conv_blocks=config.MODEL.MIXER.N_CONV_BLOCKS,
        classifier_head=True,
        num_classes=config.MODEL.NUM_CLASSES,
    )
    return model


def convmlp_m(**kwargs):
    model = ConvMLP(
        blocks=[3, 6, 3],
        dims=[128, 256, 512],
        mlp_ratios=[3, 3, 3],
        channels=64,
        n_conv_blocks=3,
        classifier_head=True,
        num_classes=1000,
    )
    return model


def convmlp_l(**kwargs):
    model = ConvMLP(
        blocks=[4, 8, 3],
        dims=[192, 384, 768],
        mlp_ratios=[3, 3, 3],
        channels=96,
        n_conv_blocks=3,
        classifier_head=True,
        num_classes=1000,
    )
    return model
