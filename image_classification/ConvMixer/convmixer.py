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
ConvMixer in Paddle
A Paddle Implementation of ConvMixer as described in:
"Patches Are All You Need?"
    - Paper Link: https://arxiv.org/abs/2201.09792
"""

import paddle
import paddle.nn as nn


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim,
              depth,
              kernel_size=9,
              patch_size=7,
              num_classes=1000,
              activation='GELU'):
    if activation == 'ReLU':
        convmixer_act = nn.ReLU()
    else:
        convmixer_act = nn.GELU()

    return nn.Sequential(
        nn.Conv2D(3, dim, kernel_size=patch_size, stride=patch_size),
        convmixer_act,
        nn.BatchNorm2D(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2D(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                        convmixer_act,
                        nn.BatchNorm2D(dim),
                    )
                ),
                nn.Conv2D(dim, dim, kernel_size=1),
                convmixer_act,
                nn.BatchNorm2D(dim),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2D((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    )


def build_convmixer(config):
    model = ConvMixer(
        dim=config.MODEL.CNN.DIM,
        depth=config.MODEL.CNN.DEPTH,
        kernel_size=config.MODEL.CNN.KERNEL_SIZE,
        patch_size=config.MODEL.CNN.PATCH_SIZE,
        num_classes=config.MODEL.NUM_CLASSES,
        activation=config.MODEL.CNN.ACTIVATION,
    )
    return model
